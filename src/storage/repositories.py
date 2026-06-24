"""
Repository layer for Postgres-backed Second Brain persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vector_literal(vector_values: List[float]) -> str:
    # pgvector accepts bracket vector syntax, e.g. "[0.1,0.2,...]"
    return "[" + ",".join(f"{float(value):.8f}" for value in vector_values) + "]"


class SecondBrainRepository:
    """DB-backed storage adapter for documents, chunks, chats and decisions."""

    def __init__(self, client: PostgresClient):
        self.client = client

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def ensure_ready(self) -> bool:
        if not self.enabled:
            return False
        if not self.client.test_connection():
            return False
        return self.client.initialize_schema()

    def get_migration_status(self) -> Dict[str, Any]:
        if not self.enabled or not hasattr(self.client, "get_migration_status"):
            return {"enabled": False, "applied": [], "pending": [], "error": None}
        return self.client.get_migration_status()

    def upsert_collection(self, name: str, collection_type: str = "topic") -> Optional[int]:
        if not self.enabled:
            return None

        collection_name = (name or "").strip()
        if not collection_name:
            return None

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO collections(name, type, created_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT(name) DO UPDATE
                          SET type = EXCLUDED.type
                        RETURNING id;
                        """,
                        (collection_name, collection_type),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("upsert_collection failed for %s: %s", collection_name, exc)
            return None

    def upsert_document(
        self,
        file_hash: str,
        original_name: str,
        collection_name: str,
        source_path: str,
        indexed: bool = False,
    ) -> Optional[int]:
        if not self.enabled:
            return None

        collection_type = "general" if (collection_name or "").strip().lower() in {"general", "default"} else "topic"
        collection_id = self.upsert_collection(collection_name, collection_type=collection_type)
        if collection_id is None:
            return None

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO documents(file_hash, original_name, collection_id, source_path, indexed, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT(file_hash) DO UPDATE SET
                            original_name = EXCLUDED.original_name,
                            collection_id = EXCLUDED.collection_id,
                            source_path = EXCLUDED.source_path,
                            indexed = EXCLUDED.indexed,
                            updated_at = NOW()
                        RETURNING id;
                        """,
                        (file_hash, original_name, collection_id, source_path, indexed),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("upsert_document failed for %s: %s", original_name, exc)
            return None

    def replace_document_chunks(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[Any] = None,
    ) -> bool:
        if not self.enabled:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    # Remove previous rows for idempotent re-index.
                    cur.execute(
                        "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = %s);",
                        (document_id,),
                    )
                    cur.execute("DELETE FROM chunks WHERE document_id = %s;", (document_id,))

                    chunk_ids: List[int] = []
                    for idx, chunk in enumerate(chunks):
                        content = (chunk.get("content", "") or "").replace("\x00", "")
                        metadata = chunk.get("metadata", {})
                        token_count = max(1, len(content) // 4)
                        cur.execute(
                            """
                            INSERT INTO chunks(document_id, chunk_order, text, token_count, metadata_json, created_at)
                            VALUES (%s, %s, %s, %s, %s, NOW())
                            RETURNING id;
                            """,
                            (document_id, idx, content, token_count, json.dumps(metadata, ensure_ascii=False)),
                        )
                        row = cur.fetchone()
                        if row:
                            chunk_ids.append(int(row[0]))

                    if embeddings is not None and len(chunk_ids) > 0:
                        embedding_rows: List[Any] = []
                        for emb_idx, chunk_id in enumerate(chunk_ids):
                            vector = embeddings[emb_idx]
                            if hasattr(vector, "tolist"):
                                vector = vector.tolist()
                            embedding_rows.append((chunk_id, _vector_literal(vector)))

                        cur.executemany(
                            """
                            INSERT INTO chunk_embeddings(chunk_id, embedding, created_at)
                            VALUES (%s, %s::vector, NOW())
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                created_at = NOW();
                            """,
                            embedding_rows,
                        )

                    cur.execute("UPDATE documents SET indexed = TRUE, updated_at = NOW() WHERE id = %s;", (document_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("replace_document_chunks failed for document_id=%s: %s", document_id, exc)
            return False

    def ingest_processed_documents(
        self,
        file_paths: List[str],
        processed_docs: List[Dict[str, Any]],
        embeddings: Optional[Any] = None,
    ) -> bool:
        """Persist processed docs/chunks grouped by source_path."""
        if not self.enabled:
            return False

        docs_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in processed_docs:
            source_path = chunk.get("metadata", {}).get("source_path", "")
            if not source_path:
                continue
            docs_by_source.setdefault(source_path, []).append(chunk)

        if not docs_by_source:
            return False

        # Build embedding slices in source order.
        embedding_cursor = 0
        try:
            for source_path in file_paths:
                source_chunks = docs_by_source.get(source_path, [])
                if not source_chunks:
                    continue

                file_hash = self._hash_file(source_path)
                first_meta = source_chunks[0].get("metadata", {})
                doc_name = first_meta.get("filename") or source_path.split("\\")[-1].split("/")[-1]
                collection = first_meta.get("collection", "general")

                doc_id = self.upsert_document(
                    file_hash=file_hash,
                    original_name=doc_name,
                    collection_name=collection,
                    source_path=source_path,
                    indexed=True,
                )
                if doc_id is None:
                    continue

                source_embeddings = None
                if embeddings is not None:
                    source_count = len(source_chunks)
                    source_embeddings = embeddings[embedding_cursor:embedding_cursor + source_count]
                    embedding_cursor += source_count

                self.replace_document_chunks(doc_id, source_chunks, source_embeddings)

            return True
        except Exception as exc:
            logger.error("ingest_processed_documents failed: %s", exc)
            return False

    def vector_search(
        self,
        query_embedding: List[float],
        collection_filters: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Run pgvector retrieval directly from DB."""
        if not self.enabled:
            return []

        try:
            vector_sql = _vector_literal(query_embedding)
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if collection_filters:
                        cur.execute(
                            """
                            SELECT
                                chunks.id,
                                chunks.text,
                                chunks.metadata_json,
                                documents.original_name,
                                collections.name AS collection_name,
                                (1 - (chunk_embeddings.embedding <=> %s::vector)) AS similarity
                            FROM chunk_embeddings
                            JOIN chunks ON chunks.id = chunk_embeddings.chunk_id
                            JOIN documents ON documents.id = chunks.document_id
                            JOIN collections ON collections.id = documents.collection_id
                            WHERE collections.name = ANY(%s)
                            ORDER BY chunk_embeddings.embedding <=> %s::vector
                            LIMIT %s;
                            """,
                            (vector_sql, collection_filters, vector_sql, int(top_k)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                chunks.id,
                                chunks.text,
                                chunks.metadata_json,
                                documents.original_name,
                                collections.name AS collection_name,
                                (1 - (chunk_embeddings.embedding <=> %s::vector)) AS similarity
                            FROM chunk_embeddings
                            JOIN chunks ON chunks.id = chunk_embeddings.chunk_id
                            JOIN documents ON documents.id = chunks.document_id
                            JOIN collections ON collections.id = documents.collection_id
                            ORDER BY chunk_embeddings.embedding <=> %s::vector
                            LIMIT %s;
                            """,
                            (vector_sql, vector_sql, int(top_k)),
                        )
                    rows = cur.fetchall()
                conn.commit()

            results: List[Dict[str, Any]] = []
            for row in rows:
                metadata = row[2] or {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                metadata.update(
                    {
                        "filename": row[3],
                        "collection": row[4],
                        "db_chunk_id": row[0],
                    }
                )
                results.append(
                    {
                        "content": row[1],
                        "metadata": metadata,
                        "retrieval_score": float(row[5]),
                        "semantic_score": float(row[5]),
                        "keyword_score": 0.0,
                    }
                )
            return results
        except Exception as exc:
            logger.error("vector_search failed: %s", exc)
            return []

    def list_indexed_chunks(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """Return indexed chunks in the same shape expected by graph ingestion."""
        if not self.enabled:
            return []

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            chunks.id,
                            chunks.text,
                            chunks.chunk_order,
                            chunks.metadata_json,
                            documents.id,
                            documents.source_path,
                            documents.original_name,
                            collections.name AS collection_name
                        FROM chunks
                        JOIN documents ON documents.id = chunks.document_id
                        JOIN collections ON collections.id = documents.collection_id
                        WHERE documents.indexed = TRUE
                        ORDER BY documents.id ASC, chunks.chunk_order ASC
                        LIMIT %s;
                        """,
                        (int(limit),),
                    )
                    rows = cur.fetchall() or []
                conn.commit()

            chunks: List[Dict[str, Any]] = []
            for row in rows:
                metadata = row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {})
                metadata.update(
                    {
                        "db_chunk_id": row[0],
                        "chunk_id": row[2],
                        "doc_id": row[4],
                        "source_path": row[5] or "",
                        "filename": row[6] or "document",
                        "collection": row[7] or "general",
                    }
                )
                chunks.append({"content": row[1] or "", "metadata": metadata})
            return chunks
        except Exception as exc:
            logger.error("list_indexed_chunks failed: %s", exc)
            return []

    def save_chat_message(
        self,
        chat_title: str,
        topic_collection: str,
        query_mode: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        answer_origin: str = "internal",
    ) -> bool:
        if not self.enabled:
            return False

        session = self.create_session(
            title=chat_title,
            topic_collection=topic_collection,
            query_mode=query_mode,
        )
        session_id = session.get("id")
        if not session_id:
            return False

        if role == "assistant":
            exchange = {
                "question": "",
                "response": content,
                "sources": sources or [],
                "graph_sources": [],
                "memory_hits": [],
                "provenance": [],
                "external_sources": [],
                "response_time": 0.0,
                "cached": False,
                "model_used": "",
                "answer_origin": answer_origin,
                "retrieval_mode": query_mode,
                "route_used": "vector",
                "router_reason": "",
            }
            return self.append_exchange(session_id, exchange)

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO messages(chat_id, role, content, sources_json, answer_origin, created_at)
                        VALUES (%s, %s, %s, %s, %s, NOW());
                        """,
                        (int(session_id), "user", content, json.dumps([], ensure_ascii=False), "internal"),
                    )
                    cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s;", (int(session_id),))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("save_chat_message failed: %s", exc)
            return False

    @staticmethod
    def _normalize_chat_id(session_id: Any) -> Optional[int]:
        raw_value = str(session_id or "").strip()
        if not raw_value:
            return None
        if raw_value.isdigit():
            return int(raw_value)
        return None

    @staticmethod
    def _iso(value: Any) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value) if value is not None else _now_iso()

    @staticmethod
    def _normalize_metadata_json(raw_value: Any) -> Dict[str, Any]:
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def is_healthy(self) -> bool:
        return bool(self.enabled and self.client.test_connection())

    def create_session(
        self,
        title: str = "Chat nou",
        topic_collection: str = "",
        query_mode: str = "topic_general",
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        cleaned_title = (title or "Chat nou").strip() or "Chat nou"
        cleaned_topic = (topic_collection or "").strip()
        cleaned_mode = (query_mode or "topic_general").strip() or "topic_general"

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return {}
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chats(title, topic_collection, query_mode, created_at, updated_at)
                        VALUES (%s, %s, %s, NOW(), NOW())
                        RETURNING id, created_at, updated_at;
                        """,
                        (cleaned_title, cleaned_topic, cleaned_mode),
                    )
                    row = cur.fetchone()
                conn.commit()

            if not row:
                return {}

            return {
                "id": str(row[0]),
                "title": cleaned_title,
                "topic_collection": cleaned_topic,
                "query_mode": cleaned_mode,
                "created_at": self._iso(row[1]),
                "updated_at": self._iso(row[2]),
                "message_count": 0,
            }
        except Exception as exc:
            logger.error("create_session failed: %s", exc)
            return {}

    def list_sessions(self, topic_collection: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        topic_filter = (topic_collection or "").strip()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic_filter:
                        cur.execute(
                            """
                            SELECT
                                c.id,
                                c.title,
                                c.topic_collection,
                                c.query_mode,
                                c.created_at,
                                c.updated_at,
                                COUNT(m.id) FILTER (WHERE m.role = 'assistant') AS message_count
                            FROM chats c
                            LEFT JOIN messages m ON m.chat_id = c.id
                            WHERE c.topic_collection = %s
                            GROUP BY c.id
                            ORDER BY c.updated_at DESC, c.id DESC;
                            """,
                            (topic_filter,),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                c.id,
                                c.title,
                                c.topic_collection,
                                c.query_mode,
                                c.created_at,
                                c.updated_at,
                                COUNT(m.id) FILTER (WHERE m.role = 'assistant') AS message_count
                            FROM chats c
                            LEFT JOIN messages m ON m.chat_id = c.id
                            GROUP BY c.id
                            ORDER BY c.updated_at DESC, c.id DESC;
                            """
                        )
                    rows = cur.fetchall() or []
                conn.commit()

            sessions: List[Dict[str, Any]] = []
            for row in rows:
                sessions.append(
                    {
                        "id": str(row[0]),
                        "title": row[1] or "Chat nou",
                        "topic_collection": row[2] or "",
                        "query_mode": row[3] or "topic_general",
                        "created_at": self._iso(row[4]),
                        "updated_at": self._iso(row[5]),
                        "message_count": int(row[6] or 0),
                    }
                )
            return sessions
        except Exception as exc:
            logger.error("list_sessions failed: %s", exc)
            return []

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        numeric_id = self._normalize_chat_id(session_id)
        if numeric_id is None:
            return None

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, topic_collection, query_mode, created_at, updated_at
                        FROM chats
                        WHERE id = %s;
                        """,
                        (numeric_id,),
                    )
                    chat_row = cur.fetchone()
                    if not chat_row:
                        conn.commit()
                        return None

                    cur.execute(
                        """
                        SELECT role, content, sources_json, answer_origin, created_at, id
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY created_at ASC, id ASC;
                        """,
                        (numeric_id,),
                    )
                    message_rows = cur.fetchall() or []
                conn.commit()

            messages: List[Dict[str, Any]] = []
            pending_question = ""
            for row in message_rows:
                role = (row[0] or "").strip().lower()
                content = row[1] or ""
                metadata = self._normalize_metadata_json(row[2])
                answer_origin = row[3] or metadata.get("answer_origin", "internal")

                if role == "user":
                    pending_question = content
                    continue

                if role != "assistant":
                    continue

                question = metadata.get("question") or pending_question or ""
                exchange = {
                    "question": question,
                    "response": content,
                    "sources": metadata.get("sources", []),
                    "graph_sources": metadata.get("graph_sources", []),
                    "memory_hits": metadata.get("memory_hits", []),
                    "memory_sources": metadata.get("memory_sources", []),
                    "provenance": metadata.get("provenance", []),
                    "citation_map": metadata.get("citation_map", {}),
                    "retrieval_metrics": metadata.get("retrieval_metrics", {}),
                    "external_sources": metadata.get("external_sources", []),
                    "response_time": metadata.get("response_time", 0.0),
                    "cached": bool(metadata.get("cached", False)),
                    "model_used": metadata.get("model_used", ""),
                    "answer_origin": answer_origin,
                    "retrieval_mode": metadata.get("retrieval_mode", "topic_general"),
                    "route_used": metadata.get("route_used", "vector"),
                    "router_reason": metadata.get("router_reason", ""),
                }
                messages.append(exchange)
                pending_question = ""

            return {
                "id": str(chat_row[0]),
                "title": chat_row[1] or "Chat nou",
                "topic_collection": chat_row[2] or "",
                "query_mode": chat_row[3] or "topic_general",
                "created_at": self._iso(chat_row[4]),
                "updated_at": self._iso(chat_row[5]),
                "messages": messages,
            }
        except Exception as exc:
            logger.error("load_session failed: %s", exc)
            return None

    def _insert_exchange_messages(self, cursor: Any, chat_id: int, exchange: Dict[str, Any]) -> None:
        question = (exchange.get("question") or "").strip()
        response = exchange.get("response", "")
        answer_origin = exchange.get("answer_origin", "internal")

        payload = {
            "question": question,
            "sources": exchange.get("sources", []),
            "graph_sources": exchange.get("graph_sources", []),
            "memory_hits": exchange.get("memory_hits", []),
            "memory_sources": exchange.get("memory_sources", []),
            "provenance": exchange.get("provenance", []),
            "citation_map": exchange.get("citation_map", {}),
            "retrieval_metrics": exchange.get("retrieval_metrics", {}),
            "external_sources": exchange.get("external_sources", []),
            "response_time": exchange.get("response_time", 0.0),
            "cached": bool(exchange.get("cached", False)),
            "model_used": exchange.get("model_used", ""),
            "answer_origin": answer_origin,
            "retrieval_mode": exchange.get("retrieval_mode", "topic_general"),
            "route_used": exchange.get("route_used", "vector"),
            "router_reason": exchange.get("router_reason", ""),
        }

        cursor.execute(
            """
            INSERT INTO messages(chat_id, role, content, sources_json, answer_origin, created_at)
            VALUES (%s, 'user', %s, %s, 'internal', NOW());
            """,
            (chat_id, question, json.dumps({}, ensure_ascii=False)),
        )
        cursor.execute(
            """
            INSERT INTO messages(chat_id, role, content, sources_json, answer_origin, created_at)
            VALUES (%s, 'assistant', %s, %s, %s, NOW());
            """,
            (chat_id, response, json.dumps(payload, ensure_ascii=False), answer_origin),
        )

    def append_exchange(self, session_id: str, exchange: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        numeric_id = self._normalize_chat_id(session_id)
        if numeric_id is None:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    self._insert_exchange_messages(cur, numeric_id, exchange)
                    cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s;", (numeric_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("append_exchange failed: %s", exc)
            return False

    def save_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
        topic_collection: Optional[str] = None,
        query_mode: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return False

        numeric_id = self._normalize_chat_id(session_id)
        if numeric_id is None:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    if title is not None:
                        cur.execute("UPDATE chats SET title = %s WHERE id = %s;", ((title or "Chat nou").strip() or "Chat nou", numeric_id))
                    if topic_collection is not None:
                        cur.execute("UPDATE chats SET topic_collection = %s WHERE id = %s;", ((topic_collection or "").strip(), numeric_id))
                    if query_mode is not None:
                        cur.execute("UPDATE chats SET query_mode = %s WHERE id = %s;", ((query_mode or "topic_general").strip() or "topic_general", numeric_id))
                    cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s;", (numeric_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("save_session failed: %s", exc)
            return False

    def clear_session(self, session_id: str) -> bool:
        if not self.enabled:
            return False

        numeric_id = self._normalize_chat_id(session_id)
        if numeric_id is None:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM messages WHERE chat_id = %s;", (numeric_id,))
                    cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s;", (numeric_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("clear_session failed: %s", exc)
            return False

    def rename_session(self, session_id: str, new_title: str) -> bool:
        return self.save_session(session_id, [], title=new_title)

    def delete_session(self, session_id: str) -> bool:
        if not self.enabled:
            return False

        numeric_id = self._normalize_chat_id(session_id)
        if numeric_id is None:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chats WHERE id = %s;", (numeric_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
            return deleted
        except Exception as exc:
            logger.error("delete_session failed: %s", exc)
            return False

    def save_decision(
        self,
        title: str,
        rationale: str,
        topic_collection: str,
        confidence: float,
        source_message_id: Optional[int] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO decisions(title, rationale, topic_collection, confidence, source_message_id, created_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        RETURNING id;
                        """,
                        (title, rationale, topic_collection, float(confidence), source_message_id),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("save_decision failed: %s", exc)
            return None

    def search_decisions(
        self,
        question: str,
        topic_collection: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        query_text = (question or "").strip().lower()
        if not query_text:
            return []

        terms = [token for token in query_text.split() if len(token) >= 3]
        if not terms:
            return []

        pattern = "|" + "|".join(terms) + "|"
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic_collection:
                        cur.execute(
                            """
                            SELECT id, title, rationale, topic_collection, confidence, created_at
                            FROM decisions
                            WHERE topic_collection = %s
                              AND (LOWER(title) SIMILAR TO %s OR LOWER(rationale) SIMILAR TO %s)
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT %s;
                            """,
                            (topic_collection, pattern, pattern, int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, title, rationale, topic_collection, confidence, created_at
                            FROM decisions
                            WHERE LOWER(title) SIMILAR TO %s OR LOWER(rationale) SIMILAR TO %s
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT %s;
                            """,
                            (pattern, pattern, int(limit)),
                        )
                    rows = cur.fetchall()
                conn.commit()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "rationale": row[2],
                    "topic_collection": row[3],
                    "confidence": float(row[4]),
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                    "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                    "memory_type": "semantic",
                    "source": "decision",
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("search_decisions failed: %s", exc)
            return []

    def upsert_preference(
        self,
        preference_key: str,
        preference_value: str,
        topic_collection: str = "",
        confidence: float = 0.8,
        source_message_id: Optional[int] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None

        key = (preference_key or "").strip().lower()
        value = (preference_value or "").strip()
        topic = (topic_collection or "").strip()
        if not key or not value:
            return None

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO preferences(
                            preference_key,
                            preference_value,
                            topic_collection,
                            confidence,
                            source_message_id,
                            created_at,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT(preference_key, topic_collection)
                        DO UPDATE SET
                            preference_value = EXCLUDED.preference_value,
                            confidence = EXCLUDED.confidence,
                            source_message_id = EXCLUDED.source_message_id,
                            updated_at = NOW()
                        RETURNING id;
                        """,
                        (key, value, topic, float(confidence), source_message_id),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("upsert_preference failed: %s", exc)
            return None

    def search_preferences(
        self,
        question: str,
        topic_collection: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        terms = self._tokenize_query_terms(question)
        if not terms:
            return []

        like_terms = [f"%{term}%" for term in terms]
        topic = (topic_collection or "").strip()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic:
                        cur.execute(
                            """
                            SELECT id, preference_key, preference_value, topic_collection, confidence, updated_at
                            FROM preferences
                            WHERE topic_collection = %s
                              AND (
                                LOWER(preference_key) LIKE ANY(%s)
                                OR LOWER(preference_value) LIKE ANY(%s)
                              )
                            ORDER BY confidence DESC, updated_at DESC
                            LIMIT %s;
                            """,
                            (topic, like_terms, like_terms, int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, preference_key, preference_value, topic_collection, confidence, updated_at
                            FROM preferences
                            WHERE LOWER(preference_key) LIKE ANY(%s)
                               OR LOWER(preference_value) LIKE ANY(%s)
                            ORDER BY confidence DESC, updated_at DESC
                            LIMIT %s;
                            """,
                            (like_terms, like_terms, int(limit)),
                        )
                    rows = cur.fetchall() or []
                conn.commit()

            return [
                {
                    "id": row[0],
                    "title": f"Preferinta: {row[1]}",
                    "rationale": row[2],
                    "topic_collection": row[3],
                    "confidence": float(row[4]),
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                    "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                    "memory_type": "procedural",
                    "source": "preference",
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("search_preferences failed: %s", exc)
            return []

    def search_episodes(
        self,
        question: str,
        topic_collection: Optional[str] = None,
        limit: int = 5,
        time_window_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        terms = self._tokenize_query_terms(question)
        if not terms:
            return []

        like_terms = [f"%{term}%" for term in terms]
        topic = (topic_collection or "").strip()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic and time_window_days:
                        cur.execute(
                            """
                            SELECT m.id, m.content, m.sources_json, m.created_at, c.topic_collection
                            FROM messages m
                            JOIN chats c ON c.id = m.chat_id
                            WHERE m.role = 'assistant'
                              AND c.topic_collection = %s
                              AND m.created_at >= NOW() - (%s || ' days')::interval
                            ORDER BY m.created_at DESC
                            LIMIT 150;
                            """,
                            (topic, int(time_window_days)),
                        )
                    elif topic:
                        cur.execute(
                            """
                            SELECT m.id, m.content, m.sources_json, m.created_at, c.topic_collection
                            FROM messages m
                            JOIN chats c ON c.id = m.chat_id
                            WHERE m.role = 'assistant'
                              AND c.topic_collection = %s
                            ORDER BY m.created_at DESC
                            LIMIT 150;
                            """,
                            (topic,),
                        )
                    elif time_window_days:
                        cur.execute(
                            """
                            SELECT m.id, m.content, m.sources_json, m.created_at, c.topic_collection
                            FROM messages m
                            JOIN chats c ON c.id = m.chat_id
                            WHERE m.role = 'assistant'
                              AND m.created_at >= NOW() - (%s || ' days')::interval
                            ORDER BY m.created_at DESC
                            LIMIT 150;
                            """,
                            (int(time_window_days),),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT m.id, m.content, m.sources_json, m.created_at, c.topic_collection
                            FROM messages m
                            JOIN chats c ON c.id = m.chat_id
                            WHERE m.role = 'assistant'
                            ORDER BY m.created_at DESC
                            LIMIT 150;
                            """
                        )
                    rows = cur.fetchall() or []
                conn.commit()

            scored: List[Dict[str, Any]] = []
            for row in rows:
                metadata = self._normalize_metadata_json(row[2])
                episode_question = (metadata.get("question") or "").lower()
                episode_answer = (row[1] or "").lower()
                haystack = f"{episode_question} {episode_answer}"
                matches = sum(1 for term in terms if term in haystack)
                if matches == 0:
                    continue
                confidence = min(0.95, 0.45 + 0.1 * matches)
                scored.append(
                    {
                        "id": row[0],
                        "title": (metadata.get("question") or "Memorie episodica")[:140],
                        "rationale": row[1][:900] if isinstance(row[1], str) else "",
                        "topic_collection": row[4] or "",
                        "confidence": confidence,
                        "created_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                        "updated_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                        "memory_type": "episodic",
                        "source": "episode",
                    }
                )

            scored.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
            return scored[: max(1, int(limit))]
        except Exception as exc:
            logger.error("search_episodes failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Notes — user-authored Second Brain primitive
    # ------------------------------------------------------------------

    def upsert_memory_artifact(
        self,
        artifact_type: str,
        source_table: str,
        source_id: Any,
        title: str,
        content: str,
        topic_collection: str = "",
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
        status: str = "active",
        source_message_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[int]:
        """Upsert a canonical memory artifact and optional vector embedding."""
        if not self.enabled:
            return None

        clean_type = (artifact_type or "").strip().lower()
        clean_source_table = (source_table or "").strip().lower()
        clean_source_id = str(source_id or "").strip()
        clean_title = (title or "").strip()
        clean_content = (content or "").strip()
        if not clean_type or not clean_source_table or not clean_source_id:
            return None
        if not clean_title and not clean_content:
            return None

        clean_tags = [t.strip() for t in (tags or []) if t and t.strip()]
        clean_status = (status or "active").strip().lower() or "active"

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_artifacts(
                            artifact_type, source_table, source_id, title, content,
                            topic_collection, tags, confidence, status, source_message_id,
                            metadata_json, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT(artifact_type, source_table, source_id)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            topic_collection = EXCLUDED.topic_collection,
                            tags = EXCLUDED.tags,
                            confidence = EXCLUDED.confidence,
                            status = EXCLUDED.status,
                            source_message_id = EXCLUDED.source_message_id,
                            metadata_json = EXCLUDED.metadata_json,
                            updated_at = NOW()
                        RETURNING id;
                        """,
                        (
                            clean_type,
                            clean_source_table,
                            clean_source_id,
                            clean_title,
                            clean_content,
                            (topic_collection or "").strip(),
                            clean_tags,
                            float(confidence),
                            clean_status,
                            source_message_id,
                            json.dumps(metadata or {}, ensure_ascii=False),
                        ),
                    )
                    row = cur.fetchone()
                    artifact_id = int(row[0]) if row else None
                    if artifact_id is not None and embedding is not None:
                        cur.execute(
                            """
                            INSERT INTO memory_artifact_embeddings(artifact_id, embedding, created_at)
                            VALUES (%s, %s::vector, NOW())
                            ON CONFLICT(artifact_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                created_at = NOW();
                            """,
                            (artifact_id, _vector_literal(embedding)),
                        )
                conn.commit()
            return artifact_id
        except Exception as exc:
            logger.error("upsert_memory_artifact failed: %s", exc)
            return None

    def delete_memory_artifact(
        self,
        artifact_id: Optional[int] = None,
        artifact_type: Optional[str] = None,
        source_table: Optional[str] = None,
        source_id: Optional[Any] = None,
    ) -> bool:
        """Delete a memory artifact by id or by source triple."""
        if not self.enabled:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    if artifact_id is not None:
                        cur.execute("DELETE FROM memory_artifacts WHERE id = %s;", (int(artifact_id),))
                    else:
                        cur.execute(
                            """
                            DELETE FROM memory_artifacts
                            WHERE artifact_type = %s
                              AND source_table = %s
                              AND source_id = %s;
                            """,
                            (
                                (artifact_type or "").strip().lower(),
                                (source_table or "").strip().lower(),
                                str(source_id or "").strip(),
                            ),
                        )
                    deleted = cur.rowcount > 0
                conn.commit()
            return deleted
        except Exception as exc:
            logger.error("delete_memory_artifact failed: %s", exc)
            return False

    def vector_search_memory(
        self,
        query_embedding: List[float],
        topic_collection: Optional[str] = None,
        artifact_types: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Cosine-similarity search over all canonical memory artifacts."""
        if not self.enabled:
            return []

        clauses = ["ma.status = 'active'"]
        params: List[Any] = []
        if topic_collection is not None:
            clauses.append("ma.topic_collection = %s")
            params.append((topic_collection or "").strip())
        if artifact_types:
            clean_types = [t.strip().lower() for t in artifact_types if t and t.strip()]
            if clean_types:
                clauses.append("ma.artifact_type = ANY(%s)")
                params.append(clean_types)

        vector_sql = _vector_literal(query_embedding)
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT
                            ma.id, ma.artifact_type, ma.source_table, ma.source_id,
                            ma.title, ma.content, ma.topic_collection, ma.tags,
                            ma.confidence, ma.status, ma.source_message_id,
                            ma.metadata_json, ma.created_at, ma.updated_at,
                            (1 - (mae.embedding <=> %s::vector)) AS similarity
                        FROM memory_artifact_embeddings mae
                        JOIN memory_artifacts ma ON ma.id = mae.artifact_id
                        WHERE {' AND '.join(clauses)}
                        ORDER BY mae.embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (vector_sql, *params, vector_sql, int(top_k)),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [self._memory_artifact_row_to_dict(row, include_similarity=True) for row in rows]
        except Exception as exc:
            logger.error("vector_search_memory failed: %s", exc)
            return []

    def list_memory_artifacts(
        self,
        artifact_type: Optional[str] = None,
        topic_collection: Optional[str] = None,
        status: str = "active",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status.strip().lower())
        if artifact_type:
            clauses.append("artifact_type = %s")
            params.append(artifact_type.strip().lower())
        if topic_collection is not None:
            clauses.append("topic_collection = %s")
            params.append((topic_collection or "").strip())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, artifact_type, source_table, source_id, title, content,
                               topic_collection, tags, confidence, status, source_message_id,
                               metadata_json, created_at, updated_at
                        FROM memory_artifacts
                        {where}
                        ORDER BY updated_at DESC, id DESC
                        LIMIT %s;
                        """,
                        tuple(params),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [self._memory_artifact_row_to_dict(row) for row in rows]
        except Exception as exc:
            logger.error("list_memory_artifacts failed: %s", exc)
            return []

    def list_memory_backfill_candidates(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Return existing notes/tasks/decisions/preferences/messages missing canonical artifacts."""
        if not self.enabled:
            return []

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH candidates AS (
                            SELECT 'note' AS artifact_type, 'notes' AS source_table, n.id::text AS source_id,
                                   COALESCE(NULLIF(n.title, ''), 'Nota #' || n.id::text) AS title,
                                   n.content AS content, n.topic_collection, n.tags,
                                   1.0::double precision AS confidence, 'active' AS status,
                                   NULL::bigint AS source_message_id,
                                   jsonb_build_object('note_id', n.id) AS metadata_json,
                                   n.created_at AS created_at, n.updated_at AS updated_at,
                                   'note_created' AS event_type
                            FROM notes n
                            WHERE NOT EXISTS (
                                SELECT 1 FROM memory_artifacts ma
                                WHERE ma.artifact_type = 'note'
                                  AND ma.source_table = 'notes'
                                  AND ma.source_id = n.id::text
                            )

                            UNION ALL
                            SELECT 'task', 'tasks', t.id::text, t.title,
                                   trim(t.title || E'\n\n' || COALESCE(t.details, '') || E'\nStatus: ' || t.status || '; Prioritate: ' || t.priority),
                                   t.topic_collection, ARRAY[]::text[], t.confidence, t.status,
                                   t.source_message_id,
                                   jsonb_build_object('task_id', t.id, 'priority', t.priority, 'due_at', t.due_at),
                                   t.created_at, t.updated_at, 'task_created'
                            FROM tasks t
                            WHERE NOT EXISTS (
                                SELECT 1 FROM memory_artifacts ma
                                WHERE ma.artifact_type = 'task'
                                  AND ma.source_table = 'tasks'
                                  AND ma.source_id = t.id::text
                            )

                            UNION ALL
                            SELECT 'decision', 'decisions', d.id::text, d.title, d.rationale,
                                   d.topic_collection, ARRAY[]::text[], d.confidence, 'active',
                                   d.source_message_id,
                                   jsonb_build_object('decision_id', d.id),
                                   d.created_at, d.created_at, 'decision'
                            FROM decisions d
                            WHERE NOT EXISTS (
                                SELECT 1 FROM memory_artifacts ma
                                WHERE ma.artifact_type = 'decision'
                                  AND ma.source_table = 'decisions'
                                  AND ma.source_id = d.id::text
                            )

                            UNION ALL
                            SELECT 'preference', 'preferences', p.id::text,
                                   'Preferinta: ' || p.preference_key,
                                   p.preference_value, p.topic_collection, ARRAY[]::text[],
                                   p.confidence, 'active', p.source_message_id,
                                   jsonb_build_object('preference_id', p.id, 'preference_key', p.preference_key),
                                   p.created_at, p.updated_at, 'preference_updated'
                            FROM preferences p
                            WHERE NOT EXISTS (
                                SELECT 1 FROM memory_artifacts ma
                                WHERE ma.artifact_type = 'preference'
                                  AND ma.source_table = 'preferences'
                                  AND ma.source_id = p.id::text
                            )

                            UNION ALL
                            SELECT 'episode', 'messages', m.id::text,
                                   COALESCE(NULLIF(m.sources_json->>'question', ''), 'Episod conversatie #' || m.id::text),
                                   trim('Intrebare: ' || COALESCE(m.sources_json->>'question', '') || E'\n\nRaspuns: ' || m.content),
                                   c.topic_collection, ARRAY[]::text[],
                                   0.6::double precision, 'active', m.id,
                                   jsonb_build_object('message_id', m.id, 'chat_id', c.id, 'question', m.sources_json->>'question'),
                                   m.created_at, m.created_at, 'episode_captured'
                            FROM messages m
                            JOIN chats c ON c.id = m.chat_id
                            WHERE m.role = 'assistant'
                              AND length(trim(m.content)) > 0
                              AND NOT EXISTS (
                                  SELECT 1 FROM memory_artifacts ma
                                  WHERE ma.artifact_type = 'episode'
                                    AND ma.source_table = 'messages'
                                    AND ma.source_id = m.id::text
                              )
                        )
                        SELECT artifact_type, source_table, source_id, title, content,
                               topic_collection, tags, confidence, status, source_message_id,
                               metadata_json, created_at, updated_at, event_type
                        FROM candidates
                        ORDER BY updated_at DESC
                        LIMIT %s;
                        """,
                        (int(limit),),
                    )
                    rows = cur.fetchall() or []
                conn.commit()

            candidates: List[Dict[str, Any]] = []
            for row in rows:
                metadata = row[10] if isinstance(row[10], dict) else (json.loads(row[10]) if row[10] else {})
                candidates.append(
                    {
                        "artifact_type": row[0],
                        "source_table": row[1],
                        "source_id": row[2],
                        "title": row[3] or "",
                        "content": row[4] or "",
                        "topic_collection": row[5] or "",
                        "tags": list(row[6] or []),
                        "confidence": float(row[7] or 0.0),
                        "status": row[8] or "active",
                        "source_message_id": int(row[9]) if row[9] is not None else None,
                        "metadata": metadata,
                        "created_at": self._iso(row[11]),
                        "updated_at": self._iso(row[12]),
                        "event_type": row[13] or "",
                    }
                )
            return candidates
        except Exception as exc:
            logger.error("list_memory_backfill_candidates failed: %s", exc)
            return []

    def get_memory_artifact_counts(self) -> Dict[str, int]:
        if not self.enabled:
            return {}
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return {}
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT artifact_type, COUNT(*)
                        FROM memory_artifacts
                        GROUP BY artifact_type;
                        """
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return {str(row[0]): int(row[1] or 0) for row in rows}
        except Exception as exc:
            logger.error("get_memory_artifact_counts failed: %s", exc)
            return {}

    def create_memory_event(
        self,
        event_type: str,
        artifact_type: str = "",
        artifact_id: Optional[int] = None,
        topic_collection: str = "",
        title: str = "",
        preview: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        occurred_at: Optional[str] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_events(
                            event_type, artifact_type, artifact_id, topic_collection,
                            title, preview, metadata_json, occurred_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, NOW()))
                        RETURNING id;
                        """,
                        (
                            (event_type or "").strip(),
                            (artifact_type or "").strip().lower(),
                            artifact_id,
                            (topic_collection or "").strip(),
                            (title or "").strip(),
                            (preview or "").strip()[:1000],
                            json.dumps(metadata or {}, ensure_ascii=False),
                            occurred_at,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("create_memory_event failed: %s", exc)
            return None

    def list_memory_events(
        self,
        topic_collection: Optional[str] = None,
        days: int = 14,
        event_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        clauses = ["occurred_at >= NOW() - (%s || ' days')::INTERVAL"]
        params: List[Any] = [int(days)]
        if topic_collection is not None:
            clauses.append("topic_collection = %s")
            params.append((topic_collection or "").strip())
        if event_types:
            clean_types = [t for t in event_types if t]
            if clean_types:
                clauses.append("event_type = ANY(%s)")
                params.append(clean_types)
        params.append(int(limit))
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, event_type, artifact_type, artifact_id, topic_collection,
                               title, preview, metadata_json, occurred_at
                        FROM memory_events
                        WHERE {' AND '.join(clauses)}
                        ORDER BY occurred_at DESC, id DESC
                        LIMIT %s;
                        """,
                        tuple(params),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "type": row[1] or "",
                    "artifact_type": row[2] or "",
                    "artifact_id": int(row[3]) if row[3] is not None else None,
                    "topic_collection": row[4] or "",
                    "title": row[5] or "",
                    "preview": row[6] or "",
                    "metadata": row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else {}),
                    "timestamp": row[8].isoformat() if hasattr(row[8], "isoformat") else str(row[8]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_memory_events failed: %s", exc)
            return []

    def create_memory_proposal(
        self,
        proposal_type: str,
        payload: Dict[str, Any],
        confidence: float,
        topic_collection: str = "",
        source_message_id: Optional[int] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_proposals(
                            proposal_type, payload_json, confidence, status,
                            source_message_id, topic_collection, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, 'pending', %s, %s, NOW(), NOW())
                        RETURNING id;
                        """,
                        (
                            (proposal_type or "").strip().lower(),
                            json.dumps(payload or {}, ensure_ascii=False),
                            float(confidence),
                            source_message_id,
                            (topic_collection or "").strip(),
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("create_memory_proposal failed: %s", exc)
            return None

    def list_memory_proposals(
        self,
        status: str = "pending",
        topic_collection: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status.strip().lower())
        if topic_collection is not None:
            clauses.append("topic_collection = %s")
            params.append((topic_collection or "").strip())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, proposal_type, payload_json, confidence, status,
                               source_message_id, topic_collection, created_at, updated_at, resolved_at
                        FROM memory_proposals
                        {where}
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s;
                        """,
                        tuple(params),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            proposals: List[Dict[str, Any]] = []
            for row in rows:
                payload = row[2] if isinstance(row[2], dict) else (json.loads(row[2]) if row[2] else {})
                proposals.append(
                    {
                        "id": int(row[0]),
                        "proposal_type": row[1] or "",
                        "payload": payload,
                        "confidence": float(row[3] or 0.0),
                        "status": row[4] or "",
                        "source_message_id": int(row[5]) if row[5] is not None else None,
                        "topic_collection": row[6] or "",
                        "created_at": row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
                        "updated_at": row[8].isoformat() if hasattr(row[8], "isoformat") else str(row[8]),
                        "resolved_at": row[9].isoformat() if hasattr(row[9], "isoformat") else (str(row[9]) if row[9] else None),
                    }
                )
            return proposals
        except Exception as exc:
            logger.error("list_memory_proposals failed: %s", exc)
            return []

    def resolve_memory_proposal(self, proposal_id: int, status: str) -> bool:
        if not self.enabled:
            return False
        clean_status = (status or "").strip().lower()
        if clean_status not in {"accepted", "dismissed"}:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE memory_proposals
                        SET status = %s, updated_at = NOW(), resolved_at = NOW()
                        WHERE id = %s;
                        """,
                        (clean_status, int(proposal_id)),
                    )
                    updated = cur.rowcount > 0
                conn.commit()
            return updated
        except Exception as exc:
            logger.error("resolve_memory_proposal failed: %s", exc)
            return False

    @staticmethod
    def _memory_artifact_row_to_dict(row: Any, include_similarity: bool = False) -> Dict[str, Any]:
        base = {
            "id": int(row[0]),
            "artifact_type": row[1] or "",
            "source_table": row[2] or "",
            "source_id": row[3] or "",
            "title": row[4] or "",
            "content": row[5] or "",
            "topic_collection": row[6] or "",
            "tags": list(row[7] or []),
            "confidence": float(row[8] or 0.0),
            "status": row[9] or "",
            "source_message_id": int(row[10]) if row[10] is not None else None,
            "metadata": row[11] if isinstance(row[11], dict) else (json.loads(row[11]) if row[11] else {}),
            "created_at": row[12].isoformat() if hasattr(row[12], "isoformat") else str(row[12]),
            "updated_at": row[13].isoformat() if hasattr(row[13], "isoformat") else str(row[13]),
        }
        if include_similarity:
            base["similarity"] = float(row[14] or 0.0)
        return base

    def create_note(
        self,
        content: str,
        title: str = "",
        topic_collection: str = "",
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[int]:
        """Persist a user-authored note + optional vector embedding."""
        if not self.enabled or not content or not content.strip():
            return None
        clean_tags = [t.strip() for t in (tags or []) if t and t.strip()]
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO notes(title, content, topic_collection, tags, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """,
                        (title.strip(), content.strip(), (topic_collection or "").strip(), clean_tags),
                    )
                    row = cur.fetchone()
                    note_id = int(row[0]) if row else None
                    if note_id is not None and embedding is not None:
                        cur.execute(
                            """
                            INSERT INTO note_embeddings(note_id, embedding, created_at)
                            VALUES (%s, %s::vector, NOW())
                            ON CONFLICT(note_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                created_at = NOW();
                            """,
                            (note_id, _vector_literal(embedding)),
                        )
                conn.commit()
            return note_id
        except Exception as exc:
            logger.error("create_note failed: %s", exc)
            return None

    def update_note(
        self,
        note_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        topic_collection: Optional[str] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Patch a note. Pass only the fields you want to change."""
        if not self.enabled:
            return False
        sets: List[str] = []
        params: List[Any] = []
        if title is not None:
            sets.append("title = %s")
            params.append(title.strip())
        if content is not None:
            sets.append("content = %s")
            params.append(content.strip())
        if topic_collection is not None:
            sets.append("topic_collection = %s")
            params.append(topic_collection.strip())
        if tags is not None:
            sets.append("tags = %s")
            params.append([t.strip() for t in tags if t and t.strip()])
        if not sets and embedding is None:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    if sets:
                        sets.append("updated_at = NOW()")
                        sql = f"UPDATE notes SET {', '.join(sets)} WHERE id = %s;"
                        cur.execute(sql, (*params, int(note_id)))
                    if embedding is not None:
                        cur.execute(
                            """
                            INSERT INTO note_embeddings(note_id, embedding, created_at)
                            VALUES (%s, %s::vector, NOW())
                            ON CONFLICT(note_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                created_at = NOW();
                            """,
                            (int(note_id), _vector_literal(embedding)),
                        )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("update_note failed: %s", exc)
            return False

    def delete_note(self, note_id: int) -> bool:
        if not self.enabled:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM notes WHERE id = %s;", (int(note_id),))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("delete_note failed: %s", exc)
            return False

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, content, topic_collection, tags, created_at, updated_at
                        FROM notes WHERE id = %s;
                        """,
                        (int(note_id),),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "title": row[1] or "",
                "content": row[2] or "",
                "topic_collection": row[3] or "",
                "tags": list(row[4] or []),
                "created_at": row[5].isoformat() if row[5] else "",
                "updated_at": row[6].isoformat() if row[6] else "",
            }
        except Exception as exc:
            logger.error("get_note failed: %s", exc)
            return None

    def list_notes(
        self,
        topic_collection: Optional[str] = None,
        text_query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List notes ordered by updated_at desc, with optional topic + full-text filter."""
        if not self.enabled:
            return []
        clauses: List[str] = []
        params: List[Any] = []
        if topic_collection is not None and topic_collection.strip():
            clauses.append("topic_collection = %s")
            params.append(topic_collection.strip())
        if text_query and text_query.strip():
            clauses.append(
                "to_tsvector('simple', coalesce(title, '') || ' ' || content) @@ plainto_tsquery('simple', %s)"
            )
            params.append(text_query.strip())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, title, content, topic_collection, tags, created_at, updated_at
                        FROM notes
                        {where}
                        ORDER BY updated_at DESC
                        LIMIT %s;
                        """,
                        tuple(params),
                    )
                    rows = cur.fetchall()
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "title": row[1] or "",
                    "content": row[2] or "",
                    "topic_collection": row[3] or "",
                    "tags": list(row[4] or []),
                    "created_at": row[5].isoformat() if row[5] else "",
                    "updated_at": row[6].isoformat() if row[6] else "",
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_notes failed: %s", exc)
            return []

    def vector_search_notes(
        self,
        query_embedding: List[float],
        topic_collection: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Cosine-similarity search over note embeddings."""
        if not self.enabled:
            return []
        try:
            vector_sql = _vector_literal(query_embedding)
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic_collection and topic_collection.strip():
                        cur.execute(
                            """
                            SELECT
                                notes.id, notes.title, notes.content, notes.topic_collection,
                                notes.tags, notes.created_at, notes.updated_at,
                                (1 - (note_embeddings.embedding <=> %s::vector)) AS similarity
                            FROM note_embeddings
                            JOIN notes ON notes.id = note_embeddings.note_id
                            WHERE notes.topic_collection = %s
                            ORDER BY note_embeddings.embedding <=> %s::vector
                            LIMIT %s;
                            """,
                            (vector_sql, topic_collection.strip(), vector_sql, int(top_k)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                notes.id, notes.title, notes.content, notes.topic_collection,
                                notes.tags, notes.created_at, notes.updated_at,
                                (1 - (note_embeddings.embedding <=> %s::vector)) AS similarity
                            FROM note_embeddings
                            JOIN notes ON notes.id = note_embeddings.note_id
                            ORDER BY note_embeddings.embedding <=> %s::vector
                            LIMIT %s;
                            """,
                            (vector_sql, vector_sql, int(top_k)),
                        )
                    rows = cur.fetchall()
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "title": row[1] or "",
                    "content": row[2] or "",
                    "topic_collection": row[3] or "",
                    "tags": list(row[4] or []),
                    "created_at": row[5].isoformat() if row[5] else "",
                    "updated_at": row[6].isoformat() if row[6] else "",
                    "similarity": float(row[7] or 0.0),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("vector_search_notes failed: %s", exc)
            return []

    def create_task(
        self,
        title: str,
        details: str = "",
        topic_collection: str = "",
        status: str = "open",
        priority: str = "normal",
        due_at: Optional[str] = None,
        confidence: float = 0.75,
        source_message_id: Optional[int] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None

        clean_title = (title or "").strip()
        if not clean_title:
            return None

        clean_status = (status or "open").strip().lower()
        if clean_status not in {"open", "in_progress", "done", "cancelled"}:
            clean_status = "open"

        clean_priority = (priority or "normal").strip().lower()
        if clean_priority not in {"low", "normal", "high"}:
            clean_priority = "normal"

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO tasks(
                            title,
                            details,
                            topic_collection,
                            status,
                            priority,
                            due_at,
                            confidence,
                            source_message_id,
                            created_at,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """,
                        (
                            clean_title,
                            (details or "").strip(),
                            (topic_collection or "").strip(),
                            clean_status,
                            clean_priority,
                            due_at,
                            float(confidence),
                            source_message_id,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.error("create_task failed: %s", exc)
            return None

    def list_tasks(
        self,
        topic_collection: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        topic = (topic_collection or "").strip()
        status_filter = (status or "").strip().lower()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic and status_filter:
                        cur.execute(
                            """
                            SELECT id, title, details, topic_collection, status, priority, due_at, confidence, created_at, updated_at
                            FROM tasks
                            WHERE topic_collection = %s
                              AND status = %s
                            ORDER BY updated_at DESC, id DESC
                            LIMIT %s;
                            """,
                            (topic, status_filter, int(limit)),
                        )
                    elif topic:
                        cur.execute(
                            """
                            SELECT id, title, details, topic_collection, status, priority, due_at, confidence, created_at, updated_at
                            FROM tasks
                            WHERE topic_collection = %s
                            ORDER BY updated_at DESC, id DESC
                            LIMIT %s;
                            """,
                            (topic, int(limit)),
                        )
                    elif status_filter:
                        cur.execute(
                            """
                            SELECT id, title, details, topic_collection, status, priority, due_at, confidence, created_at, updated_at
                            FROM tasks
                            WHERE status = %s
                            ORDER BY updated_at DESC, id DESC
                            LIMIT %s;
                            """,
                            (status_filter, int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, title, details, topic_collection, status, priority, due_at, confidence, created_at, updated_at
                            FROM tasks
                            ORDER BY updated_at DESC, id DESC
                            LIMIT %s;
                            """,
                            (int(limit),),
                        )
                    rows = cur.fetchall() or []
                conn.commit()

            task_rows: List[Dict[str, Any]] = []
            for row in rows:
                due_at = row[6].isoformat() if hasattr(row[6], "isoformat") else (str(row[6]) if row[6] else None)
                created_at = row[8].isoformat() if hasattr(row[8], "isoformat") else str(row[8])
                updated_at = row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9])
                task_rows.append(
                    {
                        "id": int(row[0]),
                        "title": row[1],
                        "details": row[2] or "",
                        "topic_collection": row[3] or "",
                        "status": row[4] or "open",
                        "priority": row[5] or "normal",
                        "due_at": due_at,
                        "confidence": float(row[7] or 0.0),
                        "created_at": created_at,
                        "updated_at": updated_at,
                    }
                )
            return task_rows
        except Exception as exc:
            logger.error("list_tasks failed: %s", exc)
            return []

    def update_task(
        self,
        task_id: int,
        title: Optional[str] = None,
        details: Optional[str] = None,
        priority: Optional[str] = None,
        due_at: Optional[str] = None,
        topic_collection: Optional[str] = None,
    ) -> bool:
        """Update partial al unui task. None = nu modifica acel camp.

        Pentru due_at: trimite '' pentru a curata data scadenta.
        """
        if not self.enabled:
            return False

        sets: List[str] = []
        params: List[Any] = []

        if title is not None:
            clean_title = title.strip()
            if not clean_title:
                return False
            sets.append("title = %s")
            params.append(clean_title)

        if details is not None:
            sets.append("details = %s")
            params.append(details.strip())

        if priority is not None:
            clean_priority = (priority or "normal").strip().lower()
            if clean_priority not in {"low", "normal", "high"}:
                clean_priority = "normal"
            sets.append("priority = %s")
            params.append(clean_priority)

        if due_at is not None:
            # '' sau None-ish trimis explicit -> NULL in DB
            value = due_at.strip() if isinstance(due_at, str) else due_at
            sets.append("due_at = %s")
            params.append(value if value else None)

        if topic_collection is not None:
            sets.append("topic_collection = %s")
            params.append((topic_collection or "").strip())

        if not sets:
            return False

        sets.append("updated_at = NOW()")
        params.append(int(task_id))
        sql = f"UPDATE tasks SET {', '.join(sets)} WHERE id = %s;"

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    updated = cur.rowcount > 0
                conn.commit()
            return updated
        except Exception as exc:
            logger.error("update_task failed: %s", exc)
            return False

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, details, topic_collection, status, priority,
                               due_at, confidence, created_at, updated_at
                        FROM tasks
                        WHERE id = %s;
                        """,
                        (int(task_id),),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "title": row[1],
                "details": row[2] or "",
                "topic_collection": row[3] or "",
                "status": row[4] or "open",
                "priority": row[5] or "normal",
                "due_at": row[6].isoformat() if hasattr(row[6], "isoformat") else (str(row[6]) if row[6] else None),
                "confidence": float(row[7] or 0.0),
                "created_at": row[8].isoformat() if hasattr(row[8], "isoformat") else str(row[8]),
                "updated_at": row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9]),
            }
        except Exception as exc:
            logger.error("get_task failed: %s", exc)
            return None

    def delete_task(self, task_id: int) -> bool:
        if not self.enabled:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM tasks WHERE id = %s;", (int(task_id),))
                    deleted = cur.rowcount > 0
                conn.commit()
            return deleted
        except Exception as exc:
            logger.error("delete_task failed: %s", exc)
            return False

    def list_due_soon_tasks(
        self,
        within_days: int = 7,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Tasks cu due_at in urmatoarele N zile sau deja expirate (status open/in_progress)."""
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, details, topic_collection, status, priority, due_at, confidence, created_at, updated_at,
                               EXTRACT(EPOCH FROM (due_at - NOW()))/86400.0 AS days_until_due
                        FROM tasks
                        WHERE status IN ('open', 'in_progress')
                          AND due_at IS NOT NULL
                          AND due_at <= NOW() + (%s || ' days')::INTERVAL
                        ORDER BY due_at ASC
                        LIMIT %s;
                        """,
                        (int(within_days), int(limit)),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "title": row[1],
                    "details": row[2] or "",
                    "topic_collection": row[3] or "",
                    "status": row[4] or "open",
                    "priority": row[5] or "normal",
                    "due_at": row[6].isoformat() if hasattr(row[6], "isoformat") else (str(row[6]) if row[6] else None),
                    "confidence": float(row[7] or 0.0),
                    "created_at": row[8].isoformat() if hasattr(row[8], "isoformat") else str(row[8]),
                    "updated_at": row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9]),
                    "days_until_due": float(row[10] or 0.0),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_due_soon_tasks failed: %s", exc)
            return []

    def update_task_status(self, task_id: int, status: str) -> bool:
        if not self.enabled:
            return False

        clean_status = (status or "").strip().lower()
        if clean_status not in {"open", "in_progress", "done", "cancelled"}:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE tasks
                        SET status = %s,
                            updated_at = NOW()
                        WHERE id = %s;
                        """,
                        (clean_status, int(task_id)),
                    )
                    updated = cur.rowcount > 0
                conn.commit()
            return updated
        except Exception as exc:
            logger.error("update_task_status failed: %s", exc)
            return False

    def search_tasks(
        self,
        question: str,
        topic_collection: Optional[str] = None,
        limit: int = 5,
        include_closed: bool = False,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        terms = self._tokenize_query_terms(question)
        if not terms:
            return []

        like_terms = [f"%{term}%" for term in terms]
        topic = (topic_collection or "").strip()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    status_clause = "" if include_closed else "AND t.status IN ('open', 'in_progress')"
                    if topic:
                        cur.execute(
                            f"""
                            SELECT t.id, t.title, t.details, t.topic_collection, t.status, t.priority, t.confidence, t.updated_at
                            FROM tasks t
                            WHERE t.topic_collection = %s
                              {status_clause}
                              AND (
                                LOWER(t.title) LIKE ANY(%s)
                                OR LOWER(t.details) LIKE ANY(%s)
                              )
                            ORDER BY t.confidence DESC, t.updated_at DESC
                            LIMIT %s;
                            """,
                            (topic, like_terms, like_terms, int(limit)),
                        )
                    else:
                        cur.execute(
                            f"""
                            SELECT t.id, t.title, t.details, t.topic_collection, t.status, t.priority, t.confidence, t.updated_at
                            FROM tasks t
                            WHERE 1=1
                              {status_clause}
                              AND (
                                LOWER(t.title) LIKE ANY(%s)
                                OR LOWER(t.details) LIKE ANY(%s)
                              )
                            ORDER BY t.confidence DESC, t.updated_at DESC
                            LIMIT %s;
                            """,
                            (like_terms, like_terms, int(limit)),
                        )
                    rows = cur.fetchall() or []
                conn.commit()

            return [
                {
                    "id": int(row[0]),
                    "title": f"Task: {row[1]}",
                    "rationale": row[2] or "",
                    "topic_collection": row[3] or "",
                    "status": row[4] or "open",
                    "priority": row[5] or "normal",
                    "confidence": float(row[6] or 0.0),
                    "created_at": row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
                    "updated_at": row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
                    "memory_type": "task",
                    "source": "task",
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("search_tasks failed: %s", exc)
            return []

    def update_decision(
        self,
        decision_id: int,
        title: Optional[str] = None,
        rationale: Optional[str] = None,
        topic_collection: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Partial update pentru o decizie. None = nu modifica."""
        if not self.enabled:
            return False

        sets: List[str] = []
        params: List[Any] = []

        if title is not None:
            clean_title = title.strip()
            if not clean_title:
                return False
            sets.append("title = %s")
            params.append(clean_title)

        if rationale is not None:
            sets.append("rationale = %s")
            params.append(rationale.strip())

        if topic_collection is not None:
            sets.append("topic_collection = %s")
            params.append((topic_collection or "").strip())

        if confidence is not None:
            sets.append("confidence = %s")
            params.append(float(confidence))

        if not sets:
            return False

        params.append(int(decision_id))
        sql = f"UPDATE decisions SET {', '.join(sets)} WHERE id = %s;"

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    updated = cur.rowcount > 0
                conn.commit()
            return updated
        except Exception as exc:
            logger.error("update_decision failed: %s", exc)
            return False

    def get_decision(self, decision_id: int) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, rationale, topic_collection, confidence, created_at
                        FROM decisions
                        WHERE id = %s;
                        """,
                        (int(decision_id),),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "title": row[1] or "",
                "rationale": row[2] or "",
                "topic_collection": row[3] or "",
                "confidence": float(row[4] or 0.0),
                "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            }
        except Exception as exc:
            logger.error("get_decision failed: %s", exc)
            return None

    def delete_decision(self, decision_id: int) -> bool:
        if not self.enabled:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM decisions WHERE id = %s;", (int(decision_id),))
                    deleted = cur.rowcount > 0
                conn.commit()
            return deleted
        except Exception as exc:
            logger.error("delete_decision failed: %s", exc)
            return False

    def list_all_decisions(
        self,
        topic_collection: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Toate deciziile (cu filtru optional pe topic)."""
        if not self.enabled:
            return []
        topic = (topic_collection or "").strip()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic:
                        cur.execute(
                            """
                            SELECT id, title, rationale, topic_collection, confidence, created_at
                            FROM decisions
                            WHERE topic_collection = %s
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            (topic, int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, title, rationale, topic_collection, confidence, created_at
                            FROM decisions
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            (int(limit),),
                        )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "title": row[1],
                    "rationale": row[2] or "",
                    "topic_collection": row[3] or "",
                    "confidence": float(row[4] or 0.0),
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_all_decisions failed: %s", exc)
            return []

    def update_preference(
        self,
        preference_id: int,
        preference_value: Optional[str] = None,
        confidence: Optional[float] = None,
        topic_collection: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return False

        sets: List[str] = []
        params: List[Any] = []

        if preference_value is not None:
            clean_val = preference_value.strip()
            if not clean_val:
                return False
            sets.append("preference_value = %s")
            params.append(clean_val)

        if confidence is not None:
            sets.append("confidence = %s")
            params.append(float(confidence))

        if topic_collection is not None:
            sets.append("topic_collection = %s")
            params.append((topic_collection or "").strip())

        if not sets:
            return False

        sets.append("updated_at = NOW()")
        params.append(int(preference_id))
        sql = f"UPDATE preferences SET {', '.join(sets)} WHERE id = %s;"

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    updated = cur.rowcount > 0
                conn.commit()
            return updated
        except Exception as exc:
            logger.error("update_preference failed: %s", exc)
            return False

    def get_preference(self, preference_id: int) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, preference_key, preference_value, topic_collection, confidence, updated_at
                        FROM preferences
                        WHERE id = %s;
                        """,
                        (int(preference_id),),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "preference_key": row[1] or "",
                "preference_value": row[2] or "",
                "topic_collection": row[3] or "",
                "confidence": float(row[4] or 0.0),
                "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            }
        except Exception as exc:
            logger.error("get_preference failed: %s", exc)
            return None

    def delete_preference(self, preference_id: int) -> bool:
        if not self.enabled:
            return False
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM preferences WHERE id = %s;", (int(preference_id),))
                    deleted = cur.rowcount > 0
                conn.commit()
            return deleted
        except Exception as exc:
            logger.error("delete_preference failed: %s", exc)
            return False

    def list_all_preferences(
        self,
        topic_collection: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        topic = (topic_collection or "").strip()
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic:
                        cur.execute(
                            """
                            SELECT id, preference_key, preference_value, topic_collection, confidence, updated_at
                            FROM preferences
                            WHERE topic_collection = %s
                            ORDER BY confidence DESC, updated_at DESC
                            LIMIT %s;
                            """,
                            (topic, int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, preference_key, preference_value, topic_collection, confidence, updated_at
                            FROM preferences
                            ORDER BY confidence DESC, updated_at DESC
                            LIMIT %s;
                            """,
                            (int(limit),),
                        )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "preference_key": row[1] or "",
                    "preference_value": row[2] or "",
                    "topic_collection": row[3] or "",
                    "confidence": float(row[4] or 0.0),
                    "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_all_preferences failed: %s", exc)
            return []

    def get_weekly_summary(self, days: int = 7) -> Dict[str, Any]:
        """Sumar pentru ultimele N zile: decizii noi, task-uri inchise/deschise, documente noi."""
        empty = {
            "window_days": days,
            "decisions_added": 0,
            "tasks_added": 0,
            "tasks_closed": 0,
            "tasks_currently_open": 0,
            "documents_added": 0,
            "messages_count": 0,
            "recent_decisions": [],
            "recent_closed_tasks": [],
        }
        if not self.enabled:
            return empty
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return empty
                with conn.cursor() as cur:
                    # Counters
                    cur.execute(
                        "SELECT COUNT(*) FROM decisions WHERE created_at >= NOW() - (%s || ' days')::INTERVAL;",
                        (int(days),),
                    )
                    decisions_added = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute(
                        "SELECT COUNT(*) FROM tasks WHERE created_at >= NOW() - (%s || ' days')::INTERVAL;",
                        (int(days),),
                    )
                    tasks_added = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute(
                        """
                        SELECT COUNT(*) FROM tasks
                        WHERE status IN ('done', 'cancelled')
                          AND updated_at >= NOW() - (%s || ' days')::INTERVAL;
                        """,
                        (int(days),),
                    )
                    tasks_closed = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('open', 'in_progress');")
                    tasks_open = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute(
                        "SELECT COUNT(*) FROM documents WHERE created_at >= NOW() - (%s || ' days')::INTERVAL;",
                        (int(days),),
                    )
                    documents_added = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute(
                        "SELECT COUNT(*) FROM messages WHERE created_at >= NOW() - (%s || ' days')::INTERVAL;",
                        (int(days),),
                    )
                    messages_count = int((cur.fetchone() or [0])[0] or 0)

                    # Recent decisions list
                    cur.execute(
                        """
                        SELECT id, title, topic_collection, created_at
                        FROM decisions
                        WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                        ORDER BY created_at DESC
                        LIMIT 5;
                        """,
                        (int(days),),
                    )
                    recent_decisions = [
                        {
                            "id": int(row[0]),
                            "title": row[1],
                            "topic_collection": row[2] or "",
                            "created_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                        }
                        for row in (cur.fetchall() or [])
                    ]

                    # Recent closed tasks
                    cur.execute(
                        """
                        SELECT id, title, topic_collection, status, updated_at
                        FROM tasks
                        WHERE status IN ('done', 'cancelled')
                          AND updated_at >= NOW() - (%s || ' days')::INTERVAL
                        ORDER BY updated_at DESC
                        LIMIT 5;
                        """,
                        (int(days),),
                    )
                    recent_closed_tasks = [
                        {
                            "id": int(row[0]),
                            "title": row[1],
                            "topic_collection": row[2] or "",
                            "status": row[3] or "done",
                            "updated_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
                        }
                        for row in (cur.fetchall() or [])
                    ]
                conn.commit()
            return {
                "window_days": days,
                "decisions_added": decisions_added,
                "tasks_added": tasks_added,
                "tasks_closed": tasks_closed,
                "tasks_currently_open": tasks_open,
                "documents_added": documents_added,
                "messages_count": messages_count,
                "recent_decisions": recent_decisions,
                "recent_closed_tasks": recent_closed_tasks,
            }
        except Exception as exc:
            logger.error("get_weekly_summary failed: %s", exc)
            return empty

    def list_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returneaza cele mai recente decizii memorate, indiferent de topic."""
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, rationale, topic_collection, confidence, created_at
                        FROM decisions
                        ORDER BY created_at DESC
                        LIMIT %s;
                        """,
                        (int(limit),),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "title": row[1],
                    "rationale": row[2] or "",
                    "topic_collection": row[3] or "",
                    "confidence": float(row[4] or 0.0),
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_recent_decisions failed: %s", exc)
            return []

    def list_aged_decisions(
        self,
        min_age_days: int = 30,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Decizii mai vechi decat min_age_days, candidati pentru review (drift)."""
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, rationale, topic_collection, confidence, created_at,
                               EXTRACT(DAY FROM NOW() - created_at)::INT AS age_days
                        FROM decisions
                        WHERE created_at < NOW() - (%s || ' days')::INTERVAL
                        ORDER BY created_at ASC
                        LIMIT %s;
                        """,
                        (int(min_age_days), int(limit)),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "title": row[1],
                    "rationale": row[2] or "",
                    "topic_collection": row[3] or "",
                    "confidence": float(row[4] or 0.0),
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                    "age_days": int(row[6] or 0),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_aged_decisions failed: %s", exc)
            return []

    def list_top_preferences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returneaza preferintele cu cea mai mare confidence."""
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, preference_key, preference_value, topic_collection, confidence, updated_at
                        FROM preferences
                        ORDER BY confidence DESC, updated_at DESC
                        LIMIT %s;
                        """,
                        (int(limit),),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "preference_key": row[1] or "",
                    "preference_value": row[2] or "",
                    "topic_collection": row[3] or "",
                    "confidence": float(row[4] or 0.0),
                    "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_top_preferences failed: %s", exc)
            return []

    def get_second_brain_status(self) -> Dict[str, Any]:
        default_status = {
            "decisions_count": 0,
            "preferences_count": 0,
            "tasks_open_count": 0,
            "tasks_total_count": 0,
            "messages_count": 0,
        }
        if not self.enabled:
            return default_status

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return default_status
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM decisions;")
                    decisions_count = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM preferences;")
                    preferences_count = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM tasks;")
                    tasks_total_count = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('open', 'in_progress');")
                    tasks_open_count = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM messages;")
                    messages_count = int((cur.fetchone() or [0])[0] or 0)
                conn.commit()
            return {
                "decisions_count": decisions_count,
                "preferences_count": preferences_count,
                "tasks_open_count": tasks_open_count,
                "tasks_total_count": tasks_total_count,
                "messages_count": messages_count,
            }
        except Exception as exc:
            logger.error("get_second_brain_status failed: %s", exc)
            return default_status

    def clear_all_embeddings(self) -> Dict[str, int]:
        """Sterge toate embeddings + chunks si reseteaza documents.indexed.

        Folosit la reindexare completa (de ex. dupa schimbarea modelului de embedding).
        Returneaza counters: {chunks_deleted, embeddings_deleted, documents_reset}.
        """
        result = {"chunks_deleted": 0, "embeddings_deleted": 0, "documents_reset": 0}
        if not self.enabled:
            return result
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return result
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                    result["embeddings_deleted"] = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM chunks;")
                    result["chunks_deleted"] = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM documents WHERE indexed = TRUE;")
                    result["documents_reset"] = int((cur.fetchone() or [0])[0] or 0)

                    # CASCADE pe chunks va sterge si chunk_embeddings.
                    cur.execute("DELETE FROM chunks;")
                    cur.execute("UPDATE documents SET indexed = FALSE, updated_at = NOW();")
                conn.commit()
            logger.info(
                "clear_all_embeddings: removed %d chunks / %d embeddings, reset %d documents",
                result["chunks_deleted"], result["embeddings_deleted"], result["documents_reset"],
            )
            return result
        except Exception as exc:
            logger.error("clear_all_embeddings failed: %s", exc)
            return result

    def log_retrieval(
        self,
        question: str,
        route_used: str,
        latency_ms: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO retrieval_logs(question, route_used, latency_ms, metrics_json, created_at)
                        VALUES (%s, %s, %s, %s, NOW());
                        """,
                        (question, route_used, float(latency_ms), json.dumps(metrics or {}, ensure_ascii=False)),
                    )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("log_retrieval failed: %s", exc)
            return False

    def list_retrieval_logs(
        self,
        limit: int = 100,
        route_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returneaza ultimele N retrieval logs (cu filtru optional pe route)."""
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if route_filter:
                        cur.execute(
                            """
                            SELECT id, question, route_used, latency_ms, metrics_json, created_at
                            FROM retrieval_logs
                            WHERE route_used = %s
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            (route_filter, int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, question, route_used, latency_ms, metrics_json, created_at
                            FROM retrieval_logs
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            (int(limit),),
                        )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "question": row[1] or "",
                    "route_used": row[2] or "",
                    "latency_ms": float(row[3] or 0.0),
                    "metrics": row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] else {}),
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_retrieval_logs failed: %s", exc)
            return []

    def get_retrieval_stats(self, days: int = 7) -> Dict[str, Any]:
        """Statistici pe retrieval_logs in ultima fereastra de N zile (avg/p95/route distribution)."""
        empty = {
            "window_days": days,
            "total": 0,
            "by_route": {},
            "latency_ms_avg": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
        }
        if not self.enabled:
            return empty
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return empty
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*),
                               COALESCE(AVG(latency_ms), 0),
                               COALESCE(PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY latency_ms), 0),
                               COALESCE(PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY latency_ms), 0)
                        FROM retrieval_logs
                        WHERE created_at >= NOW() - (%s || ' days')::INTERVAL;
                        """,
                        (int(days),),
                    )
                    row = cur.fetchone() or (0, 0.0, 0.0, 0.0)
                    total = int(row[0] or 0)
                    avg = float(row[1] or 0.0)
                    p50 = float(row[2] or 0.0)
                    p95 = float(row[3] or 0.0)

                    cur.execute(
                        """
                        SELECT route_used, COUNT(*) AS n, COALESCE(AVG(latency_ms), 0) AS avg_lat
                        FROM retrieval_logs
                        WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                        GROUP BY route_used
                        ORDER BY n DESC;
                        """,
                        (int(days),),
                    )
                    by_route_rows = cur.fetchall() or []
                conn.commit()
            return {
                "window_days": days,
                "total": total,
                "latency_ms_avg": round(avg, 1),
                "latency_ms_p50": round(p50, 1),
                "latency_ms_p95": round(p95, 1),
                "by_route": {
                    (r[0] or "unknown"): {
                        "count": int(r[1] or 0),
                        "latency_ms_avg": round(float(r[2] or 0.0), 1),
                    }
                    for r in by_route_rows
                },
            }
        except Exception as exc:
            logger.error("get_retrieval_stats failed: %s", exc)
            return empty

    def get_index_status(self) -> Dict[str, Any]:
        """Return indexed documents/chunks counters for startup bootstrap."""
        default_status = {
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "last_sync_at": None,
        }
        if not self.enabled:
            return default_status

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return default_status
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) FILTER (WHERE indexed = TRUE) AS indexed_documents,
                            COALESCE(MAX(updated_at), MAX(created_at)) AS last_sync_at
                        FROM documents;
                        """
                    )
                    doc_row = cur.fetchone() or (0, None)

                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM chunks
                        JOIN documents ON documents.id = chunks.document_id
                        WHERE documents.indexed = TRUE;
                        """
                    )
                    chunk_row = cur.fetchone() or (0,)
                conn.commit()

            indexed_documents = int(doc_row[0] or 0)
            indexed_chunks = int(chunk_row[0] or 0)
            last_sync_at = doc_row[1]
            if hasattr(last_sync_at, "isoformat"):
                last_sync_at = last_sync_at.isoformat()
            elif last_sync_at is not None:
                last_sync_at = str(last_sync_at)

            return {
                "indexed_documents": indexed_documents,
                "indexed_chunks": indexed_chunks,
                "last_sync_at": last_sync_at,
            }
        except Exception as exc:
            logger.error("get_index_status failed: %s", exc)
            return default_status

    def list_recent_documents(
        self,
        topic_collection: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return documents ordered by created_at desc, with optional topic + recency filters.

        Used by the Timeline view to surface "documents added recently" events.
        """
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    if topic_collection is not None and topic_collection.strip():
                        cur.execute(
                            """
                            SELECT
                                d.id, d.original_name, d.source_path, d.indexed,
                                c.name AS collection_name,
                                d.created_at, d.updated_at
                            FROM documents d
                            JOIN collections c ON c.id = d.collection_id
                            WHERE c.name = %s
                              AND d.created_at >= NOW() - (%s || ' days')::INTERVAL
                            ORDER BY d.created_at DESC
                            LIMIT %s;
                            """,
                            (topic_collection.strip(), int(days), int(limit)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                d.id, d.original_name, d.source_path, d.indexed,
                                c.name AS collection_name,
                                d.created_at, d.updated_at
                            FROM documents d
                            JOIN collections c ON c.id = d.collection_id
                            WHERE d.created_at >= NOW() - (%s || ' days')::INTERVAL
                            ORDER BY d.created_at DESC
                            LIMIT %s;
                            """,
                            (int(days), int(limit)),
                        )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "original_name": row[1] or "",
                    "source_path": row[2] or "",
                    "indexed": bool(row[3]),
                    "collection_name": row[4] or "",
                    "created_at": row[5].isoformat() if row[5] else "",
                    "updated_at": row[6].isoformat() if row[6] else "",
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_recent_documents failed: %s", exc)
            return []

    def list_collections(self) -> List[Dict[str, Any]]:
        """Return all collections with document counts (shared source of truth).

        Used by the Notebooks view so notebooks persist in Postgres and are
        visible across machines/deployments, not only in the local JSON library.
        """
        if not self.enabled:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            c.id, c.name, c.type, c.created_at,
                            COUNT(d.id) AS doc_count,
                            COUNT(d.id) FILTER (WHERE d.indexed) AS indexed_count
                        FROM collections c
                        LEFT JOIN documents d ON d.collection_id = c.id
                        GROUP BY c.id, c.name, c.type, c.created_at
                        ORDER BY c.name;
                        """
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "name": row[1] or "",
                    "type": row[2] or "topic",
                    "created_at": row[3].isoformat() if row[3] else "",
                    "document_count": int(row[4] or 0),
                    "indexed_count": int(row[5] or 0),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_collections failed: %s", exc)
            return []

    def list_documents_by_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """Return documents of a collection from Postgres (shared source of truth)."""
        if not self.enabled:
            return []
        name = (collection_name or "").strip()
        if not name:
            return []
        try:
            with self.client.connection() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT d.id, d.file_hash, d.original_name, d.source_path,
                               d.indexed, d.created_at, d.updated_at
                        FROM documents d
                        JOIN collections c ON c.id = d.collection_id
                        WHERE c.name = %s
                        ORDER BY d.original_name;
                        """,
                        (name,),
                    )
                    rows = cur.fetchall() or []
                conn.commit()
            return [
                {
                    "id": int(row[0]),
                    "file_hash": row[1] or "",
                    "original_name": row[2] or "",
                    "source_path": row[3] or "",
                    "indexed": bool(row[4]),
                    "created_at": row[5].isoformat() if row[5] else "",
                    "updated_at": row[6].isoformat() if row[6] else "",
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("list_documents_by_collection failed for %s: %s", name, exc)
            return []

    def is_file_hash_indexed(self, file_hash: str) -> bool:
        """Return True when file hash exists and document is already indexed."""
        if not self.enabled:
            return False
        normalized_hash = (file_hash or "").strip()
        if not normalized_hash:
            return False

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT indexed
                        FROM documents
                        WHERE file_hash = %s
                        LIMIT 1;
                        """,
                        (normalized_hash,),
                    )
                    row = cur.fetchone()
                conn.commit()
            return bool(row and row[0])
        except Exception as exc:
            logger.error("is_file_hash_indexed failed: %s", exc)
            return False

    def list_missing_hashes(self, file_hashes: List[str]) -> List[str]:
        """Return hashes that are absent or not indexed yet."""
        if not self.enabled:
            return [hash_value for hash_value in file_hashes if hash_value]

        normalized_hashes = [hash_value.strip() for hash_value in file_hashes if (hash_value or "").strip()]
        if not normalized_hashes:
            return []

        try:
            with self.client.connection() as conn:
                if conn is None:
                    return normalized_hashes
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT file_hash
                        FROM documents
                        WHERE file_hash = ANY(%s)
                          AND indexed = TRUE;
                        """,
                        (normalized_hashes,),
                    )
                    rows = cur.fetchall() or []
                conn.commit()

            indexed = {str(row[0]) for row in rows}
            return [hash_value for hash_value in normalized_hashes if hash_value not in indexed]
        except Exception as exc:
            logger.error("list_missing_hashes failed: %s", exc)
            return normalized_hashes

    @staticmethod
    def _hash_file(path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def _tokenize_query_terms(question: str) -> List[str]:
        text = (question or "").strip().lower()
        if not text:
            return []
        terms = [token for token in text.split() if len(token) >= 3]
        deduped: List[str] = []
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped[:10]
