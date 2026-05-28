"""Graph ingestion pipeline: chunk -> concepts/relations -> Neo4j upsert."""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

STOPWORDS = {
    "care",
    "este",
    "sunt",
    "pentru",
    "despre",
    "acest",
    "aceasta",
    "the",
    "si",
    "sau",
    "din",
    "prin",
    "fara",
    "with",
    "that",
    "this",
    "from",
    "unde",
    "cand",
    "cum",
    "iar",
    "asa",
    "deci",
    "fiind",
    "also",
}

RELATION_MARKERS = {
    "DEPENDS_ON": {"depinde de", "depends on", "bazat pe", "based on"},
    "CONTRADICTS": {"contrazice", "incompatibil", "conflict", "opune"},
    "DERIVED_FROM": {"deriva din", "provenit din", "derived from", "rezulta din"},
}

ALLOWED_RELATIONS = {"RELATED_TO", "DEPENDS_ON", "CONTRADICTS", "DERIVED_FROM"}


class GraphIngestionService:
    """Offline incremental graph ingestion service with confidence gating."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        min_confidence: float = 0.70,
        extraction_mode: str = "heuristic_fallback",
        structured_extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self.client = neo4j_client
        self.min_confidence = float(min_confidence)
        self.extraction_mode = (extraction_mode or "heuristic_fallback").strip().lower()
        self.structured_extractor = structured_extractor

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest chunks and structured concept relations with provenance."""
        if not self.enabled:
            return {"documents": 0, "chunks": 0, "concepts": 0, "relations": 0}

        unique_docs = set()
        chunks_count = 0
        concepts_count = 0
        relations_count = 0

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")
            source_path = metadata.get("source_path", "")
            if not source_path:
                continue

            doc_id = str(metadata.get("doc_id") or source_path)
            filename = metadata.get("filename", "document")
            collection = metadata.get("collection", "general")
            chunk_id = f"{doc_id}:{metadata.get('chunk_id', 0)}"
            chunk_order = int(metadata.get("chunk_id", 0))
            topic_id = f"topic:{(collection or 'general').lower()}"

            unique_docs.add(doc_id)

            self.client.run_write(
                """
                MERGE (t:Topic {id: $topic_id})
                SET t.name = $topic_name
                MERGE (d:Document {id: $doc_id})
                SET d.filename = $filename,
                    d.collection = $collection,
                    d.source_path = $source_path
                MERGE (d)-[:ABOUT_TOPIC]->(t)
                MERGE (c:Chunk {id: $chunk_id})
                SET c.order = $chunk_order,
                    c.text = $chunk_text
                MERGE (c)-[:SOURCED_FROM]->(d)
                """,
                {
                    "topic_id": topic_id,
                    "topic_name": collection,
                    "doc_id": doc_id,
                    "filename": filename,
                    "collection": collection,
                    "source_path": source_path,
                    "chunk_id": str(chunk_id),
                    "chunk_order": chunk_order,
                    "chunk_text": content[:3000],
                },
            )
            chunks_count += 1

            concepts, relations = self._extract_graph_payload(content)
            for concept, confidence in concepts:
                if confidence < self.min_confidence:
                    continue
                self.client.run_write(
                    """
                    MERGE (k:Concept {name: $concept})
                    MERGE (c:Chunk {id: $chunk_id})
                    MERGE (t:Topic {id: $topic_id})
                    MERGE (c)-[r:MENTIONS]->(k)
                    SET r.confidence = $confidence
                    MERGE (k)-[:RELATED_TO]->(t)
                    """,
                    {
                        "concept": concept,
                        "chunk_id": str(chunk_id),
                        "topic_id": topic_id,
                        "confidence": float(confidence),
                    },
                )
                concepts_count += 1

            for relation in relations:
                relation_type = relation.get("type", "RELATED_TO")
                confidence = float(relation.get("confidence", 0.0))
                if relation_type not in ALLOWED_RELATIONS or confidence < self.min_confidence:
                    continue

                self.client.run_write(
                    f"""
                    MERGE (a:Concept {{name: $source}})
                    MERGE (b:Concept {{name: $target}})
                    MERGE (a)-[r:{relation_type}]->(b)
                    SET r.confidence = $confidence,
                        r.chunk_id = $chunk_id,
                        r.document_id = $document_id,
                        r.topic = $topic
                    """,
                    {
                        "source": relation["source"],
                        "target": relation["target"],
                        "confidence": confidence,
                        "chunk_id": str(chunk_id),
                        "document_id": doc_id,
                        "topic": collection,
                    },
                )
                relations_count += 1

        return {
            "documents": len(unique_docs),
            "chunks": chunks_count,
            "concepts": concepts_count,
            "relations": relations_count,
        }

    def _extract_graph_payload(
        self,
        text: str,
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]]]:
        """
        Extract concepts and relations using configured mode:
        - llm_structured: strict JSON extraction + heuristic fallback
        - heuristic_fallback: heuristic extraction only
        """
        if (
            self.extraction_mode == "llm_structured"
            and self.structured_extractor is not None
        ):
            llm_concepts, llm_relations = self._extract_structured_graph_llm(
                text=text,
                max_concepts=max_concepts,
                max_relations=max_relations,
            )
            if llm_concepts or llm_relations:
                return llm_concepts, llm_relations

        return self._extract_structured_graph(
            text=text,
            max_concepts=max_concepts,
            max_relations=max_relations,
        )

    def _extract_structured_graph_llm(
        self,
        text: str,
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]]]:
        if self.structured_extractor is None:
            return [], []

        try:
            payload = self.structured_extractor(text) or {}
            concepts_raw = payload.get("concepts", [])
            relations_raw = payload.get("relations", [])
        except Exception as exc:
            logger.warning("Structured graph extraction failed, fallback heuristic: %s", exc)
            return [], []

        concepts: List[Tuple[str, float]] = []
        seen_concepts = set()
        for item in concepts_raw:
            if isinstance(item, dict):
                concept_name = self._canonical(str(item.get("name", "")))
                confidence = float(item.get("confidence", 0.0) or 0.0)
            else:
                concept_name = self._canonical(str(item))
                confidence = 0.75
            if not concept_name or concept_name in seen_concepts:
                continue
            seen_concepts.add(concept_name)
            concepts.append((concept_name, max(0.0, min(1.0, confidence))))
            if len(concepts) >= max_concepts:
                break

        relations: List[Dict[str, Any]] = []
        seen_relations = set()
        for item in relations_raw:
            if not isinstance(item, dict):
                continue
            source = self._canonical(str(item.get("source", "")))
            target = self._canonical(str(item.get("target", "")))
            relation_type = str(item.get("type", "RELATED_TO")).strip().upper()
            confidence = float(item.get("confidence", 0.0) or 0.0)

            if not source or not target or source == target:
                continue
            if relation_type not in ALLOWED_RELATIONS:
                relation_type = "RELATED_TO"

            relation_key = (source, target, relation_type)
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)

            relations.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
            if len(relations) >= max_relations:
                break

        return concepts, relations

    def ingest_memory_artifacts(self, artifacts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Project canonical Postgres memory artifacts into Neo4j."""
        if not self.enabled:
            return {"artifacts": 0, "concepts": 0, "relations": 0}

        artifacts_count = 0
        concepts_count = 0
        relations_count = 0

        for artifact in artifacts:
            artifact_id = str(artifact.get("id") or "")
            if not artifact_id:
                continue
            artifact_type = artifact.get("artifact_type", "memory")
            title = artifact.get("title", "")
            content = artifact.get("content", "")
            collection = artifact.get("topic_collection", "") or "general"
            topic_id = f"topic:{collection.lower()}"

            self.client.run_write(
                """
                MERGE (t:Topic {id: $topic_id})
                SET t.name = $topic_name
                MERGE (a:Artifact {id: $artifact_id})
                SET a.type = $artifact_type,
                    a.title = $title,
                    a.source_table = $source_table,
                    a.source_id = $source_id,
                    a.content = $content
                MERGE (a)-[:ABOUT_TOPIC]->(t)
                """,
                {
                    "topic_id": topic_id,
                    "topic_name": collection,
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "title": title,
                    "source_table": artifact.get("source_table", ""),
                    "source_id": str(artifact.get("source_id", "")),
                    "content": content[:3000],
                },
            )
            artifacts_count += 1

            concepts, relations = self._extract_graph_payload(content or title)
            for concept, confidence in concepts:
                if confidence < self.min_confidence:
                    continue
                self.client.run_write(
                    """
                    MERGE (k:Concept {name: $concept})
                    MERGE (a:Artifact {id: $artifact_id})
                    MERGE (t:Topic {id: $topic_id})
                    MERGE (a)-[r:MENTIONS]->(k)
                    SET r.confidence = $confidence
                    MERGE (k)-[:RELATED_TO]->(t)
                    """,
                    {
                        "concept": concept,
                        "artifact_id": artifact_id,
                        "topic_id": topic_id,
                        "confidence": float(confidence),
                    },
                )
                concepts_count += 1

            for relation in relations:
                relation_type = relation.get("type", "RELATED_TO")
                confidence = float(relation.get("confidence", 0.0))
                if relation_type not in ALLOWED_RELATIONS or confidence < self.min_confidence:
                    continue
                self.client.run_write(
                    f"""
                    MERGE (a:Concept {{name: $source}})
                    MERGE (b:Concept {{name: $target}})
                    MERGE (a)-[r:{relation_type}]->(b)
                    SET r.confidence = $confidence,
                        r.artifact_id = $artifact_id,
                        r.topic = $topic
                    """,
                    {
                        "source": relation["source"],
                        "target": relation["target"],
                        "confidence": confidence,
                        "artifact_id": artifact_id,
                        "topic": collection,
                    },
                )
                relations_count += 1

        return {
            "artifacts": artifacts_count,
            "concepts": concepts_count,
            "relations": relations_count,
        }

    def rebuild_projection(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        memory_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Delete derived Neo4j projection and rebuild it from Postgres data."""
        if not self.enabled:
            return {
                "deleted": 0,
                "documents": 0,
                "chunks": 0,
                "artifacts": 0,
                "concepts": 0,
                "relations": 0,
            }

        delete_result = self.client.run_read(
            "MATCH (n) WITH count(n) AS count DETACH DELETE n RETURN count",
            {},
        )
        deleted = int(delete_result[0].get("count", 0)) if delete_result else 0

        chunk_result = self.ingest_chunks(chunks or [])
        artifact_result = self.ingest_memory_artifacts(memory_artifacts or [])
        return {
            "deleted": deleted,
            "documents": int(chunk_result.get("documents", 0)),
            "chunks": int(chunk_result.get("chunks", 0)),
            "artifacts": int(artifact_result.get("artifacts", 0)),
            "concepts": int(chunk_result.get("concepts", 0)) + int(artifact_result.get("concepts", 0)),
            "relations": int(chunk_result.get("relations", 0)) + int(artifact_result.get("relations", 0)),
        }

    def ingest_decision(
        self,
        decision_id: str,
        title: str,
        rationale: str,
        topic_collection: str,
    ) -> bool:
        if not self.enabled:
            return False

        return self.client.run_write(
            """
            MERGE (t:Topic {id: $topic_id})
            SET t.name = $topic_name
            MERGE (d:Decision {id: $decision_id})
            SET d.title = $title,
                d.rationale = $rationale
            MERGE (d)-[:DECIDED_IN]->(t)
            """,
            {
                "topic_id": f"topic:{(topic_collection or 'general').lower()}",
                "topic_name": topic_collection or "general",
                "decision_id": decision_id,
                "title": title,
                "rationale": rationale[:4000],
            },
        )

    def _extract_structured_graph(
        self,
        text: str,
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]]]:
        """Extract canonical concepts and typed relations from chunk text."""
        concepts = self._extract_concepts(text, max_concepts=max_concepts)
        concept_with_confidence = [(concept, 0.78) for concept in concepts]
        if len(concepts) < 2:
            return concept_with_confidence, []

        relations: List[Dict[str, Any]] = []
        seen = set()
        sentences = re.split(r"[.!?\n]+", text)
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if not sentence_clean:
                continue

            sentence_concepts = self._extract_concepts(sentence, max_concepts=4)
            if len(sentence_concepts) < 2:
                continue

            relation_type = self._detect_relation_type(sentence_clean)
            base_confidence = 0.82 if relation_type != "RELATED_TO" else 0.72

            for left, right in combinations(sentence_concepts[:4], 2):
                left_norm = self._canonical(left)
                right_norm = self._canonical(right)
                if not left_norm or not right_norm or left_norm == right_norm:
                    continue
                key = (left_norm, right_norm, relation_type)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    {
                        "source": left_norm,
                        "target": right_norm,
                        "type": relation_type,
                        "confidence": base_confidence,
                    }
                )
                if len(relations) >= max_relations:
                    return concept_with_confidence, relations

        return concept_with_confidence, relations

    def _detect_relation_type(self, sentence: str) -> str:
        for relation_type, markers in RELATION_MARKERS.items():
            if any(marker in sentence for marker in markers):
                return relation_type
        return "RELATED_TO"

    @classmethod
    def _extract_concepts(cls, text: str, max_concepts: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_\-]{4,}", text.lower())
        filtered = [
            cls._canonical(token)
            for token in tokens
            if token not in STOPWORDS and not token.isdigit()
        ]
        filtered = [token for token in filtered if token]
        if not filtered:
            return []

        concepts: List[str] = []
        seen = set()
        for token in filtered:
            if token in seen:
                continue
            seen.add(token)
            concepts.append(token)
            if len(concepts) >= max_concepts:
                break
        return concepts

    @staticmethod
    def _canonical(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", (value or "").strip().lower())
        cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)
        return cleaned
