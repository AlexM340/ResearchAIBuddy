"""
Neo4j client wrapper for graph ingestion and querying.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except Exception:
    GraphDatabase = None
    NEO4J_AVAILABLE = False


class Neo4jClient:
    """Simple neo4j wrapper with graceful disable mode."""

    def __init__(self, uri: str = "", user: str = "", password: str = "", enabled: bool = True):
        self.uri = (uri or "").strip()
        self.user = (user or "").strip()
        self.password = (password or "").strip()
        self.enabled = bool(enabled and self.uri and self.user and self.password and NEO4J_AVAILABLE)
        self._driver = None

        if enabled and not NEO4J_AVAILABLE:
            logger.warning("neo4j driver nu este disponibil; graph layer dezactivat.")

        if self.enabled:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except Exception as exc:
                logger.warning("Neo4j init failed, graph layer disabled: %s", exc)
                self.enabled = False
                self._driver = None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def ensure_constraints(self) -> bool:
        if not self.enabled or self._driver is None:
            return False
        try:
            statements = [
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            ]
            with self._driver.session() as session:
                for stmt in statements:
                    session.run(stmt)
            return True
        except Exception as exc:
            logger.warning("Neo4j constraint init failed: %s", exc)
            return False

    def run_write(self, cypher: str, parameters: Dict[str, Any] | None = None) -> bool:
        if not self.enabled or self._driver is None:
            return False
        try:
            with self._driver.session() as session:
                session.run(cypher, parameters or {})
            return True
        except Exception as exc:
            logger.error("Neo4j write failed: %s", exc)
            return False

    def run_read(self, cypher: str, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        if not self.enabled or self._driver is None:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except Exception as exc:
            logger.error("Neo4j read failed: %s", exc)
            return []

    def get_graph_stats(self) -> Dict[str, Any]:
        """Returns node and relation counts for dashboard metrics."""
        if not self.enabled:
            return {}
        try:
            def _count(cypher: str) -> int:
                rows = self.run_read(cypher)
                return int(rows[0]["n"]) if rows else 0

            return {
                "concepts": _count("MATCH (c:Concept) RETURN count(c) AS n"),
                "relations": _count("MATCH ()-[r]->() RETURN count(r) AS n"),
                "contradictions": _count("MATCH ()-[r:CONTRADICTS]->() RETURN count(r) AS n"),
                "documents": _count("MATCH (d:Document) RETURN count(d) AS n"),
            }
        except Exception as exc:
            logger.error("get_graph_stats failed: %s", exc)
            return {}

    def get_contradictions(self, limit: int = 20, include_dismissed: bool = False) -> List[Dict[str, Any]]:
        """Returns concept pairs in CONTRADICTS relations with source documents.

        Each row: {source, target, confidence, documents: [filename...], topic, dismissed}
        Implicit exclude relatii marcate cu dismissed=true.
        """
        if not self.enabled:
            return []
        try:
            where_dismissed = "" if include_dismissed else "WHERE coalesce(r.dismissed, false) = false"
            cypher = f"""
            MATCH (a:Concept)-[r:CONTRADICTS]->(b:Concept)
            {where_dismissed}
            OPTIONAL MATCH (ch:Chunk)-[:MENTIONS]->(a)
            OPTIONAL MATCH (ch)-[:SOURCED_FROM]->(d:Document)
            OPTIONAL MATCH (a)-[:RELATED_TO]->(t:Topic)
            WITH a, b, r, t,
                 collect(DISTINCT d.filename) AS doc_names
            RETURN a.name AS source,
                   b.name AS target,
                   coalesce(r.confidence, 0.0) AS confidence,
                   coalesce(t.name, 'general') AS topic,
                   doc_names AS documents,
                   coalesce(r.dismissed, false) AS dismissed
            ORDER BY confidence DESC
            LIMIT $limit
            """
            rows = self.run_read(cypher, {"limit": int(limit)})
            return [
                {
                    "source": row.get("source", ""),
                    "target": row.get("target", ""),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "topic": row.get("topic", "general") or "general",
                    "documents": [d for d in (row.get("documents") or []) if d],
                    "dismissed": bool(row.get("dismissed", False)),
                }
                for row in rows
                if row.get("source") and row.get("target")
            ]
        except Exception as exc:
            logger.error("get_contradictions failed: %s", exc)
            return []

    def dismiss_contradiction(self, source: str, target: str, note: str = "") -> bool:
        """Marcheaza o relatie CONTRADICTS ca dismissed (rezolvata manual)."""
        if not self.enabled:
            return False
        cypher = """
        MATCH (a:Concept {name: $source})-[r:CONTRADICTS]->(b:Concept {name: $target})
        SET r.dismissed = true,
            r.dismissed_at = datetime(),
            r.dismiss_note = $note
        RETURN count(r) AS updated
        """
        try:
            rows = self.run_read(cypher, {"source": source, "target": target, "note": note or ""})
            return bool(rows and int(rows[0].get("updated", 0) or 0) > 0)
        except Exception as exc:
            logger.error("dismiss_contradiction failed: %s", exc)
            return False

    def restore_contradiction(self, source: str, target: str) -> bool:
        """Anuleaza dismissed pe o relatie CONTRADICTS."""
        if not self.enabled:
            return False
        cypher = """
        MATCH (a:Concept {name: $source})-[r:CONTRADICTS]->(b:Concept {name: $target})
        REMOVE r.dismissed, r.dismissed_at, r.dismiss_note
        RETURN count(r) AS updated
        """
        try:
            rows = self.run_read(cypher, {"source": source, "target": target})
            return bool(rows and int(rows[0].get("updated", 0) or 0) > 0)
        except Exception as exc:
            logger.error("restore_contradiction failed: %s", exc)
            return False

    def get_graph_viz_data(
        self,
        limit_nodes: int = 150,
        relation_types: List[str] | None = None,
        collection_filter: str | None = None,
    ) -> Dict[str, Any]:
        """Fetches nodes and edges for interactive visualization.

        Returns {"nodes": [...], "edges": [...]} where each node has
        {id, label, size, collection} and each edge has {source, target, type}.
        """
        if not self.enabled:
            return {"nodes": [], "edges": []}
        try:
            rel_filter = ""
            if relation_types:
                types_str = "|".join(relation_types)
                rel_filter = f"[r:{types_str}]"
            else:
                rel_filter = "[r:RELATED_TO|DEPENDS_ON|CONTRADICTS|DERIVED_FROM]"

            params: Dict[str, Any] = {"limit": limit_nodes}
            coll_match = ""
            coll_where = ""
            if collection_filter:
                coll_match = "MATCH (k)-[:RELATED_TO]->(t:Topic {name: $collection}) "
                params["collection"] = collection_filter

            node_rows = self.run_read(
                "MATCH (k:Concept) "
                + coll_match
                + "OPTIONAL MATCH (ch:Chunk)-[:MENTIONS]->(k) "
                "OPTIONAL MATCH (k)-[:RELATED_TO]->(t:Topic) "
                "RETURN k.name AS label, "
                "       head(collect(t.name)) AS collection, "
                "       count(ch) AS mention_count "
                "ORDER BY mention_count DESC LIMIT $limit",
                params,
            )
            if not node_rows:
                return {"nodes": [], "edges": []}

            node_ids = {r["label"] for r in node_rows}
            nodes = [
                {
                    "id": r["label"],
                    "label": r["label"],
                    "size": max(8, min(40, int(r.get("mention_count") or 1) * 2)),
                    "collection": r.get("collection") or "general",
                }
                for r in node_rows
            ]

            edge_rows = self.run_read(
                f"MATCH (a:Concept)-{rel_filter}->(b:Concept) "
                "WHERE a.name IN $names AND b.name IN $names "
                "RETURN a.name AS source, b.name AS target, type(r) AS type "
                "LIMIT 500",
                {"names": list(node_ids)},
            )
            edges = [
                {"source": r["source"], "target": r["target"], "type": r["type"]}
                for r in edge_rows
                if r["source"] in node_ids and r["target"] in node_ids
            ]
            return {"nodes": nodes, "edges": edges}
        except Exception as exc:
            logger.error("get_graph_viz_data failed: %s", exc)
            return {"nodes": [], "edges": []}

