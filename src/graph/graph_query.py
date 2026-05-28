"""Graph retrieval helper for hybrid VectorRAG + GraphRAG routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphQueryService:
    """Read-only graph query service for relation-aware evidence retrieval."""

    def __init__(self, neo4j_client: Neo4jClient, max_hops: int = 2):
        self.client = neo4j_client
        self.max_hops = max(1, int(max_hops))

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def retrieve_paths(
        self,
        question: str,
        active_collection: Optional[str] = None,
        max_paths: int = 6,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        terms = self._extract_terms(question)
        if not terms:
            return []

        cypher = f"""
        MATCH (seed:Concept)
        WHERE ANY(term IN $terms WHERE toLower(seed.name) CONTAINS term)
        MATCH p=(seed)-[r:RELATED_TO|DEPENDS_ON|CONTRADICTS|DERIVED_FROM*1..{self.max_hops}]-(linked:Concept)
        WHERE ALL(rel IN relationships(p) WHERE coalesce(rel.confidence, 0.0) >= $min_confidence)
        OPTIONAL MATCH (ch:Chunk)-[m:MENTIONS]->(linked)
        OPTIONAL MATCH (ch)-[:SOURCED_FROM]->(d:Document)-[:ABOUT_TOPIC]->(t:Topic)
        OPTIONAL MATCH (art:Artifact)-[:MENTIONS]->(linked)
        OPTIONAL MATCH (art)-[:ABOUT_TOPIC]->(at:Topic)
        WHERE ($topic = '' OR toLower(coalesce(t.name, at.name, '')) = toLower($topic))
        RETURN seed.name AS seed_concept,
               linked.name AS concept,
               ch.id AS chunk_id,
               substring(coalesce(ch.text, art.content, ''), 0, 700) AS chunk_text,
               coalesce(d.id, art.source_id, art.id) AS document_id,
               coalesce(d.filename, art.title, '') AS filename,
               coalesce(t.name, at.name, $topic) AS topic,
               art.id AS artifact_id,
               art.type AS artifact_type,
               [rel IN relationships(p) | type(rel)] AS relation_types,
               [rel IN relationships(p) | coalesce(rel.confidence, 0.0)] AS relation_scores,
               length(p) AS hops
        LIMIT $max_paths
        """
        rows = self.client.run_read(
            cypher,
            {
                "terms": terms,
                "topic": active_collection or "",
                "max_paths": int(max_paths),
                "min_confidence": 0.70,
            },
        )

        graph_sources: List[Dict[str, Any]] = []
        for row in rows:
            relation_scores = [float(value) for value in (row.get("relation_scores") or [])]
            avg_conf = sum(relation_scores) / len(relation_scores) if relation_scores else 0.7
            hops = max(1, int(row.get("hops") or 1))
            score = max(0.0, min(1.0, avg_conf * (1.0 / hops)))

            relation_types = row.get("relation_types") or ["RELATED_TO"]
            seed = row.get("seed_concept", "")
            linked = row.get("concept", "")
            path_repr = [seed] + relation_types + [linked]

            graph_sources.append(
                {
                    "concept": linked,
                    "seed_concept": seed,
                    "chunk_id": row.get("chunk_id", ""),
                    "content": row.get("chunk_text", ""),
                    "document_id": row.get("document_id", ""),
                    "filename": row.get("filename", ""),
                    "artifact_id": row.get("artifact_id", ""),
                    "artifact_type": row.get("artifact_type", ""),
                    "collection": row.get("topic", active_collection or ""),
                    "relation_types": relation_types,
                    "hops": hops,
                    "score": score,
                    "path": path_repr,
                }
            )

        graph_sources.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return graph_sources[: max_paths]

    def retrieve_decisions(
        self,
        question: str,
        active_collection: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        terms = self._extract_terms(question)
        if not terms:
            return []

        cypher = """
        MATCH (d:Decision)-[:DECIDED_IN]->(t:Topic)
        WHERE ($topic = '' OR toLower(t.name) = toLower($topic))
          AND ANY(term IN $terms WHERE toLower(d.title) CONTAINS term OR toLower(d.rationale) CONTAINS term)
        RETURN d.id AS id, d.title AS title, d.rationale AS rationale, t.name AS topic
        LIMIT $limit
        """
        rows = self.client.run_read(
            cypher,
            {"terms": terms, "topic": active_collection or "", "limit": int(limit)},
        )

        return [
            {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "rationale": row.get("rationale", ""),
                "topic_collection": row.get("topic", ""),
                "confidence": 0.8,
                "memory_type": "semantic",
                "source": "graph_decision",
            }
            for row in rows
        ]

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        terms = re.findall(r"[A-Za-z0-9_\-]{3,}", (text or "").lower())
        deduped: List[str] = []
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped[:10]
