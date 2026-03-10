"""
Evidence fusion helper for vector + graph retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .models import EvidenceItem


class HybridEvidenceFusion:
    """Combine vector and graph evidence into a reranked list."""

    def __init__(
        self,
        vector_weight: float = 0.6,
        graph_weight: float = 0.3,
        lexical_weight: float = 0.1,
    ):
        self.vector_weight = float(vector_weight)
        self.graph_weight = float(graph_weight)
        self.lexical_weight = float(lexical_weight)

    @staticmethod
    def estimate_vector_confidence(vector_docs: List[Dict[str, Any]]) -> float:
        if not vector_docs:
            return 0.0
        top_scores = [float(doc.get("retrieval_score", 0.0)) for doc in vector_docs[:3]]
        if not top_scores:
            return 0.0
        return max(0.0, min(1.0, sum(top_scores) / len(top_scores)))

    def fuse(
        self,
        vector_docs: List[Dict[str, Any]],
        graph_paths: List[Dict[str, Any]],
        rerank_top_k: int = 8,
    ) -> Tuple[List[EvidenceItem], List[Dict[str, Any]]]:
        evidence: List[EvidenceItem] = []
        provenance: List[Dict[str, Any]] = []

        vector_scores = self._normalize([float(item.get("retrieval_score", 0.0)) for item in vector_docs])
        graph_scores = self._normalize([float(item.get("score", 0.0)) for item in graph_paths])

        for idx, doc in enumerate(vector_docs):
            semantic = float(doc.get("semantic_score", doc.get("retrieval_score", 0.0)))
            lexical = float(doc.get("keyword_score", 0.0))
            semantic_norm = vector_scores[idx] if idx < len(vector_scores) else 0.0
            lexical_norm = max(0.0, min(1.0, lexical))
            final_score = self.vector_weight * semantic_norm + self.lexical_weight * lexical_norm

            evidence_id = f"D{idx + 1}"
            source_ref = doc.get("metadata", {}).copy()
            source_ref["citation"] = evidence_id
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    kind="document",
                    score=final_score,
                    content=doc.get("content", ""),
                    source_ref=source_ref,
                )
            )
            provenance.append(
                {
                    "citation": evidence_id,
                    "kind": "document",
                    "score": final_score,
                    "source": source_ref,
                }
            )

        for idx, item in enumerate(graph_paths):
            graph_norm = graph_scores[idx] if idx < len(graph_scores) else 0.0
            final_score = self.graph_weight * graph_norm
            evidence_id = f"G{idx + 1}"
            source_ref = {
                "concept": item.get("concept", ""),
                "document_id": item.get("document_id", ""),
                "filename": item.get("filename", ""),
                "collection": item.get("collection", ""),
                "path": item.get("path", []),
                "citation": evidence_id,
            }
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    kind="graph",
                    score=final_score,
                    content=item.get("content", ""),
                    source_ref=source_ref,
                )
            )
            provenance.append(
                {
                    "citation": evidence_id,
                    "kind": "graph",
                    "score": final_score,
                    "source": source_ref,
                }
            )

        evidence.sort(key=lambda item: item.score, reverse=True)
        if rerank_top_k > 0:
            evidence = evidence[: int(rerank_top_k)]

        allowed_ids = {item.evidence_id for item in evidence}
        filtered_provenance = [item for item in provenance if item.get("citation") in allowed_ids]
        filtered_provenance.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        return evidence, filtered_provenance

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-9:
            return [1.0 if value > 0 else 0.0 for value in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

