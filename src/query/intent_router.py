"""
Rule-based router that selects vector-only vs hybrid vector+graph retrieval.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .models import QueryIntent, RetrievalPlan

RELATIONAL_MARKERS = {
    "de ce", "depinde", "legatura", "relatie", "conex", "impact", "cauza",
    "decizie", "decis", "compar", "avantaje", "dezavantaje"
}
TEMPORAL_MARKERS = {
    "cand", "ultimele", "ieri", "azi", "maine", "saptamana", "luna", "an",
    "istoric", "evolutie", "inainte", "dupa", "recent"
}
REMINDER_MARKERS = {
    "reaminteste", "adu-mi aminte", "am uitat", "remember", "ce am decis"
}


class RuleBasedIntentRouter:
    """Classify user question intent and build a retrieval plan."""

    def __init__(self, vector_confidence_threshold: float = 0.35):
        self.vector_confidence_threshold = float(vector_confidence_threshold)

    def classify(self, question: str, time_window_days: Optional[int] = None) -> QueryIntent:
        text = (question or "").lower().strip()
        entities = self._extract_entities(text)

        if self._contains_any(text, REMINDER_MARKERS):
            return QueryIntent(
                intent_type="reminder",
                confidence=0.84,
                temporal_scope=self._infer_temporal_scope(text, time_window_days),
                entities=entities,
                reason="question appears to request memory recall",
            )

        if self._contains_any(text, RELATIONAL_MARKERS):
            return QueryIntent(
                intent_type="relational",
                confidence=0.80,
                temporal_scope=self._infer_temporal_scope(text, time_window_days),
                entities=entities,
                reason="question includes relational markers",
            )

        if self._contains_any(text, TEMPORAL_MARKERS):
            return QueryIntent(
                intent_type="temporal",
                confidence=0.75,
                temporal_scope=self._infer_temporal_scope(text, time_window_days),
                entities=entities,
                reason="question includes temporal markers",
            )

        return QueryIntent(
            intent_type="factual",
            confidence=0.72,
            temporal_scope=self._infer_temporal_scope(text, time_window_days),
            entities=entities,
            reason="default factual classification",
        )

    def build_plan(
        self,
        intent: QueryIntent,
        retrieval_mode: str,
        vector_top_k: int,
        graph_top_k_paths: int,
        hybrid_rerank_top_k: int,
        vector_confidence: Optional[float] = None,
    ) -> RetrievalPlan:
        normalized_mode = (retrieval_mode or "auto").strip().lower()

        if normalized_mode in {"vector", "topic", "topic_general", "all"}:
            if intent.intent_type in {"relational", "temporal", "reminder"}:
                route = "hybrid"
                reason = f"rule_intent_{intent.intent_type}"
                use_graph = True
            else:
                route = "vector"
                reason = "rule_intent_factual"
                use_graph = False
        elif normalized_mode == "hybrid":
            route = "hybrid"
            reason = "explicit_hybrid_mode"
            use_graph = True
        else:
            route = "vector"
            reason = "auto_default_vector"
            use_graph = False

        if vector_confidence is not None and vector_confidence < self.vector_confidence_threshold:
            route = "hybrid"
            reason = (
                "low_vector_confidence:"
                f"{vector_confidence:.2f}<{self.vector_confidence_threshold:.2f}"
            )
            use_graph = True

        return RetrievalPlan(
            route=route,
            vector_k=max(1, int(vector_top_k)),
            graph_top_k_paths=max(1, int(graph_top_k_paths)),
            hybrid_rerank_top_k=max(1, int(hybrid_rerank_top_k)),
            use_vector=True,
            use_graph=use_graph,
            reason=reason,
        )

    @staticmethod
    def _contains_any(text: str, markers: set[str]) -> bool:
        return any(marker in text for marker in markers)

    @staticmethod
    def _infer_temporal_scope(text: str, time_window_days: Optional[int]) -> Optional[str]:
        if time_window_days is not None:
            return f"{int(time_window_days)}d"
        if any(token in text for token in {"azi", "ieri", "maine"}):
            return "day"
        if any(token in text for token in {"saptamana", "week"}):
            return "week"
        if any(token in text for token in {"luna", "month"}):
            return "month"
        if any(token in text for token in {"an", "year"}):
            return "year"
        return None

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        terms = re.findall(r"[a-zA-ZĂÂÎȘȚăâîșț0-9_\\-]{3,}", text)
        deduped: List[str] = []
        seen = set()
        for term in terms:
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(lowered)
        return deduped[:8]
