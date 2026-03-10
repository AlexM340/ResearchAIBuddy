"""
Context budget builder for LLM prompt assembly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import EvidenceItem


class ContextBuilder:
    """Build a bounded context with explicit provenance."""

    def __init__(
        self,
        total_budget_tokens: int = 8000,
        vector_budget_ratio: float = 0.65,
        graph_budget_ratio: float = 0.35,
    ):
        self.total_budget_tokens = max(1000, int(total_budget_tokens))
        self.vector_budget_ratio = max(0.0, min(1.0, float(vector_budget_ratio)))
        self.graph_budget_ratio = max(0.0, min(1.0, float(graph_budget_ratio)))

    def build(
        self,
        question: str,
        evidence_items: List[EvidenceItem],
    ) -> Dict[str, Any]:
        vector_budget = int(self.total_budget_tokens * self.vector_budget_ratio)
        graph_budget = int(self.total_budget_tokens * self.graph_budget_ratio)

        vector_used = 0
        graph_used = 0
        selected: List[EvidenceItem] = []

        for item in evidence_items:
            token_estimate = self._estimate_tokens(item.content)
            if item.kind == "document":
                if vector_used + token_estimate > vector_budget:
                    continue
                vector_used += token_estimate
            else:
                if graph_used + token_estimate > graph_budget:
                    continue
                graph_used += token_estimate
            selected.append(item)

        selected.sort(key=lambda item: item.score, reverse=True)

        context_blocks: List[str] = []
        provenance: List[Dict[str, Any]] = []
        vector_sources: List[Dict[str, Any]] = []
        graph_sources: List[Dict[str, Any]] = []

        for item in selected:
            content_preview = item.content[:800] + "..." if len(item.content) > 800 else item.content
            context_blocks.append(f"[{item.evidence_id}] ({item.kind}) {content_preview}")
            provenance.append(
                {
                    "citation": item.evidence_id,
                    "kind": item.kind,
                    "score": item.score,
                    "source": item.source_ref,
                }
            )
            if item.kind == "document":
                vector_sources.append(item.source_ref)
            else:
                graph_sources.append(item.source_ref)

        context_text = "\n\n".join(context_blocks)
        prompt = f"""Ești un asistent personal de tip Second Brain.

CONTEXT DOVEZI:
{context_text}

ÎNTREBARE:
{question}

INSTRUCȚIUNI:
- Răspunde strict pe baza dovezilor de mai sus.
- Citează explicit dovezile folosite: [D#] pentru documente și [G#] pentru graf.
- Dacă contextul este insuficient, răspunde exact: INSUFFICIENT_CONTEXT.
"""

        return {
            "prompt": prompt,
            "provenance": provenance,
            "vector_sources": vector_sources,
            "graph_sources": graph_sources,
            "vector_tokens_used": vector_used,
            "graph_tokens_used": graph_used,
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Fast rough estimate.
        return max(1, len(text) // 4)

