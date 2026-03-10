"""
Shared dataclasses for query planning and evidence handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryIntent:
    intent_type: str
    confidence: float
    temporal_scope: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class RetrievalPlan:
    route: str
    vector_k: int
    graph_top_k_paths: int
    hybrid_rerank_top_k: int
    use_vector: bool = True
    use_graph: bool = False
    reason: str = ""


@dataclass
class EvidenceItem:
    evidence_id: str
    kind: str
    score: float
    content: str
    source_ref: Dict[str, Any]


@dataclass
class DecisionMemory:
    id: str
    title: str
    rationale: str
    topic_collection: str
    confidence: float
    created_at: str
    memory_type: str = "semantic"
