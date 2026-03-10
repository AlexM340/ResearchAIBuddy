"""
Dataset helpers for evaluation scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, List


def default_second_brain_eval_set() -> List[Dict[str, Any]]:
    return [
        {
            "question": "Ce am decis despre arhitectura proiectului si de ce?",
            "query_type": "decision_recall",
            "expected_route": "hybrid",
        },
        {
            "question": "Ce task-uri depind de decizia privind storage-ul?",
            "query_type": "multi_hop",
            "expected_route": "hybrid",
        },
        {
            "question": "Ce s-a schimbat in topicul retrieval in ultima luna?",
            "query_type": "temporal",
            "expected_route": "hybrid",
        },
        {
            "question": "Adu-mi aminte concluziile despre GraphRAG.",
            "query_type": "memory",
            "expected_route": "hybrid",
        },
        {
            "question": "Care este definitia retrieval-ului semantic?",
            "query_type": "factual",
            "expected_route": "vector",
        },
    ]
