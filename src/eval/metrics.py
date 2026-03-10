"""
Core evaluation metrics for retrieval and answer quality.
"""

from __future__ import annotations

from typing import Iterable, List


def recall_at_k(relevant_ids: Iterable[str], retrieved_ids: List[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[: max(1, int(k))])
    return len(relevant.intersection(top_k)) / len(relevant)


def mrr(relevant_ids: Iterable[str], retrieved_ids: List[str]) -> float:
    relevant = set(relevant_ids)
    for idx, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def groundedness_rate(answer_has_provenance: List[bool]) -> float:
    if not answer_has_provenance:
        return 0.0
    return sum(1 for flag in answer_has_provenance if flag) / len(answer_has_provenance)


def average_latency(latency_values: List[float]) -> float:
    if not latency_values:
        return 0.0
    return sum(float(value) for value in latency_values) / len(latency_values)


def p95_latency(latency_values: List[float]) -> float:
    if not latency_values:
        return 0.0
    sorted_values = sorted(float(value) for value in latency_values)
    index = max(0, int(round(0.95 * (len(sorted_values) - 1))))
    return sorted_values[index]


def route_accuracy(expected_routes: List[str], actual_routes: List[str]) -> float:
    if not expected_routes or len(expected_routes) != len(actual_routes):
        return 0.0
    total = len(expected_routes)
    correct = 0
    for expected, actual in zip(expected_routes, actual_routes):
        if (expected or "").strip().lower() == (actual or "").strip().lower():
            correct += 1
    return correct / total if total else 0.0
