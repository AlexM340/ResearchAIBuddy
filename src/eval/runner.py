"""
Simple A/B evaluation runner for baseline vs hybrid retrieval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from .metrics import average_latency, groundedness_rate, p95_latency, route_accuracy


def run_ab_evaluation(
    dataset: List[Dict[str, Any]],
    baseline_fn: Callable[[str], Dict[str, Any]],
    hybrid_fn: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_grounded: List[bool] = []
    hybrid_grounded: List[bool] = []
    baseline_latencies: List[float] = []
    hybrid_latencies: List[float] = []
    expected_routes: List[str] = []
    baseline_routes: List[str] = []
    hybrid_routes: List[str] = []
    detailed: List[Dict[str, Any]] = []

    for sample in dataset:
        question = sample.get("question", "")
        expected_route = sample.get("expected_route", "")
        baseline_result = baseline_fn(question)
        hybrid_result = hybrid_fn(question)

        baseline_grounded.append(bool(baseline_result.get("provenance")))
        hybrid_grounded.append(bool(hybrid_result.get("provenance")))
        baseline_latencies.append(float(baseline_result.get("response_time", 0.0) or 0.0))
        hybrid_latencies.append(float(hybrid_result.get("response_time", 0.0) or 0.0))
        expected_routes.append(expected_route)
        baseline_routes.append(baseline_result.get("route_used", "vector"))
        hybrid_routes.append(hybrid_result.get("route_used", "vector"))

        detailed.append(
            {
                "question": question,
                "query_type": sample.get("query_type", "generic"),
                "expected_route": expected_route,
                "baseline": {
                    "route_used": baseline_result.get("route_used", "vector"),
                    "grounded": bool(baseline_result.get("provenance")),
                    "latency_s": float(baseline_result.get("response_time", 0.0) or 0.0),
                },
                "hybrid": {
                    "route_used": hybrid_result.get("route_used", "vector"),
                    "grounded": bool(hybrid_result.get("provenance")),
                    "latency_s": float(hybrid_result.get("response_time", 0.0) or 0.0),
                },
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(dataset),
        "baseline": {
            "groundedness": groundedness_rate(baseline_grounded),
            "avg_latency_s": average_latency(baseline_latencies),
            "p95_latency_s": p95_latency(baseline_latencies),
            "route_accuracy": route_accuracy(expected_routes, baseline_routes),
        },
        "hybrid": {
            "groundedness": groundedness_rate(hybrid_grounded),
            "avg_latency_s": average_latency(hybrid_latencies),
            "p95_latency_s": p95_latency(hybrid_latencies),
            "route_accuracy": route_accuracy(expected_routes, hybrid_routes),
        },
        "improvement": {
            "groundedness_delta": groundedness_rate(hybrid_grounded) - groundedness_rate(baseline_grounded),
            "route_accuracy_delta": route_accuracy(expected_routes, hybrid_routes) - route_accuracy(expected_routes, baseline_routes),
            "avg_latency_delta_s": average_latency(hybrid_latencies) - average_latency(baseline_latencies),
        },
        "details": detailed,
    }
    return report


def export_report(report: Dict[str, Any], output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target.resolve())


def export_report_markdown(report: Dict[str, Any], output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    baseline = report.get("baseline", {})
    hybrid = report.get("hybrid", {})
    improvement = report.get("improvement", {})
    samples = int(report.get("samples", 0) or 0)
    generated_at = report.get("generated_at", "")

    lines = [
        "# Second Brain Evaluation Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Samples: `{samples}`",
        "",
        "## Metrics",
        "",
        "| Variant | Groundedness | Route Accuracy | Avg Latency (s) | P95 Latency (s) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Baseline | {baseline.get('groundedness', 0.0):.3f} | "
            f"{baseline.get('route_accuracy', 0.0):.3f} | "
            f"{baseline.get('avg_latency_s', 0.0):.3f} | "
            f"{baseline.get('p95_latency_s', 0.0):.3f} |"
        ),
        (
            f"| Hybrid | {hybrid.get('groundedness', 0.0):.3f} | "
            f"{hybrid.get('route_accuracy', 0.0):.3f} | "
            f"{hybrid.get('avg_latency_s', 0.0):.3f} | "
            f"{hybrid.get('p95_latency_s', 0.0):.3f} |"
        ),
        "",
        "## Delta (Hybrid - Baseline)",
        "",
        f"- Groundedness delta: `{improvement.get('groundedness_delta', 0.0):.3f}`",
        f"- Route accuracy delta: `{improvement.get('route_accuracy_delta', 0.0):.3f}`",
        f"- Avg latency delta (s): `{improvement.get('avg_latency_delta_s', 0.0):.3f}`",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target.resolve())
