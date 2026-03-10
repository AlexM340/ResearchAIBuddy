from pathlib import Path

from src.eval.runner import export_report_markdown, run_ab_evaluation


def test_run_ab_evaluation_returns_extended_metrics():
    dataset = [
        {"question": "Q1", "expected_route": "vector", "query_type": "factual"},
        {"question": "Q2", "expected_route": "hybrid", "query_type": "multi_hop"},
    ]

    def baseline_fn(question: str):
        if question == "Q1":
            return {"provenance": [], "response_time": 1.0, "route_used": "vector"}
        return {"provenance": [{"citation": "D1"}], "response_time": 2.0, "route_used": "vector"}

    def hybrid_fn(question: str):
        if question == "Q1":
            return {"provenance": [{"citation": "D1"}], "response_time": 1.4, "route_used": "vector"}
        return {"provenance": [{"citation": "G1"}], "response_time": 2.8, "route_used": "hybrid"}

    report = run_ab_evaluation(dataset, baseline_fn=baseline_fn, hybrid_fn=hybrid_fn)

    assert report["samples"] == 2
    assert "baseline" in report and "hybrid" in report
    assert report["baseline"]["groundedness"] >= 0.0
    assert report["hybrid"]["groundedness"] >= report["baseline"]["groundedness"]
    assert "improvement" in report


def test_export_report_markdown_creates_file(tmp_path: Path):
    report = {
        "generated_at": "2026-03-10T00:00:00Z",
        "samples": 2,
        "baseline": {
            "groundedness": 0.5,
            "route_accuracy": 0.5,
            "avg_latency_s": 2.0,
            "p95_latency_s": 2.5,
        },
        "hybrid": {
            "groundedness": 1.0,
            "route_accuracy": 1.0,
            "avg_latency_s": 2.6,
            "p95_latency_s": 3.2,
        },
        "improvement": {
            "groundedness_delta": 0.5,
            "route_accuracy_delta": 0.5,
            "avg_latency_delta_s": 0.6,
        },
    }

    output = export_report_markdown(report, str(tmp_path / "report.md"))
    content = Path(output).read_text(encoding="utf-8")
    assert "Second Brain Evaluation Report" in content
    assert "Groundedness delta" in content
