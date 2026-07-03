"""Pas 3 evaluare: calculeaza metricile si tabelele pentru capitolul de testare.

Citeste results/runs.jsonl + ground_truth.json si produce:
  - tabelul principal A-D (Recall@5, MRR, citari valide, latenta medie, P50, P95);
  - tabelul pe categorii (Recall@5 / MRR per F/X/C/G);
  - route accuracy (raportata informativ; relevanta mai ales pentru D);
  - details.csv (toate rularile, pentru anexa) + report.md.

Nu incarca torch/sistemul — doar calcul pe fisierele de rezultate.

Utilizare (din radacina proiectului):
    python scripts/eval/report.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import average_latency, mrr, p95_latency, recall_at_k, route_accuracy  # noqa: E402

RESULTS_DIR = DATA_DIR / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"
GT_FILE = DATA_DIR / "ground_truth.json"
CATEGORIES = ["F", "X", "C", "G"]
CONFIG_LABELS = {
    "A": "A – vectorial",
    "B": "B – reranker",
    "C": "C – HyDE + reranker",
    "D": "D – sistem complet",
}


def _load_runs():
    if not RUNS_FILE.exists():
        print(f"Lipseste {RUNS_FILE}. Ruleaza intai scripts/eval/run.py.")
        sys.exit(2)
    return [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_relevants():
    if not GT_FILE.exists():
        print(f"Lipseste {GT_FILE}. Ruleaza scripts/eval/label.py si marcheaza relevanta.")
        sys.exit(2)
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    relevants = {}
    for qid, ent in gt.items():
        relevants[qid] = [str(c["chunk_id"]) for c in ent.get("candidates", []) if c.get("relevant")]
    return relevants


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> int:
    runs = _load_runs()
    relevants = _load_relevants()

    missing_gt = sorted({r["question_id"] for r in runs if not relevants.get(r["question_id"])})
    if missing_gt:
        print(f"AVERTISMENT: fara ground truth marcat pentru: {', '.join(missing_gt)}")
        print("(Recall@5/MRR vor fi 0 pentru ele. Marcheaza-le in ground_truth.json.)\n")

    # Imbogatim fiecare rulare cu recall@5 + RR.
    for r in runs:
        retrieved = [str(c) for c in r.get("retrieved_chunk_ids", [])]
        rel = relevants.get(r["question_id"], [])
        r["recall@5"] = recall_at_k(rel, retrieved, 5)
        r["rr"] = mrr(rel, retrieved)
        r["citation_valid_n"] = len(r.get("citation_used", []))
        r["citation_total_n"] = len(r.get("citation_used", [])) + len(r.get("citation_invalid", []))

    configs = sorted({r["configuration"] for r in runs})
    lines = ["# Raport evaluare CerebrumAI (A-D)", ""]

    # --- Tabelul principal ---
    lines += ["## Tabel principal (medii pe cele 26 de intrebari F/X/C/G)", ""]
    lines += ["| Configurație | Recall@5 | MRR | Citări valide | Latență medie (ms) | P50 (ms) | P95 (ms) |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|"]
    for cfg in configs:
        rows = [r for r in runs if r["configuration"] == cfg]
        recalls = [r["recall@5"] for r in rows]
        rrs = [r["rr"] for r in rows]
        lats = [float(r.get("latency_ms", 0.0)) for r in rows]
        valid = sum(r["citation_valid_n"] for r in rows)
        total = sum(r["citation_total_n"] for r in rows)
        cit = (valid / total) if total else 0.0
        med = statistics.median(lats) if lats else 0.0
        lines.append(
            f"| {CONFIG_LABELS.get(cfg, cfg)} | {_fmt(statistics.mean(recalls) if recalls else 0)} | "
            f"{_fmt(statistics.mean(rrs) if rrs else 0)} | {_fmt(cit)} | "
            f"{average_latency(lats):.0f} | {med:.0f} | {p95_latency(lats):.0f} |"
        )
    lines.append("")

    # --- Tabel pe categorii (Recall@5 / MRR) ---
    lines += ["## Recall@5 / MRR pe categorii", ""]
    header = "| Configurație | " + " | ".join(f"{c} (Recall@5 / MRR)" for c in CATEGORIES) + " |"
    lines += [header, "|---|" + "---:|" * len(CATEGORIES)]
    for cfg in configs:
        cells = []
        for cat in CATEGORIES:
            rows = [r for r in runs if r["configuration"] == cfg and r["category"] == cat]
            rec = statistics.mean([r["recall@5"] for r in rows]) if rows else 0.0
            rr = statistics.mean([r["rr"] for r in rows]) if rows else 0.0
            cells.append(f"{rec:.2f} / {rr:.2f}")
        lines.append(f"| {CONFIG_LABELS.get(cfg, cfg)} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Route accuracy (informativ) ---
    lines += ["## Route accuracy (informativ; relevant mai ales pentru D)", ""]
    lines += ["| Configurație | Route accuracy |", "|---|---:|"]
    for cfg in configs:
        rows = [r for r in runs if r["configuration"] == cfg]
        acc = route_accuracy([r.get("expected_route", "") for r in rows], [r.get("route_used", "") for r in rows])
        lines.append(f"| {CONFIG_LABELS.get(cfg, cfg)} | {_fmt(acc)} |")
    lines.append("")

    # --- Scoruri manuale (graf / memorie / sinteza), daca exista ---
    manual_file = DATA_DIR / "manual_scores.json"
    if manual_file.exists():
        manual = json.loads(manual_file.read_text(encoding="utf-8"))

        cov = {k: v for k, v in manual.get("answer_coverage", {}).items() if not k.startswith("_") and v is not None}
        if cov:
            lines += ["## Answer coverage (manual, config D)", "",
                      f"- Mediu: `{statistics.mean([float(v) for v in cov.values()]):.2f}` (pe {len(cov)} întrebări notate)", ""]

        graph = {k: v for k, v in manual.get("graph_contribution", {}).items() if not k.startswith("_") and v is not None}
        if graph:
            vals = list(graph.values())
            lines += ["## Contribuția grafului (manual, întrebări C/G în D)", ""]
            lines += [f"- Scor mediu: `{statistics.mean(vals):.2f}` (pe {len(vals)} întrebări notate)"]
            dist = {s: sum(1 for v in vals if v == s) for s in (2, 1, 0, -1)}
            lines += [f"- Distribuție: 2={dist[2]}, 1={dist[1]}, 0={dist[0]}, -1={dist[-1]}", ""]

        mem = {k: v for k, v in manual.get("memory", {}).items() if not k.startswith("_") and v is not None}
        if mem:
            acc = sum(float(v) for v in mem.values()) / len(mem)
            lines += ["## Memorie (manual, M01–M05)", "", f"- MemoryAccuracy: `{acc:.2f}` ({len(mem)} întrebări)", ""]

        synth = {k: v for k, v in manual.get("synthesis", {}).items() if not k.startswith("_") and isinstance(v, dict)}
        scored = {k: v for k, v in synth.items() if any(x is not None for x in v.values())}
        if scored:
            dims = ["acoperire", "corectitudine", "coerenta", "ancorare_surse", "utilitate"]
            lines += ["## Sinteză/taxonomie (manual, S01–S04, scor 1–5)", ""]
            lines += ["| Dimensiune | Medie |", "|---|---:|"]
            for d in dims:
                ds = [float(v[d]) for v in scored.values() if v.get(d) is not None]
                if ds:
                    lines.append(f"| {d} | {statistics.mean(ds):.2f} |")
            lines.append("")

    report_md = RESULTS_DIR / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- details.csv pentru anexa ---
    details = RESULTS_DIR / "details.csv"
    with details.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question_id", "category", "configuration", "recall@5", "rr", "route_used",
                    "expected_route", "latency_ms", "graph_hits", "citation_valid", "citation_total",
                    "retrieved_chunk_ids"])
        for r in sorted(runs, key=lambda x: (x["configuration"], x["question_id"])):
            w.writerow([
                r["question_id"], r["category"], r["configuration"], f"{r['recall@5']:.3f}", f"{r['rr']:.3f}",
                r.get("route_used", ""), r.get("expected_route", ""), f"{float(r.get('latency_ms', 0)):.0f}",
                r.get("graph_hits", 0), r["citation_valid_n"], r["citation_total_n"],
                ";".join(str(c) for c in r.get("retrieved_chunk_ids", [])),
            ])

    print("\n".join(lines))
    print(f"\nScris: {report_md}\nScris: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
