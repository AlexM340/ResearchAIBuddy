"""Pas 2 evaluare: ruleaza cele 26 intrebari F/X/C/G x configuratiile A-D.

Pentru fiecare rulare capteaza: ruta, latenta, retrieved_chunk_ids (ordonat),
citari valide/invalide, flagurile used_hyde/reranker/web, graph_hits si raspunsul.

- test_mode e ON (fara memorie personala in context) -> retrieval curat.
- response cache golit la inceputul fiecarei configuratii (fara cache hits).
- scope `topic` pe notebook-ul de evaluare (fara scurgeri din 'general').

Scrie results/runs.jsonl (sursa pentru report.py).

Utilizare (din radacina proiectului):
    python scripts/eval/run.py                 # toate configuratiile A-D
    python scripts/eval/run.py --configs A B   # doar unele
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _bootstrap import CONFIGS, DATA_DIR, NOTEBOOK, apply_config, build_system  # type: ignore

QUESTIONS = DATA_DIR / "questions.json"
RESULTS_DIR = DATA_DIR / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"
RETRIEVAL_CATS = {"F", "X", "C", "G"}


def _last_log(system) -> dict:
    """Ultima linie din retrieval_logs = metricile complete ale ultimei interogari."""
    try:
        with system.repository.client.connection() as conn:
            if conn is None:
                return {}
            with conn.cursor() as cur:
                cur.execute("SELECT route_used, latency_ms, metrics_json FROM retrieval_logs ORDER BY id DESC LIMIT 1;")
                row = cur.fetchone()
            conn.commit()
        if not row:
            return {}
        metrics = row[2] or {}
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        return {"route_used": row[0], "latency_ms": float(row[1] or 0.0), "metrics": metrics}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner A-D pentru intrebarile de retrieval")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), help="Subset de configuratii (A B C D)")
    args = ap.parse_args()

    configs = [c.upper() for c in args.configs if c.upper() in CONFIGS]
    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["category"] in RETRIEVAL_CATS]

    print(f"Construiesc sistemul... ({len(questions)} intrebari x {len(configs)} configuratii)")
    system = build_system()
    if not (getattr(system, "repository", None) and system.repository.enabled):
        print("EROARE: Postgres indisponibil.")
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Merge: pastram rularile pentru configuratiile care NU se ruleaza acum,
    # ca sa poti rula A/B/C acum si D mai tarziu (dupa ce se termina graful).
    kept = []
    if RUNS_FILE.exists():
        for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("configuration") not in configs:
                kept.append(line)

    written = 0
    with RUNS_FILE.open("w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")
        if kept:
            print(f"(pastrate {len(kept)} rulari pentru alte configuratii)")
        for cfg in configs:
            apply_config(system, cfg)
            system.clear_cache()  # fara cache hits in evaluare
            print(f"\n=== Configuratia {cfg} ===")
            for q in questions:
                t0 = time.time()
                try:
                    res = system.query(q["text"], retrieval_mode="topic", active_collection=NOTEBOOK)
                except Exception as exc:
                    print(f"  {q['id']}: EROARE {exc}")
                    continue
                log = _last_log(system)
                m = log.get("metrics", {})
                record = {
                    "question_id": q["id"],
                    "category": q["category"],
                    "configuration": cfg,
                    "expected_route": q.get("expected_route", ""),
                    "route_used": log.get("route_used") or res.get("route_used", ""),
                    "latency_ms": log.get("latency_ms") or round((time.time() - t0) * 1000.0, 1),
                    "used_hyde": m.get("used_hyde"),
                    "used_reranker": m.get("used_reranker"),
                    "used_web": m.get("used_web"),
                    "graph_hits": m.get("graph_hits", 0),
                    "citation_used": m.get("citation_used", []),
                    "citation_invalid": m.get("citation_invalid", []),
                    "retrieved_chunk_ids": m.get("retrieved_chunk_ids", []),
                    # Evidentele de graf (concept + snippet) — pentru scorarea manuala
                    # a contributiei grafului la intrebarile C/G in config D.
                    "graph_evidence": [
                        {"concept": g.get("concept", ""), "content": (g.get("content", "") or "").replace("\n", " ")[:220]}
                        for g in (res.get("graph_sources", []) or [])
                    ],
                    "answer": res.get("response", ""),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                print(
                    f"  {q['id']}: route={record['route_used']:<8} "
                    f"lat={record['latency_ms']:.0f}ms chunks={len(record['retrieved_chunk_ids'])} "
                    f"graph={record['graph_hits']}"
                )

    print(f"\nGata. {written} rulari scrise in {RUNS_FILE}")
    print("Urmeaza: python scripts/eval/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
