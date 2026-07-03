"""Corecteaza graph_evidence in runs.jsonl cu textul real al chunk-urilor de graf.

Captura initiala din run.py a salvat doar eticheta conceptului (content gol),
pentru ca graph_sources contineau doar source_ref. Retrieval-ul din graf este
DETERMINIST (foloseste termenii intrebarii, nu HyDE), deci putem reconstrui exact
evidentele cu text fara sa re-rulam config D / fara sa regeneram raspunsurile.

Doar Neo4j (fara torch). Rescrie graph_evidence pentru rândurile C/G din config D.

Utilizare (din radacina proiectului):
    python scripts/eval/regraph.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
RUNS_FILE = DATA_DIR / "results" / "runs.jsonl"
QUESTIONS = DATA_DIR / "questions.json"
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from graph.graph_query import GraphQueryService  # noqa: E402
from graph.neo4j_client import Neo4jClient  # noqa: E402

NOTEBOOK = os.getenv("EVAL_NOTEBOOK", "Ai engineering Evaluation")


def main() -> int:
    if not RUNS_FILE.exists():
        print(f"Lipseste {RUNS_FILE}.")
        return 2

    qtext = {q["id"]: q["text"] for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8")).get("graph", {})
    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", cfg.get("neo4j_uri", "")).strip(),
        user=os.getenv("NEO4J_USER", cfg.get("neo4j_user", "")).strip(),
        password=os.getenv("NEO4J_PASSWORD", cfg.get("neo4j_password", "")).strip(),
    )
    if not client.enabled:
        print("EROARE: Neo4j indisponibil.")
        return 2
    gq = GraphQueryService(client, max_hops=int(cfg.get("max_hops", 2)))

    rows = [json.loads(l) for l in RUNS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    patched = 0
    for r in rows:
        if r.get("configuration") != "D" or r.get("category") not in ("C", "G"):
            continue
        question = qtext.get(r["question_id"], "")
        try:
            paths = gq.retrieve_paths(question=question, active_collection=NOTEBOOK, max_paths=6)
        except Exception as exc:
            print(f"  {r['question_id']}: EROARE graf {exc}")
            continue
        r["graph_evidence"] = [
            {"concept": p.get("concept", ""), "content": (p.get("content", "") or "").replace("\n", " ")[:300]}
            for p in paths
        ]
        r["graph_hits"] = len(paths)
        with_text = sum(1 for g in r["graph_evidence"] if g["content"])
        print(f"  {r['question_id']}: {len(paths)} evidente graf ({with_text} cu text)")
        patched += 1

    client.close()
    with RUNS_FILE.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nCorectat graph_evidence pentru {patched} rânduri (C/G, config D).")
    print("Acum: python scripts/eval/review.py --graph  -> re-judeci graph_contribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
