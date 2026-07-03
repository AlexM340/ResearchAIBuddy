"""Ajutor pentru scorarea manuala: afiseaza formatat continutul unei configuratii.

Citeste results/runs.jsonl si tipareste, pentru config D (implicit), intrebarea,
raspunsul si (pentru C/G) evidentele de graf — ca sa completezi manual_scores.json
fara sa citesti JSON brut.

Utilizare (din radacina proiectului):
    python scripts/eval/review.py              # toate F/X/C/G: intrebare + raspuns
    python scripts/eval/review.py --graph      # doar C/G: + graph_evidence (pt scorul grafului)
    python scripts/eval/review.py --qid C04    # doar o intrebare
    python scripts/eval/review.py --config A    # alta configuratie
"""

from __future__ import annotations

import argparse
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


def _chunk_texts(ids):
    """Textul chunk-urilor vector dupa id (pentru evidentele D# in scorul grafului)."""
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return {}
    try:
        from storage import PostgresClient

        dsn = os.getenv("POSTGRES_DSN", "").strip() or (
            json.loads((ROOT / "config.json").read_text(encoding="utf-8")).get("storage", {}).get("postgres_dsn") or ""
        ).strip()
        client = PostgresClient(dsn=dsn, embedding_dim=384)
        out = {}
        with client.connection() as conn:
            if conn is None:
                return {}
            with conn.cursor() as cur:
                cur.execute("SELECT id, text FROM chunks WHERE id = ANY(%s);", (ids,))
                for cid, text in cur.fetchall() or []:
                    out[cid] = (text or "").replace("\n", " ")
            conn.commit()
        return out
    except Exception as exc:
        print(f"(nu am putut aduce textul vector: {exc})")
        return {}


def _question_text():
    """Mapeaza question_id -> textul intrebarii din questions.json."""
    if not QUESTIONS.exists():
        return {}
    return {q["id"]: q["text"] for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Vizualizare rezultate pentru scorare manuala")
    ap.add_argument("--config", default="D", help="Configuratia de afisat (implicit D)")
    ap.add_argument("--graph", action="store_true", help="Doar C/G, cu graph_evidence")
    ap.add_argument("--qid", help="Doar o intrebare (ex: C04)")
    args = ap.parse_args()

    if not RUNS_FILE.exists():
        print(f"Lipseste {RUNS_FILE}. Ruleaza intai scripts/eval/run.py.")
        return 2

    cfg = args.config.upper()
    qtext = _question_text()
    runs = [json.loads(l) for l in RUNS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in runs if r.get("configuration") == cfg]
    if args.graph:
        rows = [r for r in rows if r.get("category") in ("C", "G")]
    if args.qid:
        rows = [r for r in rows if r.get("question_id") == args.qid.upper()]
    rows.sort(key=lambda r: r.get("question_id", ""))

    for r in rows:
        print("=" * 90)
        print(f"[{r['question_id']}] ({r['category']}) config={cfg} | route={r.get('route_used')} | graph_hits={r.get('graph_hits')}")
        print(f"\nINTREBARE: {qtext.get(r['question_id'], '(text indisponibil)')}")
        if args.graph or r.get("category") in ("C", "G"):
            # Evidentele vector (D#) = baseline-ul fata de care se judeca graful.
            vtexts = _chunk_texts(r.get("retrieved_chunk_ids", []))
            if vtexts:
                print("\nEVIDENTE VECTOR (D#):")
                for i, cid in enumerate(r.get("retrieved_chunk_ids", []), 1):
                    t = vtexts.get(int(cid)) if str(cid).isdigit() else None
                    if t:
                        print(f"  D{i} {t[:300]}")
        print(f"\nRASPUNS:\n{r.get('answer', '').strip()}\n")
        if args.graph or r.get("category") in ("C", "G"):
            ge = r.get("graph_evidence", []) or []
            if ge:
                print("EVIDENTE GRAF (G#) — de evaluat (-1/0/1/2):")
                for i, g in enumerate(ge, 1):
                    print(f"  G{i} [{g.get('concept', '')}] {g.get('content', '')}")
                print()
    print("=" * 90)
    print(f"{len(rows)} intrebari afisate (config {cfg}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
