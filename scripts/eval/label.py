"""Pas 1 evaluare: ground truth prin pooling + cautare/adaugare manuala.

Moduri:
  (default)            construieste bazinul de candidati (top-N vector) -> ground_truth.json
  --top N              bazin mai mare (util pt intrebarile cu 0 relevante)
  --stats              cate relevante ai marcat per intrebare
  --search "termeni"   cauta chunk-uri in notebook dupa cuvinte cheie (full-text)
  --add QID CID [CID]  marcheaza chunk-urile ca relevante (relevant=1) la o intrebare

Tu citesti textul si decizi relevanta. report.py foloseste doar relevant=1.

Utilizare (din radacina proiectului):
    python scripts/eval/label.py
    python scripts/eval/label.py --top 60
    python scripts/eval/label.py --stats
    python scripts/eval/label.py --search "knowledge distillation multilingual"
    python scripts/eval/label.py --add F04 1234 1237
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from storage import PostgresClient  # noqa: E402  (lightweight, fara torch)

QUESTIONS = DATA_DIR / "questions.json"
GT_FILE = DATA_DIR / "ground_truth.json"
RETRIEVAL_CATS = {"F", "X", "C", "G"}
NOTEBOOK = os.getenv("EVAL_NOTEBOOK", "Ai engineering Evaluation")


def _dsn() -> str:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    if not dsn:
        cfg = ROOT / "config.json"
        if cfg.exists():
            dsn = (json.loads(cfg.read_text(encoding="utf-8")).get("storage", {}).get("postgres_dsn") or "").strip()
    return dsn


def _pg() -> PostgresClient:
    client = PostgresClient(dsn=_dsn(), embedding_dim=384)
    if not client.enabled or not client.test_connection():
        print("EROARE: Postgres indisponibil.")
        sys.exit(2)
    return client


def _load_questions():
    return [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["category"] in RETRIEVAL_CATS]


def _load_gt():
    return json.loads(GT_FILE.read_text(encoding="utf-8")) if GT_FILE.exists() else {}


def cmd_stats() -> int:
    if not GT_FILE.exists():
        print("ground_truth.json nu exista inca. Ruleaza intai pooling-ul (fara argumente).")
        return 0
    gt = _load_gt()
    total = 0
    for q in _load_questions():
        cands = gt.get(q["id"], {}).get("candidates", [])
        rel = [c for c in cands if c.get("relevant")]
        total += len(rel)
        flag = "  <-- 0 relevante!" if not rel else ""
        print(f"  {q['id']}: {len(rel)} relevante / {len(cands)} candidati{flag}")
    print(f"\nTotal fragmente relevante marcate: {total}")
    return 0


def cmd_search(terms: str, top: int) -> int:
    # OR pe cuvinte: returneaza chunk-urile care contin ORICARE termen, sortate
    # dupa relevanta (cele cu mai multe potriviri sus). plainto_tsquery cere TOTI
    # termenii (AND) -> deseori 0 rezultate la cautari multi-cuvant.
    words = re.findall(r"[a-zA-Z0-9]+", terms.lower())
    if not words:
        print("Niciun termen valid de cautat.")
        return 0
    tsquery = " | ".join(words)
    client = _pg()
    with client.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ch.id, d.original_name, ch.text,
                       ts_rank(to_tsvector('simple', ch.text), to_tsquery('simple', %s)) AS rank
                FROM chunks ch
                JOIN documents d ON d.id = ch.document_id
                JOIN collections col ON col.id = d.collection_id
                WHERE col.name = %s
                  AND to_tsvector('simple', ch.text) @@ to_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s;
                """,
                (tsquery, NOTEBOOK, tsquery, int(top)),
            )
            rows = cur.fetchall() or []
        conn.commit()
    print(f"{len(rows)} chunk-uri pentru '{terms}' (OR) in notebook '{NOTEBOOK}':\n")
    for cid, doc, text, rank in rows:
        print(f"[chunk_id={cid}] {doc}  (rank={rank:.4f})")
        print(f"   {(text or '').strip()[:300]}\n")
    print("Marcheaza relevantele: python scripts/eval/label.py --add <QID> <chunk_id> ...")
    return 0


def cmd_add(qid: str, cids) -> int:
    gt = _load_gt()
    if qid not in gt:
        print(f"{qid} nu e in ground_truth.json. Ruleaza intai pooling-ul (fara argumente).")
        return 2
    client = _pg()
    existing = {str(c["chunk_id"]): c for c in gt[qid]["candidates"]}
    with client.connection() as conn:
        with conn.cursor() as cur:
            for cid in cids:
                cur.execute(
                    "SELECT ch.text, d.original_name FROM chunks ch JOIN documents d ON d.id = ch.document_id WHERE ch.id = %s;",
                    (int(cid),),
                )
                row = cur.fetchone()
                if not row:
                    print(f"  chunk_id={cid}: inexistent")
                    continue
                if str(cid) in existing:
                    existing[str(cid)]["relevant"] = 1
                    print(f"  chunk_id={cid}: marcat relevant=1 (era deja candidat)")
                else:
                    gt[qid]["candidates"].append(
                        {
                            "chunk_id": int(cid),
                            "relevant": 1,
                            "document": row[1] or "",
                            "score": None,
                            "text": (row[0] or "").replace("\n", " ")[:500],
                        }
                    )
                    print(f"  chunk_id={cid}: adaugat relevant=1 ({row[1]})")
        conn.commit()
    GT_FILE.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Salvat in ground_truth.json.")
    return 0


def cmd_remap() -> int:
    """Remapeaza chunk_id-urile din ground_truth.json pe ID-urile noi (dupa text).

    Necesar dupa o re-ingestie (chunk-urile primesc id-uri noi). Pastreaza
    marcajele 'relevant' potrivind textul candidatului cu noile chunk-uri.
    """
    gt = _load_gt()
    if not gt:
        print("ground_truth.json gol/inexistent.")
        return 2
    client = _pg()

    def norm(s: str) -> str:
        return " ".join((s or "").split())[:120]

    with client.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ch.id, ch.text FROM chunks ch
                JOIN documents d ON d.id = ch.document_id
                JOIN collections col ON col.id = d.collection_id
                WHERE col.name = %s;
                """,
                (NOTEBOOK,),
            )
            rows = cur.fetchall() or []
        conn.commit()

    new_by_key = {}
    for cid, text in rows:
        new_by_key.setdefault(norm(text), cid)

    remapped = same = lost = 0
    lost_items = []
    for qid, ent in gt.items():
        for cand in ent.get("candidates", []):
            nid = new_by_key.get(norm(cand.get("text", "")))
            if nid is not None:
                if cand.get("chunk_id") != nid:
                    cand["chunk_id"] = nid
                    remapped += 1
                else:
                    same += 1
            elif cand.get("relevant"):
                lost += 1
                lost_items.append((qid, (cand.get("text", "") or "")[:60]))

    GT_FILE.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Remapate: {remapped} | neschimbate: {same} | relevante ne-gasite: {lost}")
    for qid, snippet in lost_items:
        print(f"  {qid}: ne-remapat -> {snippet!r} (re-cauta cu --search/--add)")
    print("Verifica: python scripts/eval/label.py --stats")
    return 0


def cmd_pool(top: int, only_qids=None) -> int:
    from _bootstrap import build_system  # lazy: incarca torch doar la pooling

    only = {q.upper() for q in only_qids} if only_qids else None
    questions = [q for q in _load_questions() if (only is None or q["id"] in only)]
    existing = _load_gt()

    print("Construiesc sistemul (incarc embeddings)...")
    system = build_system()
    if not (getattr(system, "repository", None) and system.repository.enabled):
        print("EROARE: Postgres indisponibil.")
        return 2

    # Pornim de la ce exista (merge), ca sa nu pierdem intrebarile ne-reprocesate.
    gt = dict(existing)
    for q in questions:
        qid, text = q["id"], q["text"]
        emb = system._embed_note_text("", text)
        rows = (
            system.repository.vector_search(query_embedding=emb, collection_filters=[NOTEBOOK], top_k=top)
            if emb is not None
            else []
        )
        prev = {str(c["chunk_id"]): int(c.get("relevant", 0)) for c in existing.get(qid, {}).get("candidates", [])}
        candidates = []
        for r in rows:
            meta = r.get("metadata", {}) or {}
            cid = meta.get("db_chunk_id")
            if cid is None:
                continue
            candidates.append(
                {
                    "chunk_id": cid,
                    "relevant": prev.get(str(cid), 0),
                    "document": meta.get("filename", ""),
                    "score": round(float(r.get("retrieval_score", 0.0) or 0.0), 4),
                    "text": (r.get("content", "") or "").replace("\n", " ")[:500],
                }
            )
        # Pastram si chunk-urile adaugate manual care nu mai apar in noul bazin.
        for cid_str, mark in prev.items():
            if mark and cid_str not in {str(c["chunk_id"]) for c in candidates}:
                old = next((c for c in existing[qid]["candidates"] if str(c["chunk_id"]) == cid_str), None)
                if old:
                    candidates.append(old)
        gt[qid] = {"question": text, "category": q["category"], "documents_hint": q.get("documents", []), "candidates": candidates}
        print(f"  {qid}: {len(candidates)} candidati")

    GT_FILE.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nScris {GT_FILE}. Pune relevant: 1 la chunk-urile relevante, apoi --stats.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground truth: pooling + cautare/adaugare manuala")
    ap.add_argument("--top", type=int, default=30, help="Marimea bazinului / a cautarii")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--search", metavar="TERMENI", help="Cauta chunk-uri dupa cuvinte cheie")
    ap.add_argument("--add", nargs="+", metavar="ARG", help="QID urmat de unul sau mai multe chunk_id")
    ap.add_argument("--qids", nargs="+", metavar="QID", help="Re-pool doar aceste intrebari (ex: F01 X02)")
    ap.add_argument("--remap", action="store_true", help="Remapeaza chunk_id-urile pe ID-uri noi dupa o re-ingestie")
    args = ap.parse_args()

    if args.remap:
        return cmd_remap()
    if args.stats:
        return cmd_stats()
    if args.search:
        return cmd_search(args.search, args.top)
    if args.add:
        if len(args.add) < 2:
            print("Utilizare: --add <QID> <chunk_id> [chunk_id ...]")
            return 2
        return cmd_add(args.add[0], args.add[1:])
    return cmd_pool(args.top, only_qids=args.qids)


if __name__ == "__main__":
    raise SystemExit(main())
