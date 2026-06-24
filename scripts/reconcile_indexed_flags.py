"""Reconciliaza flagul documents.indexed cu realitatea (prezenta chunk-urilor).

Dupa migrarea bibliotecii locale, flagul `indexed` poate sa nu corespunda
realitatii: documente marcate `indexed=true` fara niciun chunk (necautabile, dar
sarite la reindexare) sau `indexed=false` desi au chunks. Acest script seteaza:

    indexed = TRUE  daca documentul are >= 1 chunk
    indexed = FALSE daca nu are niciun chunk

Atinge DOAR coloana boolean `documents.indexed` — nu modifica chunks/embeddings.
La final listeaza documentele care raman de re-indexat (0 chunks).

Utilizare (din radacina proiectului):
    python reconcile_indexed_flags.py --dry-run   # doar raporteaza
    python reconcile_indexed_flags.py             # aplica
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Scriptul sta in scripts/; radacina proiectului e un nivel mai sus.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from storage import PostgresClient  # noqa: E402


def _load_dsn_and_dim() -> tuple[str, int]:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    dim = 384
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            storage = cfg.get("storage", {})
            if not dsn:
                dsn = (storage.get("postgres_dsn") or "").strip()
            dim = int(storage.get("pgvector_embedding_dim", 384) or 384)
        except Exception as exc:
            print(f"AVERTISMENT: nu am putut citi config.json: {exc}")
    return dsn, dim


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconciliaza documents.indexed cu chunks reale")
    parser.add_argument("--dry-run", action="store_true", help="Doar raporteaza; nu scrie")
    args = parser.parse_args()

    dsn, dim = _load_dsn_and_dim()
    if not dsn:
        print("EROARE: POSTGRES_DSN lipseste (env sau config.json -> storage.postgres_dsn).")
        return 2

    client = PostgresClient(dsn=dsn, embedding_dim=dim)
    if not client.enabled or not client.test_connection():
        print("EROARE: nu ma pot conecta la Postgres.")
        return 2

    with client.connection() as conn:
        cur = conn.cursor()

        # Documente al caror flag NU corespunde realitatii (prezenta chunk-urilor).
        cur.execute(
            """
            SELECT col.name, d.original_name, d.indexed, COUNT(ch.id) AS chunks
            FROM documents d
            JOIN collections col ON col.id = d.collection_id
            LEFT JOIN chunks ch ON ch.document_id = d.id
            GROUP BY col.name, d.id, d.original_name, d.indexed
            HAVING d.indexed <> (COUNT(ch.id) > 0)
            ORDER BY col.name, d.original_name;
            """
        )
        mismatches = cur.fetchall()

        print(f"Documente cu flag gresit: {len(mismatches)}")
        for name_col, doc_name, indexed, chunks in mismatches:
            corect = "TRUE" if chunks > 0 else "FALSE"
            print(f"  [{name_col}] {doc_name}: indexed={indexed} chunks={chunks} -> {corect}")

        if not args.dry_run and mismatches:
            # Reconciliere in masa: indexed reflecta prezenta chunk-urilor.
            cur.execute(
                """
                UPDATE documents d
                SET indexed = sub.has_chunks, updated_at = NOW()
                FROM (
                    SELECT d2.id,
                           (COUNT(ch.id) > 0) AS has_chunks
                    FROM documents d2
                    LEFT JOIN chunks ch ON ch.document_id = d2.id
                    GROUP BY d2.id
                ) AS sub
                WHERE d.id = sub.id
                  AND d.indexed <> sub.has_chunks;
                """
            )
            updated = cur.rowcount
            conn.commit()
            print(f"\nActualizat: {updated} documente.")
        elif args.dry_run:
            print("\n[DRY-RUN] Nu s-a scris nimic.")
        else:
            print("\nNimic de actualizat — flagurile sunt deja corecte.")

        # Lista finala: ce ramane de re-indexat (0 chunks).
        cur.execute(
            """
            SELECT col.name, d.original_name
            FROM documents d
            JOIN collections col ON col.id = d.collection_id
            LEFT JOIN chunks ch ON ch.document_id = d.id
            GROUP BY col.name, d.id, d.original_name
            HAVING COUNT(ch.id) = 0
            ORDER BY col.name, d.original_name;
            """
        )
        empty = cur.fetchall()
        print(f"\n=== De re-indexat (0 chunks): {len(empty)} documente ===")
        cur_col = None
        for name_col, doc_name in empty:
            if name_col != cur_col:
                cur_col = name_col
                print(f"\n[{cur_col}]")
            print(f"  - {doc_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
