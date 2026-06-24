"""Migreaza biblioteca locala (JSON) in Postgres, sursa partajata cu Streamlit.

Citeste ./data/document_library/library_metadata.json (DocumentManager) si face
upsert pe colectii + documente in baza Postgres configurata, astfel incat
deploy-ul de pe Streamlit (care citeste din Postgres) sa vada aceleasi notebook-uri.

Idempotent: rularile repetate doar fac upsert (ON CONFLICT). NU rescrie nimic
distructiv.

ATENTIE: migreaza DOAR metadatele (colectii + randuri din `documents`), NU si
chunk-urile/embeddings-urile. Daca un document local nu a fost niciodata indexat
in Postgres-ul partajat, va aparea ca sursa in notebook, dar fara continut pentru
retrieval — pentru continut e nevoie de re-indexare din aplicatie.

Utilizare (din radacina proiectului):
    python migrate_library_to_postgres.py            # aplica
    python migrate_library_to_postgres.py --dry-run  # doar raporteaza
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Scriptul sta in scripts/; radacina proiectului e un nivel mai sus.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:  # incarca .env daca exista (POSTGRES_DSN poate veni de acolo)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from document_manager import DocumentManager  # noqa: E402
from storage import PostgresClient, SecondBrainRepository  # noqa: E402


def _load_dsn_and_dim() -> tuple[str, int]:
    """DSN + dimensiune embeddings din env, cu fallback la config.json."""
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
    parser = argparse.ArgumentParser(description="Migreaza biblioteca locala in Postgres")
    parser.add_argument(
        "--library",
        default="./data/document_library",
        help="Calea catre biblioteca locala DocumentManager",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Doar raporteaza ce s-ar migra; nu scrie in Postgres",
    )
    args = parser.parse_args()

    dsn, dim = _load_dsn_and_dim()
    if not dsn:
        print("EROARE: POSTGRES_DSN lipseste (env sau config.json -> storage.postgres_dsn).")
        return 2

    client = PostgresClient(dsn=dsn, embedding_dim=dim)
    if not client.enabled:
        print("EROARE: clientul Postgres nu e activ (psycopg lipsa sau DSN gol).")
        return 2

    repo = SecondBrainRepository(client)
    if not repo.enabled or not client.test_connection():
        print("EROARE: nu ma pot conecta la Postgres cu DSN-ul dat.")
        return 2

    dm = DocumentManager(library_path=args.library)
    collections = dm.get_collections() or {}
    documents = dm.get_all_documents() or {}

    safe_target = dsn.split("@")[-1] if "@" in dsn else "(host ascuns)"
    print(f"Biblioteca locala: {len(collections)} colectii, {len(documents)} documente.")
    print(f"Tinta Postgres: {safe_target}")
    if args.dry_run:
        print("\n[DRY-RUN] Nu se scrie nimic. Ce s-ar migra:\n")

    col_ok = 0
    doc_ok = 0
    doc_fail = 0

    for name in collections.keys():
        clean = (name or "").strip()
        if not clean:
            continue
        ctype = "general" if clean.lower() in {"general", "default"} else "topic"
        if args.dry_run:
            print(f"  colectie: {clean} ({ctype})")
            col_ok += 1
            continue
        if repo.upsert_collection(clean, collection_type=ctype) is not None:
            col_ok += 1

    for doc_id, info in documents.items():
        file_hash = info.get("file_hash")
        name = info.get("original_name") or doc_id
        collection = info.get("collection", "general")
        # Biblioteca locala nu pastreaza source_path absolut; folosim library_path.
        source_path = info.get("source_path") or info.get("library_path") or name
        indexed = bool(info.get("indexed", False))
        file_size = int(info.get("file_size", 0) or 0)
        file_type = info.get("file_type", "") or ""

        if not file_hash:
            print(f"  SKIP (fara file_hash): {name}")
            doc_fail += 1
            continue
        if args.dry_run:
            print(f"  document: {name} -> {collection} (indexed={indexed}, {file_size} bytes)")
            doc_ok += 1
            continue

        new_id = repo.upsert_document(
            file_hash=file_hash,
            original_name=name,
            collection_name=collection,
            source_path=source_path,
            indexed=indexed,
            file_size=file_size,
            file_type=file_type,
        )
        if new_id is not None:
            doc_ok += 1
        else:
            doc_fail += 1

    print("\nRezumat:")
    print(f"  colectii migrate: {col_ok}/{len(collections)}")
    print(f"  documente migrate: {doc_ok} (esuate: {doc_fail})")
    if args.dry_run:
        print("\nReruleaza FARA --dry-run ca sa aplici efectiv.")
    else:
        print("\nGata. Verifica pagina Notebooks pe Streamlit (eventual reboot la app).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
