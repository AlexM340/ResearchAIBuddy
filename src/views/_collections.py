"""Helpers pentru colectiile (notebook-urile) afisate in pagina Notebooks.

Istoric, lista de notebook-uri venea DOAR din biblioteca locala JSON
(`DocumentManager`), care e per-masina si nu e partajata cu deploy-ul Streamlit.
Aceste functii prefera Postgres (sursa de adevar partajata) si fac merge cu
biblioteca locala, ca sa nu se piarda nimic si ca acelasi notebook sa fie vizibil
si local, si pe Streamlit, atat timp cat ambele instante folosesc aceeasi baza.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Dict, List

import streamlit as st


def _get_repository():
    """Returneaza repository-ul Postgres activ din sesiune, sau None."""
    apci_system = st.session_state.get("apci_system")
    repository = getattr(apci_system, "repository", None) if apci_system else None
    if repository is not None and getattr(repository, "enabled", False):
        return repository
    return None


def _adapt_db_document(row: Dict[str, Any]) -> Dict[str, Any]:
    """Mapeaza un rand din `documents` (Postgres) la forma asteptata de UI."""
    name = row.get("original_name", "") or row.get("source_path", "")
    suffix = PurePath(name).suffix.lower()
    return {
        # Prefix 'pg_' marcheaza un document care exista doar in Postgres (nu si
        # in biblioteca locala) — stergerea din UI il ignora elegant (vezi nota).
        "id": f"pg_{row.get('id')}",
        "original_name": row.get("original_name", "") or "fara nume",
        "source_path": row.get("source_path", ""),
        "file_hash": row.get("file_hash", ""),
        "file_type": suffix,
        "file_size": 0,
        "indexed": bool(row.get("indexed", False)),
        "added_date": row.get("created_at", "") or "",
        "collection": "",
        "_origin": "postgres",
    }


def get_notebook_collections(document_manager) -> Dict[str, Dict[str, Any]]:
    """Merge intre colectiile din Postgres (partajat) si biblioteca locala."""
    merged: Dict[str, Dict[str, Any]] = {}
    if document_manager:
        for name, info in (document_manager.get_collections() or {}).items():
            merged[name] = dict(info)

    repository = _get_repository()
    if repository is not None:
        for col in repository.list_collections():
            name = (col.get("name") or "").strip()
            if not name:
                continue
            entry = merged.setdefault(name, {"name": name, "documents": []})
            entry["name"] = name
            entry["document_count"] = col.get("document_count", entry.get("document_count"))
            entry["indexed_count"] = col.get("indexed_count")
    return merged


def get_notebook_documents(collection_name: str, document_manager) -> List[Dict[str, Any]]:
    """Merge intre documentele unei colectii din Postgres (partajat) + local.

    Deduplicarea se face pe `file_hash`; randurile locale au prioritate fiindca
    pastreaza metadate mai bogate (file_size, file_type, id local pentru stergere).
    """
    docs: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    if document_manager:
        for doc in document_manager.get_documents_by_collection(collection_name) or []:
            docs.append(doc)
            file_hash = doc.get("file_hash")
            if file_hash:
                seen_hashes.add(file_hash)

    repository = _get_repository()
    if repository is not None:
        for row in repository.list_documents_by_collection(collection_name):
            file_hash = row.get("file_hash")
            if file_hash and file_hash in seen_hashes:
                continue
            docs.append(_adapt_db_document(row))
            if file_hash:
                seen_hashes.add(file_hash)
    return docs


def create_notebook(name: str, document_manager) -> bool:
    """Creeaza un notebook in biblioteca locala SI in Postgres (partajat).

    Idempotent: daca exista deja intr-una din surse, il asigura in cealalta si
    raporteaza succes, ca sa nu apara fals 'Nu am putut crea notebook-ul'.
    """
    created = False
    if document_manager:
        # create_collection -> False daca exista deja local; acceptabil.
        created = bool(document_manager.create_collection(name))

    repository = _get_repository()
    if repository is not None:
        collection_id = repository.upsert_collection(name, collection_type="topic")
        created = created or (collection_id is not None)
    return created
