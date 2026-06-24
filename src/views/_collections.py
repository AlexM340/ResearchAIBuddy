"""Helpers pentru colectiile (notebook-urile) afisate in pagina Notebooks.

Postgres este sursa de adevar pentru ce se afiseaza (colectii, documente, flag
`indexed`, marime, tip). Biblioteca locala `DocumentManager` ramane doar zona de
staging pentru fisiere la upload/indexare (invizibila in UI) si serveste ca
fallback de citire DOAR cand Postgres nu e disponibil.
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
    file_type = row.get("file_type") or PurePath(name).suffix.lower()
    return {
        "id": row.get("id"),  # id real din Postgres (int) — folosit la stergere
        "original_name": name or "fara nume",
        "source_path": row.get("source_path", ""),
        "file_hash": row.get("file_hash", ""),
        "file_type": file_type,
        "file_size": int(row.get("file_size", 0) or 0),
        "indexed": bool(row.get("indexed", False)),
        "added_date": row.get("created_at", "") or "",
        "_origin": "postgres",
    }


def get_notebook_collections(document_manager) -> Dict[str, Dict[str, Any]]:
    """Colectiile din Postgres (sursa de adevar). Fallback la local doar daca DB e jos."""
    repository = _get_repository()
    if repository is not None:
        merged: Dict[str, Dict[str, Any]] = {}
        for col in repository.list_collections():
            name = (col.get("name") or "").strip()
            if not name:
                continue
            merged[name] = {
                "name": name,
                "document_count": col.get("document_count", 0),
                "indexed_count": col.get("indexed_count", 0),
            }
        return merged

    # Fallback: Postgres indisponibil -> biblioteca locala.
    if document_manager:
        return dict(document_manager.get_collections() or {})
    return {}


def get_notebook_documents(collection_name: str, document_manager) -> List[Dict[str, Any]]:
    """Documentele unei colectii din Postgres. Fallback la local doar daca DB e jos."""
    repository = _get_repository()
    if repository is not None:
        return [_adapt_db_document(row) for row in repository.list_documents_by_collection(collection_name)]

    if document_manager:
        return list(document_manager.get_documents_by_collection(collection_name) or [])
    return []


def create_notebook(name: str, document_manager) -> bool:
    """Creeaza un notebook in Postgres (sursa de adevar) + local (staging)."""
    created = False
    repository = _get_repository()
    if repository is not None:
        collection_id = repository.upsert_collection(name, collection_type="topic")
        created = collection_id is not None

    if document_manager:
        # Asigura colectia local ca tinta de staging pentru upload-uri viitoare.
        local_created = bool(document_manager.create_collection(name))
        created = created or local_created
    return created


def delete_notebook(name: str, document_manager) -> bool:
    """Sterge notebook-ul din Postgres (+ documente/chunks prin CASCADE) si local."""
    deleted = False
    repository = _get_repository()
    if repository is not None:
        deleted = repository.delete_collection_by_name(name)

    if document_manager:
        local_deleted = bool(document_manager.delete_collection(name))
        deleted = deleted or local_deleted
    return deleted


def delete_document(doc: Dict[str, Any], document_manager) -> bool:
    """Sterge un document din Postgres dupa id (+ chunks prin CASCADE) si local dupa hash."""
    deleted = False
    doc_id = doc.get("id")
    file_hash = doc.get("file_hash")

    repository = _get_repository()
    if repository is not None and isinstance(doc_id, int):
        deleted = repository.delete_document(doc_id)

    # Curata si copia locala de staging (daca exista), identificata prin hash.
    if document_manager and file_hash:
        for local_id, info in (document_manager.get_all_documents() or {}).items():
            if info.get("file_hash") == file_hash:
                document_manager.remove_document(local_id)
                deleted = True
                break
    return deleted
