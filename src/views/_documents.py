"""Operatii pe documente partajate intre vederi si sidebar.

Toate functiile de aici asuma ca `st.session_state.apci_system` este deja
initializat (bootstrap se face in `main_flash.main()` inainte de render).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from views._shared import build_metadata_entry, normalize_collection_name

logger = logging.getLogger(__name__)


def update_system_status() -> None:
    """Actualizeaza statusul sistemului in session_state."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return
    try:
        status = apci_system.get_system_status()
        st.session_state.system_stats = status
        storage_status = status.get("storage_index_status", {})
        if storage_status:
            prev_summary = st.session_state.get("last_sync_summary", {}) or {}
            st.session_state.last_sync_summary = {
                "indexed_documents": storage_status.get("indexed_documents", 0),
                "indexed_chunks": storage_status.get("indexed_chunks", 0),
                "last_sync_at": storage_status.get("last_sync_at"),
                "storage_mode": storage_status.get("storage_mode", "local_fallback"),
                "bootstrap": prev_summary.get("bootstrap", False),
            }
    except Exception as exc:
        logger.error("Eroare la update_system_status: %s", exc)


def collect_library_payload(doc_ids: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Construieste payload-ul pentru documentele din biblioteca."""
    document_manager = st.session_state.get("document_manager")
    file_paths: List[str] = []
    metadata_by_path: Dict[str, Dict[str, Any]] = {}
    path_to_doc_id: Dict[str, str] = {}
    if not document_manager:
        return file_paths, metadata_by_path, path_to_doc_id

    unique_paths: set[str] = set()
    for doc_id in doc_ids:
        file_path = document_manager.get_document_file_path(doc_id)
        doc_info = document_manager.get_document_info(doc_id)
        if not file_path or not file_path.exists() or not doc_info:
            continue
        resolved_path = str(file_path.resolve())
        if resolved_path in unique_paths:
            continue
        unique_paths.add(resolved_path)
        file_paths.append(resolved_path)
        path_to_doc_id[resolved_path] = doc_id
        metadata_by_path[resolved_path] = build_metadata_entry(doc_id, doc_info, file_path)

    return file_paths, metadata_by_path, path_to_doc_id


def run_full_reindex() -> Dict[str, Any]:
    """Reindexeaza toate documentele din biblioteca cu modelul curent de embedding.

    Folosit dupa schimbarea modelului de embedding (ex: switch la multilingv).
    """
    result: Dict[str, Any] = {"success": False, "message": "", "summary": {}}

    document_manager = st.session_state.get("document_manager")
    apci_system = st.session_state.get("apci_system")

    if not document_manager:
        result["message"] = "Biblioteca locala nu este disponibila."
        return result
    if not apci_system:
        result["message"] = "Sistemul CerebrumAI nu este initializat."
        return result

    all_docs = document_manager.get_all_documents()
    doc_ids = list(all_docs.keys())
    file_paths, metadata_by_path, _ = collect_library_payload(doc_ids)

    if not file_paths:
        result["message"] = "Nu exista documente de reindexat."
        return result

    try:
        summary = apci_system.reindex_all_documents(
            file_paths,
            metadata_by_path=metadata_by_path,
        )
        result["summary"] = summary
        result["success"] = bool(summary.get("success"))
        if result["success"]:
            result["message"] = (
                f"Reindexat {summary.get('documents_reindexed', 0)} documente "
                f"(sterse {summary.get('chunks_deleted', 0)} chunks vechi) "
                f"in {summary.get('duration_s', 0)}s."
            )
        else:
            result["message"] = summary.get("error") or "Reindexare esuata."
    except Exception as exc:
        logger.error("run_full_reindex failed: %s", exc)
        result["message"] = f"Eroare la reindexare: {exc}"

    update_system_status()
    return result


def add_web_url_as_document(url: str, title: str, target_collection: str) -> Dict[str, Any]:
    """Fetch a URL, strip HTML, save as .txt and index it in the given collection."""
    result: Dict[str, Any] = {"success": False, "message": ""}

    apci_system = st.session_state.get("apci_system")
    document_manager = st.session_state.get("document_manager")
    if not apci_system:
        result["message"] = "Sistemul CerebrumAI nu este initializat."
        return result
    if not document_manager:
        result["message"] = "Biblioteca de documente nu este disponibila."
        return result

    try:
        headers = {"User-Agent": "Mozilla/5.0 CerebrumAI/1.0 (research assistant)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        raw_html = resp.text
        # strip scripts + styles then tags
        clean = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<style[^>]*>.*?</style>", " ", clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        if len(clean) < 150:
            result["message"] = "Continutul paginii este prea scurt sau nu a putut fi extras."
            return result

        safe_title = re.sub(r"[^\w\-]", "_", title)[:50]
        filename = f"web_{safe_title}.txt"

        temp_dir = Path("./data/uploaded_docs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / filename
        file_path.write_text(f"Source: {url}\nTitle: {title}\n\n{clean}", encoding="utf-8")

        target_collection = normalize_collection_name(target_collection)
        doc_id = document_manager.add_document(
            str(file_path),
            collection=target_collection,
            description=f"Web: {url}",
        )
        if not doc_id:
            result["message"] = "Nu am putut inregistra documentul in biblioteca."
            return result

        doc_info = document_manager.get_document_info(doc_id)
        library_path = document_manager.get_document_file_path(doc_id)
        if not (doc_info and library_path):
            result["message"] = "Eroare interna la inregistrarea documentului."
            return result

        resolved_path = str(library_path.resolve())
        metadata = build_metadata_entry(doc_id, doc_info, library_path)
        success = apci_system.load_documents([resolved_path], metadata_by_path={resolved_path: metadata})
        if success:
            document_manager.mark_document_indexed(doc_id)
            st.session_state.documents_loaded = True
            update_system_status()
            result["success"] = True
            result["message"] = f"Sursa '{title}' adaugata si indexata."
        else:
            result["message"] = "Sursa adaugata in biblioteca, dar indexarea a esuat."
        return result
    except requests.RequestException as exc:
        logger.error("add_web_url_as_document fetch failed: %s", exc)
        result["message"] = f"Nu am putut accesa URL-ul: {exc}"
        return result
    except Exception as exc:
        logger.error("add_web_url_as_document failed: %s", exc)
        result["message"] = f"Eroare: {exc}"
        return result


def save_text_as_document(
    content: str,
    title: str,
    target_collection: str,
    description: str = "",
    file_extension: str = "md",
    filename_prefix: str = "synth",
) -> Dict[str, Any]:
    """Persist raw text (sinteza, outline, notita lunga) ca document indexat.

    Acelasi pipeline cu add_web_url_as_document si upload manual:
    save → register in document_manager → load_documents (chunking + embedding).
    """
    result: Dict[str, Any] = {"success": False, "message": "", "doc_id": None}

    apci_system = st.session_state.get("apci_system")
    document_manager = st.session_state.get("document_manager")
    if not apci_system:
        result["message"] = "Sistemul CerebrumAI nu este initializat."
        return result
    if not document_manager:
        result["message"] = "Biblioteca de documente nu este disponibila."
        return result
    if not content or not content.strip():
        result["message"] = "Continut gol."
        return result

    try:
        timestamp = time.strftime("%Y%m%d_%H%M")
        safe_title = re.sub(r"[^\w\-]", "_", title)[:50] or "untitled"
        filename = f"{filename_prefix}_{safe_title}_{timestamp}.{file_extension}"

        temp_dir = Path("./data/uploaded_docs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / filename
        file_path.write_text(content, encoding="utf-8")

        target_collection = normalize_collection_name(target_collection)
        doc_id = document_manager.add_document(
            str(file_path),
            collection=target_collection,
            description=description or f"Auto-generated: {title}",
        )
        if not doc_id:
            result["message"] = "Nu am putut inregistra documentul."
            return result

        doc_info = document_manager.get_document_info(doc_id)
        library_path = document_manager.get_document_file_path(doc_id)
        if not (doc_info and library_path):
            result["message"] = "Inregistrare incompleta."
            return result

        resolved_path = str(library_path.resolve())
        metadata = build_metadata_entry(doc_id, doc_info, library_path)
        success = apci_system.load_documents([resolved_path], metadata_by_path={resolved_path: metadata})
        if success:
            document_manager.mark_document_indexed(doc_id)
            st.session_state.documents_loaded = True
            update_system_status()
            result["success"] = True
            result["doc_id"] = doc_id
            result["message"] = f"Document '{title}' salvat si indexat."
        else:
            result["message"] = "Salvat in biblioteca, dar indexarea a esuat."
        return result
    except Exception as exc:
        logger.error("save_text_as_document failed: %s", exc)
        result["message"] = f"Eroare: {exc}"
        return result


def sync_library_to_db(doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sync biblioteca -> index intern (DB primary, delta-only)."""
    result: Dict[str, Any] = {
        "success": False,
        "ingested": 0,
        "skipped": 0,
        "total_candidates": 0,
        "message": "",
    }

    document_manager = st.session_state.get("document_manager")
    apci_system = st.session_state.get("apci_system")

    if not document_manager:
        result["message"] = "Biblioteca locala nu este disponibila."
        return result
    if not apci_system:
        result["message"] = "Sistemul CerebrumAI nu este initializat."
        return result

    try:
        if doc_ids is None:
            all_docs = document_manager.get_all_documents()
            doc_ids = list(all_docs.keys())

        file_paths, metadata_by_path, path_to_doc_id = collect_library_payload(doc_ids)
        result["total_candidates"] = len(file_paths)
        if not file_paths:
            result["message"] = "Nu exista documente valide pentru sync."
            return result

        filtered = apci_system.filter_paths_for_ingestion(
            file_paths,
            metadata_by_path=metadata_by_path,
        )
        to_ingest = filtered.get("to_ingest", [])
        already_indexed = filtered.get("already_indexed", [])
        result["ingested"] = len(to_ingest)
        result["skipped"] = len(already_indexed)

        for indexed_path in already_indexed:
            doc_id = path_to_doc_id.get(indexed_path)
            if doc_id:
                document_manager.mark_document_indexed(doc_id)

        if to_ingest:
            ingest_metadata = {
                path: metadata_by_path[path]
                for path in to_ingest
                if path in metadata_by_path
            }
            success = apci_system.load_documents(to_ingest, metadata_by_path=ingest_metadata)
            if not success:
                result["message"] = "Eroare la ingestia documentelor in index."
                return result
            for ingested_path in to_ingest:
                doc_id = path_to_doc_id.get(ingested_path)
                if doc_id:
                    document_manager.mark_document_indexed(doc_id)

        storage_status = apci_system.get_storage_index_status()
        st.session_state.last_sync_summary = {
            "indexed_documents": storage_status.get("indexed_documents", 0),
            "indexed_chunks": storage_status.get("indexed_chunks", 0),
            "last_sync_at": storage_status.get("last_sync_at"),
            "storage_mode": storage_status.get("storage_mode", "local_fallback"),
            "db_primary_strict": bool(storage_status.get("db_primary_strict", False)),
            "bootstrap": False,
        }
        st.session_state.documents_loaded = bool(
            int(storage_status.get("indexed_chunks", 0) or 0) > 0
            or already_indexed
            or to_ingest
        )
        st.session_state.documents_bootstrap_source = "manual_sync"
        update_system_status()

        result["success"] = True
        if to_ingest:
            result["message"] = f"Sync finalizat: {len(to_ingest)} ingestate, {len(already_indexed)} deja indexate."
        else:
            result["message"] = f"Nimic de ingestat. {len(already_indexed)} documente erau deja indexate."
        return result
    except Exception as exc:
        logger.error("Eroare la sync biblioteca -> DB: %s", exc)
        result["message"] = f"Eroare la sync: {exc}"
        return result


def process_uploaded_files(
    uploaded_files: List[Any],
    target_collection: str,
) -> Dict[str, Any]:
    """Salveaza, inregistreaza in biblioteca si indexeaza fisierele uploadate.

    Returneaza un dict cu chei: success, message, ingested, skipped.
    """
    result: Dict[str, Any] = {"success": False, "ingested": 0, "skipped": 0, "message": ""}

    apci_system = st.session_state.get("apci_system")
    document_manager = st.session_state.get("document_manager")

    if not apci_system:
        result["message"] = "Sistemul CerebrumAI nu este initializat."
        return result
    if not document_manager:
        result["message"] = "Biblioteca de documente nu este disponibila."
        return result
    if not uploaded_files:
        result["message"] = "Nu ai selectat fisiere."
        return result

    target_collection = normalize_collection_name(target_collection)
    temp_dir = Path("./data/uploaded_docs")
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_paths: List[str] = []
    path_to_doc_id: Dict[str, str] = {}
    metadata_by_path: Dict[str, Dict[str, Any]] = {}
    unique_paths: set[str] = set()

    try:
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as fh:
                fh.write(uploaded_file.getbuffer())

            doc_id = document_manager.add_document(
                str(file_path),
                collection=target_collection,
                description=f"Incarcat la {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            if not doc_id:
                continue

            doc_info = document_manager.get_document_info(doc_id)
            library_path = document_manager.get_document_file_path(doc_id)
            if not (doc_info and library_path):
                continue
            resolved_path = str(library_path.resolve())
            if resolved_path in unique_paths:
                continue
            unique_paths.add(resolved_path)
            file_paths.append(resolved_path)
            path_to_doc_id[resolved_path] = doc_id
            metadata_by_path[resolved_path] = build_metadata_entry(doc_id, doc_info, library_path)

        if not file_paths:
            result["message"] = "Nu s-au putut inregistra fisierele."
            return result

        filtered = apci_system.filter_paths_for_ingestion(
            file_paths,
            metadata_by_path=metadata_by_path,
        )
        to_ingest = filtered.get("to_ingest", [])
        already_indexed = filtered.get("already_indexed", [])

        for existing_path in already_indexed:
            doc_id = path_to_doc_id.get(existing_path)
            if doc_id:
                document_manager.mark_document_indexed(doc_id)

        if to_ingest:
            ingest_metadata = {
                path: metadata_by_path[path]
                for path in to_ingest
                if path in metadata_by_path
            }
            success = apci_system.load_documents(to_ingest, metadata_by_path=ingest_metadata)
            if not success:
                result["message"] = "Eroare la indexare."
                return result
            for ingested_path in to_ingest:
                doc_id = path_to_doc_id.get(ingested_path)
                if doc_id:
                    document_manager.mark_document_indexed(doc_id)

        st.session_state.documents_loaded = True
        st.session_state.documents_bootstrap_source = "manual_sync"
        update_system_status()

        result["success"] = True
        result["ingested"] = len(to_ingest)
        result["skipped"] = len(already_indexed)
        if to_ingest:
            result["message"] = f"Indexate: {len(to_ingest)}. Deja existau: {len(already_indexed)}."
        else:
            result["message"] = f"Toate fisierele ({len(already_indexed)}) erau deja indexate."
        return result
    except Exception as exc:
        logger.error("Eroare la procesarea fisierelor: %s", exc)
        result["message"] = f"Eroare: {exc}"
        return result
