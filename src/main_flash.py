"""
Aplicația principală APCI (Asistentul Personalizat de Cercetare și Învățare)
Interfață web construită cu Streamlit pentru Gemini 2.5 Flash
"""

import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Încarcă variabilele de mediu din .env
try:
    from dotenv import load_dotenv
    # Caută .env în directorul părinte (root al proiectului)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv nu este disponibil

# Configurare logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurare pagină
st.set_page_config(
    page_title="APCI - Asistent Personalizat de Cercetare și Învățare",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import module locale
try:
    from rag_module_flash import create_apci_system, RAGConfig
    RAG_MODULE_AVAILABLE = True
except ImportError as e:
    RAG_MODULE_AVAILABLE = False
    st.error(f"Modulul RAG nu este disponibil: {e}")

try:
    from document_manager import DocumentManager
    DOCUMENT_MANAGER_AVAILABLE = True
except ImportError as e:
    DOCUMENT_MANAGER_AVAILABLE = False
    st.error(f"Document Manager nu este disponibil: {e}")

try:
    from chat_session_manager import ChatSessionManager
    CHAT_SESSION_MANAGER_AVAILABLE = True
except ImportError as e:
    CHAT_SESSION_MANAGER_AVAILABLE = False
    st.error(f"Chat Session Manager nu este disponibil: {e}")

GENERAL_COLLECTION_NAME = "general"
QUERY_MODE_LABELS = {
    "topic": "Doar Topic",
    "topic_general": "Topic + General",
    "all": "Toate Sursele"
}
DEFAULT_CHAT_TITLE = "Chat nou"

def normalize_collection_name(name: str) -> str:
    """Normalizează numele colecției și aplică fallback pentru general."""
    cleaned = (name or "").strip()
    return cleaned if cleaned else GENERAL_COLLECTION_NAME

def is_general_collection(name: str) -> bool:
    """Verifică dacă o colecție este colecție generală."""
    normalized = normalize_collection_name(name).lower()
    return normalized in {GENERAL_COLLECTION_NAME, "default"}

def build_metadata_entry(doc_id: str, doc_info: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    """Construiește metadata standardizată pentru indexare și filtrare la query."""
    collection = normalize_collection_name(doc_info.get("collection", GENERAL_COLLECTION_NAME))
    source_scope = "general" if is_general_collection(collection) else "topic"
    resolved_path = str(file_path.resolve())

    return {
        "doc_id": doc_id,
        "filename": doc_info.get("original_name", file_path.name),
        "original_name": doc_info.get("original_name", file_path.name),
        "file_hash": doc_info.get("file_hash", ""),
        "file_size": doc_info.get("file_size", 0),
        "file_type": doc_info.get("file_type", file_path.suffix),
        "collection": collection,
        "source_scope": source_scope,
        "source_path": resolved_path
    }

def load_config() -> Dict[str, Any]:
    """Încarcă configurația din fișier cu fallback la valori implicite"""
    config_path = Path("config.json")
    
    # Configurație implicită
    default_config = {
        "models": {
            "primary_llm": "gemini-2.5-flash",
            "fallback_llm": "gemini-2.0-flash-exp"
        },
        "gemini": {
            "temperature": 0.1,
            "max_tokens": 65536,  # Increased to 128K for better performance
            "rate_limits": {
                "requests_per_minute": 30,
                "requests_per_day": 3000
            }
        },
        "chunking": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_context_docs": 5
        },
        "rag": {
            "retrieval_candidates": 20,
            "rerank_top_k": 8,
            "neighbor_window": 1,
            "hybrid_alpha": 0.7
        },
        "storage": {
            "postgres_dsn": "",
            "pgvector_embedding_dim": 384
        },
        "graph": {
            "neo4j_uri": "",
            "neo4j_user": "",
            "neo4j_password": ""
        },
        "second_brain": {
            "enable_graph_rag": True,
            "router_mode": "rule_based",
            "vector_confidence_threshold": 0.35,
            "context_budget_tokens": 8000,
            "vector_budget_ratio": 0.65,
            "graph_budget_ratio": 0.35,
            "decision_extraction_enabled": True,
            "graph_max_hops": 2,
            "graph_min_confidence": 0.70,
            "graph_extraction_mode": "heuristic_fallback",
            "memory_consolidation_enabled": True,
            "memory_decay_days": 90,
            "db_primary_strict": True,
            "ingestion_embedding_batch_size": 64,
            "ingestion_max_workers": 4
        },
        "retrieval": {
            "vector_top_k": 8,
            "graph_top_k_paths": 6,
            "hybrid_rerank_top_k": 8
        },
        "ui": {
            "title": "APCI - Asistentul Personalizat de Cercetare și Învățare",
            "theme": "dark"
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Merge configurația din fișier cu cea implicită
            merged_config = default_config.copy()
            
            # Merge secțiuni principale
            for key in ['models', 'gemini', 'chunking', 'rag', 'storage', 'graph', 'second_brain', 'retrieval', 'ui']:
                if key in file_config:
                    if key in merged_config:
                        merged_config[key].update(file_config[key])
                    else:
                        merged_config[key] = file_config[key]
            
            # Adaugă secțiuni noi din fișier
            for key, value in file_config.items():
                if key not in merged_config:
                    merged_config[key] = value
            
            return merged_config
            
        except Exception as e:
            st.error(f"Eroare la încărcarea configurației: {e}")
            return default_config
    
    return default_config

def build_apci_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Construiește configurația APCI folosită la inițializarea sistemului."""
    rag_config = config.get("rag", {})
    second_brain = config.get("second_brain", {})
    retrieval_cfg = config.get("retrieval", {})
    storage_cfg = config.get("storage", {})
    graph_cfg = config.get("graph", {})

    # Permite override din .env pentru secrete/connection strings.
    postgres_dsn = os.getenv("POSTGRES_DSN", storage_cfg.get("postgres_dsn", "")).strip()
    neo4j_uri = os.getenv("NEO4J_URI", graph_cfg.get("neo4j_uri", "")).strip()
    neo4j_user = os.getenv("NEO4J_USER", graph_cfg.get("neo4j_user", "")).strip()
    neo4j_password = os.getenv("NEO4J_PASSWORD", graph_cfg.get("neo4j_password", "")).strip()

    return {
        'model_name': config['models']['primary_llm'],
        'fallback_model': config['models']['fallback_llm'],
        'temperature': config['gemini']['temperature'],
        'max_tokens': config['gemini']['max_tokens'],
        'max_context_docs': config['chunking']['max_context_docs'],
        'chunk_size': config['chunking']['chunk_size'],
        'chunk_overlap': config['chunking']['chunk_overlap'],
        'cache_enabled': True,
        'rate_limit_rpm': config['gemini']['rate_limits']['requests_per_minute'],
        'rate_limit_rpd': config['gemini']['rate_limits']['requests_per_day'],
        'retrieval_candidates': rag_config.get("retrieval_candidates", 20),
        'rerank_top_k': rag_config.get("rerank_top_k", 8),
        'neighbor_window': rag_config.get("neighbor_window", 1),
        'hybrid_alpha': rag_config.get("hybrid_alpha", 0.7),
        'vector_top_k': retrieval_cfg.get("vector_top_k", 8),
        'graph_top_k_paths': retrieval_cfg.get("graph_top_k_paths", 6),
        'hybrid_rerank_top_k': retrieval_cfg.get("hybrid_rerank_top_k", 8),
        'enable_graph_rag': second_brain.get("enable_graph_rag", True),
        'router_mode': second_brain.get("router_mode", "rule_based"),
        'vector_confidence_threshold': second_brain.get("vector_confidence_threshold", 0.35),
        'context_budget_tokens': second_brain.get("context_budget_tokens", 8000),
        'vector_budget_ratio': second_brain.get("vector_budget_ratio", 0.65),
        'graph_budget_ratio': second_brain.get("graph_budget_ratio", 0.35),
        'decision_extraction_enabled': second_brain.get("decision_extraction_enabled", True),
        'graph_max_hops': second_brain.get("graph_max_hops", 2),
        'graph_min_confidence': second_brain.get("graph_min_confidence", 0.70),
        'graph_extraction_mode': second_brain.get("graph_extraction_mode", "heuristic_fallback"),
        'memory_consolidation_enabled': second_brain.get("memory_consolidation_enabled", True),
        'memory_decay_days': second_brain.get("memory_decay_days", 90),
        'db_primary_strict': second_brain.get("db_primary_strict", True),
        'ingestion_embedding_batch_size': second_brain.get("ingestion_embedding_batch_size", 64),
        'ingestion_max_workers': second_brain.get("ingestion_max_workers", 4),
        'postgres_dsn': postgres_dsn,
        'pgvector_embedding_dim': storage_cfg.get("pgvector_embedding_dim", 384),
        'neo4j_uri': neo4j_uri,
        'neo4j_user': neo4j_user,
        'neo4j_password': neo4j_password,
    }

def initialize_session_state():
    """Inițializează starea sesiunii"""
    if 'apci_system' not in st.session_state:
        st.session_state.apci_system = None
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'chat_manager' not in st.session_state and CHAT_SESSION_MANAGER_AVAILABLE:
        st.session_state.chat_manager = ChatSessionManager()

    if 'active_chat_id' not in st.session_state:
        st.session_state.active_chat_id = ""

    if 'chat_title_draft' not in st.session_state:
        st.session_state.chat_title_draft = ""

    if 'chat_title_input' not in st.session_state:
        st.session_state.chat_title_input = DEFAULT_CHAT_TITLE
    
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = None
    
    if 'document_manager' not in st.session_state and DOCUMENT_MANAGER_AVAILABLE:
        st.session_state.document_manager = DocumentManager()
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "💬 Chat"
    
    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []
    
    if 'startup_bootstrap_done' not in st.session_state:
        st.session_state.startup_bootstrap_done = False

    if 'documents_bootstrap_source' not in st.session_state:
        st.session_state.documents_bootstrap_source = "unknown"

    if 'last_sync_summary' not in st.session_state:
        st.session_state.last_sync_summary = {}

    if 'query_mode' not in st.session_state:
        st.session_state.query_mode = "topic_general"

    if 'active_topic_collection' not in st.session_state:
        st.session_state.active_topic_collection = ""

    if 'upload_collection' not in st.session_state:
        st.session_state.upload_collection = GENERAL_COLLECTION_NAME

def get_chat_manager():
    """Return DB chat manager when storage is healthy, otherwise file-based fallback."""
    apci_system = st.session_state.get("apci_system")
    repository = getattr(apci_system, "repository", None) if apci_system else None
    if repository and getattr(repository, "enabled", False):
        try:
            if repository.is_healthy():
                return repository
        except Exception:
            pass

    manager = st.session_state.get("chat_manager")
    return manager if CHAT_SESSION_MANAGER_AVAILABLE else None

def get_local_chat_manager() -> Optional["ChatSessionManager"]:
    """Return local file-based chat manager if available."""
    manager = st.session_state.get("chat_manager")
    return manager if CHAT_SESSION_MANAGER_AVAILABLE else None

def format_chat_label(session_info: Dict[str, Any]) -> str:
    """Build a compact label for chat selector."""
    title = session_info.get("title", DEFAULT_CHAT_TITLE)
    topic = session_info.get("topic_collection", "")
    count = session_info.get("message_count", 0)
    topic_label = topic if topic else "fara topic"
    return f"{title} [{topic_label}] ({count})"

def ensure_active_chat_session() -> None:
    """Ensure there is always one active chat session."""
    manager = get_chat_manager()
    if not manager:
        return

    sessions = manager.list_sessions()
    if not sessions:
        created = manager.create_session(
            title=DEFAULT_CHAT_TITLE,
            topic_collection=st.session_state.get("active_topic_collection", ""),
            query_mode=st.session_state.get("query_mode", "topic_general"),
        )
        if not created or not created.get("id"):
            return
        st.session_state.active_chat_id = created["id"]
        st.session_state.chat_history = []
        st.session_state.chat_title_draft = created.get("title", DEFAULT_CHAT_TITLE)
        st.session_state.chat_title_input = st.session_state.chat_title_draft
        return

    active_chat_id = st.session_state.get("active_chat_id", "")
    existing_ids = {item["id"] for item in sessions}
    if active_chat_id not in existing_ids:
        active_chat_id = sessions[0]["id"]
        st.session_state.active_chat_id = active_chat_id

    active_session = manager.load_session(active_chat_id) or {}
    st.session_state.chat_history = active_session.get("messages", [])
    st.session_state.chat_title_draft = active_session.get("title", DEFAULT_CHAT_TITLE)
    st.session_state.chat_title_input = st.session_state.chat_title_draft

def persist_active_chat_state(
    *,
    title: Optional[str] = None,
    topic_collection: Optional[str] = None,
    query_mode: Optional[str] = None,
) -> None:
    """Persist active chat messages and metadata."""
    manager = get_chat_manager()
    if not manager:
        return

    active_chat_id = st.session_state.get("active_chat_id", "")
    if not active_chat_id:
        return

    manager.save_session(
        active_chat_id,
        st.session_state.get("chat_history", []),
        title=title,
        topic_collection=topic_collection,
        query_mode=query_mode,
    )

def switch_active_chat(session_id: str) -> None:
    """Load selected chat in session state."""
    manager = get_chat_manager()
    if not manager or not session_id:
        return

    session_data = manager.load_session(session_id)
    if not session_data:
        return

    st.session_state.active_chat_id = session_id
    st.session_state.chat_history = session_data.get("messages", [])
    st.session_state.chat_title_draft = session_data.get("title", DEFAULT_CHAT_TITLE)
    st.session_state.chat_title_input = st.session_state.chat_title_draft

    saved_mode = session_data.get("query_mode")
    if saved_mode in QUERY_MODE_LABELS:
        st.session_state.query_mode = saved_mode

    st.session_state.active_topic_collection = session_data.get("topic_collection", "")

def create_and_switch_chat(
    title: str = DEFAULT_CHAT_TITLE,
    topic_collection: Optional[str] = None,
    query_mode: Optional[str] = None,
) -> None:
    """Create a chat and make it active."""
    manager = get_chat_manager()
    if not manager:
        return

    created = manager.create_session(
        title=title,
        topic_collection=topic_collection if topic_collection is not None else st.session_state.get("active_topic_collection", ""),
        query_mode=query_mode if query_mode is not None else st.session_state.get("query_mode", "topic_general"),
    )
    if created and created.get("id"):
        switch_active_chat(created["id"])

def delete_active_chat() -> None:
    """Delete active chat and fallback to another one."""
    manager = get_chat_manager()
    if not manager:
        return

    active_chat_id = st.session_state.get("active_chat_id", "")
    if not active_chat_id:
        return

    manager.delete_session(active_chat_id)
    st.session_state.active_chat_id = ""
    ensure_active_chat_session()

def ensure_apci_system(api_key: str):
    """Inițializează APCI o singură dată și îl returnează."""
    if not RAG_MODULE_AVAILABLE:
        return None

    if st.session_state.apci_system:
        return st.session_state.apci_system

    config = load_config()
    config_dict = build_apci_config_dict(config)
    st.session_state.apci_system = create_apci_system(api_key, config_dict)
    return st.session_state.apci_system

def collect_library_payload(doc_ids: List[str]) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str]]:
    """Construiește payload-ul (paths + metadata) pentru documentele din bibliotecă."""
    file_paths: List[str] = []
    metadata_by_path: Dict[str, Dict[str, Any]] = {}
    path_to_doc_id: Dict[str, str] = {}
    unique_paths = set()

    for doc_id in doc_ids:
        file_path = st.session_state.document_manager.get_document_file_path(doc_id)
        doc_info = st.session_state.document_manager.get_document_info(doc_id)
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

def bootstrap_documents_from_storage(api_key: str) -> None:
    """
    Startup rapid: inițializează sistemul și citește statusul indexului din DB.
    Nu rulează ingestie la pornire.
    """
    if st.session_state.get("startup_bootstrap_done", False):
        return

    st.session_state.startup_bootstrap_done = True

    try:
        apci_system = ensure_apci_system(api_key)
        if not apci_system:
            st.session_state.documents_loaded = False
            st.session_state.documents_bootstrap_source = "apci_unavailable"
            return

        storage_status = apci_system.get_storage_index_status()
        indexed_chunks = int(storage_status.get("indexed_chunks", 0) or 0)
        indexed_documents = int(storage_status.get("indexed_documents", 0) or 0)
        postgres_enabled = bool(storage_status.get("postgres_enabled", False))

        st.session_state.last_sync_summary = {
            "indexed_documents": indexed_documents,
            "indexed_chunks": indexed_chunks,
            "last_sync_at": storage_status.get("last_sync_at"),
            "storage_mode": storage_status.get("storage_mode", "local_fallback"),
            "db_primary_strict": bool(storage_status.get("db_primary_strict", False)),
            "bootstrap": True,
        }

        if postgres_enabled and indexed_chunks > 0:
            st.session_state.documents_loaded = True
            st.session_state.documents_bootstrap_source = "db_bootstrap"
        elif postgres_enabled:
            st.session_state.documents_loaded = False
            st.session_state.documents_bootstrap_source = "db_empty"
        else:
            st.session_state.documents_loaded = False
            st.session_state.documents_bootstrap_source = "local_pending"

        update_system_status()
    except Exception as exc:
        logger.error("Eroare la bootstrap din storage: %s", exc)
        st.session_state.documents_loaded = False
        st.session_state.documents_bootstrap_source = "bootstrap_error"

def sync_library_to_db(api_key: str, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Sync manual biblioteca -> index intern (DB primary, delta-only când DB este disponibilă).
    """
    result = {
        "success": False,
        "ingested": 0,
        "skipped": 0,
        "total_candidates": 0,
        "message": "",
    }

    if not DOCUMENT_MANAGER_AVAILABLE or not st.session_state.document_manager:
        result["message"] = "Biblioteca locală nu este disponibilă."
        return result

    if not RAG_MODULE_AVAILABLE:
        result["message"] = "Modulul RAG nu este disponibil."
        return result

    try:
        apci_system = ensure_apci_system(api_key)
        if not apci_system:
            result["message"] = "Nu s-a putut inițializa APCI."
            return result

        if doc_ids is None:
            all_docs = st.session_state.document_manager.get_all_documents()
            doc_ids = list(all_docs.keys())

        file_paths, metadata_by_path, path_to_doc_id = collect_library_payload(doc_ids)
        result["total_candidates"] = len(file_paths)
        if not file_paths:
            result["message"] = "Nu există documente valide pentru sync."
            st.session_state.last_sync_summary = {
                "indexed_documents": 0,
                "indexed_chunks": 0,
                "last_sync_at": None,
                "storage_mode": "unknown",
                "bootstrap": False,
            }
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
                st.session_state.document_manager.mark_document_indexed(doc_id)

        if to_ingest:
            ingest_metadata = {path: metadata_by_path[path] for path in to_ingest if path in metadata_by_path}
            success = apci_system.load_documents(to_ingest, metadata_by_path=ingest_metadata)
            if not success:
                result["message"] = "Eroare la ingestia documentelor în index."
                return result

            for ingested_path in to_ingest:
                doc_id = path_to_doc_id.get(ingested_path)
                if doc_id:
                    st.session_state.document_manager.mark_document_indexed(doc_id)

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
            or len(already_indexed) > 0
            or len(to_ingest) > 0
        )
        st.session_state.documents_bootstrap_source = "manual_sync"
        update_system_status()

        result["success"] = True
        if to_ingest:
            result["message"] = f"Sync finalizat: {len(to_ingest)} ingestate, {len(already_indexed)} deja indexate."
        else:
            result["message"] = f"Nimic de ingestat. {len(already_indexed)} documente sunt deja indexate."
        return result
    except Exception as exc:
        logger.error("Eroare la sync biblioteca -> DB: %s", exc)
        result["message"] = f"Eroare la sync: {exc}"
        return result

def migrate_local_chats_to_db(api_key: str) -> Dict[str, Any]:
    """
    Migrare one-time a chat-urilor locale (JSON) în Postgres.
    """
    result = {
        "success": False,
        "migrated": 0,
        "failed": 0,
        "total_candidates": 0,
        "message": "",
    }

    local_manager = get_local_chat_manager()
    if not local_manager:
        result["message"] = "Nu există chat manager local pentru migrare."
        return result

    apci_system = ensure_apci_system(api_key)
    repository = getattr(apci_system, "repository", None) if apci_system else None
    if not repository or not getattr(repository, "enabled", False):
        result["message"] = "Storage Postgres nu este activ."
        return result

    try:
        if not repository.is_healthy():
            result["message"] = "Conexiunea la Postgres nu este disponibilă."
            return result
    except Exception as exc:
        result["message"] = f"Conexiunea la Postgres a eșuat: {exc}"
        return result

    try:
        candidates = local_manager.list_sessions(include_migrated=False)
        result["total_candidates"] = len(candidates)
        if not candidates:
            result["success"] = True
            result["message"] = "Nu există chat-uri locale de migrat."
            return result

        failed_ids: List[str] = []
        for session_summary in candidates:
            session_id = session_summary.get("id", "")
            if not session_id:
                continue

            session_data = local_manager.load_session(session_id)
            if not session_data:
                failed_ids.append(session_id)
                continue

            created = repository.create_session(
                title=session_data.get("title", DEFAULT_CHAT_TITLE),
                topic_collection=session_data.get("topic_collection", ""),
                query_mode=session_data.get("query_mode", "topic_general"),
            )
            db_session_id = created.get("id") if created else ""
            if not db_session_id:
                failed_ids.append(session_id)
                continue

            messages = session_data.get("messages", [])
            migrated_ok = True
            for exchange in messages:
                if not repository.append_exchange(db_session_id, exchange):
                    migrated_ok = False
                    break

            if migrated_ok:
                repository.save_session(
                    db_session_id,
                    messages=[],
                    title=session_data.get("title", DEFAULT_CHAT_TITLE),
                    topic_collection=session_data.get("topic_collection", ""),
                    query_mode=session_data.get("query_mode", "topic_general"),
                )
                local_manager.mark_session_migrated(session_id, db_session_id=db_session_id)
                result["migrated"] += 1
            else:
                repository.delete_session(db_session_id)
                failed_ids.append(session_id)

        result["failed"] = len(failed_ids)
        result["success"] = result["failed"] == 0
        if result["success"]:
            result["message"] = f"Migrare completă: {result['migrated']} chat-uri migrate în Postgres."
        else:
            result["message"] = (
                f"Migrare parțială: {result['migrated']} migrate, "
                f"{result['failed']} eșuate."
            )
        return result
    except Exception as exc:
        logger.error("Eroare la migrarea chat-urilor locale în Postgres: %s", exc)
        result["message"] = f"Eroare la migrare: {exc}"
        return result

def setup_sidebar():
    """Configurează bara laterală"""
    with st.sidebar:
        st.title("Configurare APCI")
        
        # Configurare API Key
        st.subheader("Autentificare")
        
        # Încearcă să încărce API key din environment
        default_api_key = os.getenv('GOOGLE_API_KEY', '')
        
        api_key = st.text_input(
            "Google API Key (Gemini)",
            value=default_api_key,
            type="password",
            help="Introdu API key-ul pentru Google Gemini"
        )
        
        if api_key:
            st.success("API Key configurat")
        else:
            st.warning("API Key necesar pentru funcționare")
        
        st.divider()

        # Conversații persistente
        st.subheader("Conversații")
        chat_manager = get_chat_manager()
        local_chat_manager = get_local_chat_manager()

        if chat_manager and hasattr(chat_manager, "is_healthy") and local_chat_manager and api_key:
            pending_local_sessions = local_chat_manager.list_sessions(include_migrated=False)
            if pending_local_sessions:
                st.info(f"Există {len(pending_local_sessions)} chat-uri locale nemigrate.")
                if st.button("Migrează chat-urile locale în Postgres", use_container_width=True):
                    with st.spinner("Migrez chat-urile locale în Postgres..."):
                        migration_result = migrate_local_chats_to_db(api_key)
                    if migration_result.get("success"):
                        st.success(migration_result.get("message", "Migrare finalizată."))
                    else:
                        st.warning(migration_result.get("message", "Migrare parțială/eșuată."))
                    st.rerun()

        if chat_manager:
            ensure_active_chat_session()
            sessions = chat_manager.list_sessions()
            session_map = {item["id"]: item for item in sessions}
            session_ids = list(session_map.keys())

            if session_ids:
                active_chat_id = st.session_state.get("active_chat_id", "")
                default_index = session_ids.index(active_chat_id) if active_chat_id in session_ids else 0

                selected_chat_id = st.selectbox(
                    "Chat activ",
                    options=session_ids,
                    index=default_index,
                    format_func=lambda chat_id: format_chat_label(session_map[chat_id]),
                    key="chat_selector_sidebar",
                )

                if selected_chat_id != active_chat_id:
                    switch_active_chat(selected_chat_id)
                    st.rerun()

                col_chat1, col_chat2 = st.columns(2)
                with col_chat1:
                    if st.button("Chat nou", use_container_width=True):
                        create_and_switch_chat()
                        st.rerun()
                with col_chat2:
                    if st.button("Șterge chat", use_container_width=True):
                        delete_active_chat()
                        st.rerun()

                chat_title = st.text_input("Titlu chat", key="chat_title_input")
                if st.button("Salvează titlu", use_container_width=True):
                    active_id = st.session_state.get("active_chat_id", "")
                    if active_id:
                        chat_manager.rename_session(active_id, chat_title)
                        st.session_state.chat_title_draft = chat_title
                        st.success("Titlul a fost actualizat.")
                        st.rerun()
            else:
                st.info("Nu există chat-uri salvate încă.")
        else:
            st.warning("Persistența chat-urilor nu este disponibilă.")

        active_chat_manager = get_chat_manager()
        if active_chat_manager and hasattr(active_chat_manager, "is_healthy"):
            st.caption("Stocare chat-uri: Postgres")
        else:
            st.caption("Stocare chat-uri: data/chat_sessions/")
        st.divider()
        
        # Încărcare documente
        st.subheader("Încărcare Documente")

        target_collection = st.text_input(
            "Colecție destinație (topic sau general)",
            value=st.session_state.get("upload_collection", GENERAL_COLLECTION_NAME),
            help="Folosește 'general' pentru surse globale sau un nume de topic pentru surse tematice."
        )
        target_collection = normalize_collection_name(target_collection)
        st.session_state.upload_collection = target_collection
        
        uploaded_files = st.file_uploader(
            "Alege fișiere",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            help="Încarcă documente PDF, TXT sau Markdown"
        )
        
        if uploaded_files and api_key:
            if st.button("Procesează Documente", type="primary"):
                process_documents(uploaded_files, api_key, target_collection=target_collection)
        
        st.divider()
        
        # Biblioteca de documente - secțiune compactă în sidebar
        if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
            st.subheader("Biblioteca")
            
            # Afișează statistici rapide
            stats = st.session_state.document_manager.get_library_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documente", stats.get("total_documents", 0))
            with col2:
                st.metric("Colecții", stats.get("total_collections", 0))
            
            # Indicator stare bootstrap/sync
            if st.session_state.get("documents_loaded", False):
                source = st.session_state.get("documents_bootstrap_source", "unknown")
                if source == "db_bootstrap":
                    st.success("Startup rapid: context disponibil din DB.")
                elif source == "manual_sync":
                    st.success("Context actualizat prin sync manual.")
                else:
                    st.success("Context disponibil.")
            elif stats.get("total_documents", 0) > 0:
                st.warning("DB nu are index disponibil. Rulează sync manual.")
            else:
                st.info("Biblioteca este goală.")

            if api_key and stats.get("total_documents", 0) > 0:
                if st.button("Sync biblioteca în DB", type="secondary", use_container_width=True):
                    with st.spinner("Sincronizez biblioteca în DB..."):
                        sync_result = sync_library_to_db(api_key)
                    if sync_result.get("success"):
                        st.success(sync_result.get("message", "Sync finalizat."))
                    else:
                        st.error(sync_result.get("message", "Sync eșuat."))
                    st.rerun()
            
            # Buton pentru a deschide biblioteca completă
            if st.button("Deschide Biblioteca", type="secondary"):
                st.session_state.current_tab = "📚 Biblioteca"
                st.rerun()
            
            # Selector rapid pentru documente existente
            if stats.get("total_documents", 0) > 0:
                st.write("**Încărcare rapidă:**")
                
                # Dropdown cu documente disponibile
                docs = st.session_state.document_manager.get_all_documents()
                doc_options = {f"{doc['original_name']} ({doc['collection']})": doc_id 
                             for doc_id, doc in docs.items()}
                
                if doc_options:
                    selected_docs = st.multiselect(
                        "Selectează pentru încărcare rapidă",
                        options=list(doc_options.keys()),
                        key="quick_doc_selector",
                        help="Selectează documente pentru încărcare rapidă în chat"
                    )
                    
                    if selected_docs and api_key:
                        if st.button("Încarcă în Chat", type="secondary"):
                            doc_ids = [doc_options[doc_name] for doc_name in selected_docs]
                            load_documents_from_library(doc_ids, api_key)
        
        st.divider()
        
        # Configurări avansate
        with st.expander("Configurări Avansate"):
            config = load_config()
            
            st.subheader("Startup & Storage")
            storage_summary = st.session_state.get("last_sync_summary", {})
            if storage_summary:
                col_storage_1, col_storage_2 = st.columns(2)
                with col_storage_1:
                    st.metric("DB docs indexate", int(storage_summary.get("indexed_documents", 0) or 0))
                    st.metric("DB chunks indexate", int(storage_summary.get("indexed_chunks", 0) or 0))
                with col_storage_2:
                    st.write(f"Storage mode: `{storage_summary.get('storage_mode', 'unknown')}`")
                    strict_mode = bool(storage_summary.get("db_primary_strict", False))
                    st.caption(f"DB primary strict: {'ON' if strict_mode else 'OFF'}")
                    last_sync_at = storage_summary.get("last_sync_at")
                    if last_sync_at:
                        st.caption(f"Ultimul sync: {last_sync_at}")
                    else:
                        st.caption("Nu există sync anterior.")
            else:
                st.caption("Status storage indisponibil.")

            st.divider()
            
            # Model selection
            model_name = st.selectbox(
                "Model Principal",
                ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash"],
                index=0
            )
            
            # Temperature
            temperature = st.slider(
                "Temperatura",
                min_value=0.0,
                max_value=1.0,
                value=config.get("gemini", {}).get("temperature", 0.1),
                step=0.1,
                help="Controlează creativitatea răspunsurilor"
            )
            
            # Max tokens
            max_tokens = st.slider(
                "Max Tokens",
                min_value=1000,
                max_value=65536,  # Increased to 128K
                value=config.get("gemini", {}).get("max_tokens", 65536),
                step=1000,
                help="Numărul maxim de tokens pentru răspuns (128K pentru performanță optimă)"
            )
            
            # Context docs
            max_context_docs = st.slider(
                "Documente Context",
                min_value=1,
                max_value=10,
                value=config.get("chunking", {}).get("max_context_docs", 5),
                help="Numărul maxim de documente folosite ca context"
            )
            
            # Cache
            use_cache = st.checkbox(
                "Activează Cache",
                value=True,
                help="Salvează răspunsurile pentru a reduce costurile"
            )
        
        st.divider()
        
        # Status sistem
        if st.session_state.apci_system:
            st.subheader("Status Sistem")
            
            if st.button("Reîmprospătează Status"):
                update_system_status()
            
            if st.session_state.system_stats:
                display_system_status()
            
            # Cache embeddings management
            st.divider()
            st.subheader("Cache Embeddings")
            
            try:
                cache_info = st.session_state.apci_system.get_cache_info()
                embeddings_cache_stats = cache_info.get('embeddings_cache')
                
                if embeddings_cache_stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fișiere Cache", embeddings_cache_stats.get('cached_files', 0))
                        st.metric("Documente", embeddings_cache_stats.get('total_documents', 0))
                    with col2:
                        cache_size = embeddings_cache_stats.get('cache_size_mb', 0)
                        st.metric("Dimensiune", f"{cache_size} MB")
                    
                    if embeddings_cache_stats.get('cached_files', 0) > 0:
                        st.success("Cache activ - încărcare rapidă!")
                    else:
                        st.info("Cache gol - se va popula la prima încărcare")
                    
                    # Buton pentru curățarea cache-ului
                    if st.button("Curăță Cache Embeddings", help="Șterge cache-ul de embeddings pentru a forța regenerarea"):
                        st.session_state.apci_system.clear_embeddings_cache()
                        st.success("Cache embeddings curățat!")
                        st.rerun()
                else:
                    st.info("Cache embeddings nu este disponibil")
            except Exception as e:
                st.warning(f"Nu s-au putut obține statistici cache: {e}")

def process_documents(uploaded_files, api_key: str, target_collection: str = GENERAL_COLLECTION_NAME):
    """Procesează documentele încărcate - încărcare incrementală"""
    if not RAG_MODULE_AVAILABLE:
        st.error("Modulul RAG nu este disponibil")
        return
    
    try:
        with st.spinner("🔄 Inițializez sistemul APCI..."):
            apci_system = ensure_apci_system(api_key)
        if not apci_system:
            st.error("❌ Nu s-a putut inițializa sistemul APCI.")
            return
        
        # Salvează fișierele temporar
        temp_dir = Path("./data/uploaded_docs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        path_to_doc_id: Dict[str, str] = {}
        metadata_by_path = {}
        unique_paths = set()
        target_collection = normalize_collection_name(target_collection)
        
        with st.spinner("💾 Salvez documentele..."):
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.write(f"✅ Salvat: {uploaded_file.name}")
                
                # Adaugă în biblioteca de documente
                if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
                    doc_id = st.session_state.document_manager.add_document(
                        str(file_path),
                        collection=target_collection,
                        description=f"Încărcat la {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    if doc_id:
                        doc_info = st.session_state.document_manager.get_document_info(doc_id)
                        library_file_path = st.session_state.document_manager.get_document_file_path(doc_id)

                        if library_file_path and doc_info:
                            resolved_path = str(library_file_path.resolve())
                            if resolved_path not in unique_paths:
                                unique_paths.add(resolved_path)
                                file_paths.append(resolved_path)
                            path_to_doc_id[resolved_path] = doc_id
                            metadata_by_path[resolved_path] = build_metadata_entry(
                                doc_id,
                                doc_info,
                                library_file_path
                            )
                            st.write(f"📚 Adăugat în bibliotecă: {doc_info.get('original_name', uploaded_file.name)}")
                else:
                    resolved_path = str(file_path.resolve())
                    if resolved_path not in unique_paths:
                        unique_paths.add(resolved_path)
                        file_paths.append(resolved_path)
                    path_to_doc_id[resolved_path] = ""
                    metadata_by_path[resolved_path] = {
                        "filename": uploaded_file.name,
                        "file_type": Path(uploaded_file.name).suffix,
                        "collection": target_collection,
                        "source_scope": "general" if is_general_collection(target_collection) else "topic",
                        "source_path": resolved_path
                    }
        
        if not file_paths:
            st.error("❌ Nu există documente valide pentru procesare")
            return

        filtered = apci_system.filter_paths_for_ingestion(
            file_paths,
            metadata_by_path=metadata_by_path,
        )
        to_ingest = filtered.get("to_ingest", [])
        already_indexed = filtered.get("already_indexed", [])

        for existing_path in already_indexed:
            existing_doc_id = path_to_doc_id.get(existing_path)
            if existing_doc_id and DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
                st.session_state.document_manager.mark_document_indexed(existing_doc_id)
        
        if to_ingest:
            ingest_metadata = {path: metadata_by_path[path] for path in to_ingest if path in metadata_by_path}
            with st.spinner("🧠 Procesez și indexez documentele noi..."):
                success = apci_system.load_documents(
                    to_ingest,
                    metadata_by_path=ingest_metadata
                )
        else:
            success = True

        if success:
            st.session_state.documents_loaded = True
            st.session_state.documents_bootstrap_source = "manual_sync"

            # Marchează documentele ca indexate în bibliotecă
            if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
                indexed_paths = set(to_ingest).union(set(already_indexed))
                for indexed_path in indexed_paths:
                    indexed_doc_id = path_to_doc_id.get(indexed_path)
                    if indexed_doc_id:
                        st.session_state.document_manager.mark_document_indexed(indexed_doc_id)

            # Actualizează statusul
            update_system_status()
            if to_ingest:
                st.success(f"✅ {len(to_ingest)} documente procesate; {len(already_indexed)} deja indexate.")
            else:
                st.success(f"✅ Toate documentele ({len(already_indexed)}) erau deja indexate.")
        else:
            st.error("❌ Eroare la procesarea documentelor")
    
    except Exception as e:
        st.error(f"❌ Eroare: {str(e)}")
        logger.error(f"Eroare la procesarea documentelor: {e}")

def load_documents_from_library(doc_ids: List[str], api_key: str):
    """Încarcă documente din bibliotecă - încărcare incrementală"""
    if not RAG_MODULE_AVAILABLE or not DOCUMENT_MANAGER_AVAILABLE:
        st.error("Modulele necesare nu sunt disponibile")
        return
    
    try:
        with st.spinner("📚 Sincronizez documentele selectate..."):
            sync_result = sync_library_to_db(api_key, doc_ids=doc_ids)

        if sync_result.get("success"):
            st.success(sync_result.get("message", "Documentele au fost sincronizate."))
            st.rerun()
        else:
            st.error(sync_result.get("message", "Eroare la sincronizare."))
    
    except Exception as e:
        st.error(f"❌ Eroare: {str(e)}")
        logger.error(f"Eroare la încărcarea din bibliotecă: {e}")

def update_system_status():
    """Actualizează statusul sistemului"""
    if st.session_state.apci_system:
        try:
            status = st.session_state.apci_system.get_system_status()
            st.session_state.system_stats = status

            storage_status = status.get("storage_index_status", {})
            if storage_status:
                st.session_state.last_sync_summary = {
                    "indexed_documents": storage_status.get("indexed_documents", 0),
                    "indexed_chunks": storage_status.get("indexed_chunks", 0),
                    "last_sync_at": storage_status.get("last_sync_at"),
                    "storage_mode": storage_status.get("storage_mode", "local_fallback"),
                    "bootstrap": st.session_state.last_sync_summary.get("bootstrap", False),
                }
        except Exception as e:
            st.error(f"Eroare la obținerea statusului: {e}")

def display_system_status():
    """Afișează statusul sistemului"""
    if not st.session_state.system_stats:
        return
    
    stats = st.session_state.system_stats
    
    # Model info
    model_info = stats.get('model_info', {})
    st.write(f"**Model**: {model_info.get('model_name', 'N/A')}")
    
    # Statistici generale
    general_stats = stats.get('stats', {})
    st.write(f"**Queries**: {general_stats.get('total_queries', 0)}")
    st.write(f"**Cache hits**: {general_stats.get('cache_hits', 0)}")
    st.write(f"**Documente**: {general_stats.get('documents_indexed', 0)}")
    
    # Cache stats
    cache_stats = stats.get('cache_stats')
    if cache_stats:
        hit_rate = cache_stats.get('hit_rate_percent', 0)
        st.write(f"**Cache hit rate**: {hit_rate:.1f}%")
    
    # Embeddings cache stats
    embeddings_cache_stats = stats.get('embeddings_cache_stats')
    if embeddings_cache_stats:
        cached_files = embeddings_cache_stats.get('cached_files', 0)
        if cached_files > 0:
            st.write(f"**Cache embeddings**: {cached_files} fișiere")
    
    # Rate limiter
    rate_stats = stats.get('rate_limiter')
    if rate_stats:
        st.write(f"**Requests/min**: {rate_stats.get('minute_requests', 0)}")

    storage_status = stats.get("storage_index_status", {})
    if storage_status:
        st.write(f"**Storage mode**: {storage_status.get('storage_mode', 'unknown')}")
        st.write(f"**DB indexed docs**: {storage_status.get('indexed_documents', 0)}")
        st.write(f"**DB indexed chunks**: {storage_status.get('indexed_chunks', 0)}")

def main_chat_interface():
    """Interfața principală de chat"""
    # Verifică dacă sistemul este gata
    if not st.session_state.apci_system:
        st.title("🧠 APCI - Asistentul Personalizat de Cercetare și Învățare")
        
        # Verifică dacă există documente în bibliotecă
        if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
            stats = st.session_state.document_manager.get_library_stats()
            if stats.get('total_documents', 0) > 0:
                st.info("🚀 Configurează API key-ul, apoi rulează sync manual biblioteca -> DB.")
            else:
                st.info("👈 Configurează API key-ul și încarcă documente în bara laterală pentru a începe.")
        else:
            st.info("👈 Configurează API key-ul și încarcă documente în bara laterală pentru a începe.")
        
        # Demo queries pentru test
        st.subheader("🎯 Exemple de întrebări")
        example_queries = [
            "Ce este machine learning și cum funcționează?",
            "Explică diferența între AI, ML și Deep Learning",
            "Care sunt avantajele și dezavantajele AI în educație?",
            "Cum poate fi folosită tehnologia AI pentru cercetare?"
        ]
        
        for query in example_queries:
            if st.button(f"💡 {query}", key=f"demo_{hash(query)}"):
                if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
                    stats = st.session_state.document_manager.get_library_stats()
                    if stats.get('total_documents', 0) > 0:
                        st.info("Configurează API key-ul mai întâi, apoi rulează sync manual biblioteca -> DB.")
                    else:
                        st.info("Încarcă documente mai întâi pentru a primi răspunsuri personalizate!")
                else:
                    st.info("Încarcă documente mai întâi pentru a primi răspunsuri personalizate!")
        
        return
    
    if not st.session_state.documents_loaded:
        st.title("🧠 APCI - Asistentul Personalizat de Cercetare și Învățare")

        bootstrap_source = st.session_state.get("documents_bootstrap_source", "unknown")
        if bootstrap_source == "db_empty":
            st.warning("DB este conectată, dar indexul este gol.")
        elif bootstrap_source == "local_pending":
            st.warning("Storage DB nu este activă. Poți folosi ingestia locală manuală.")
        elif bootstrap_source == "bootstrap_error":
            st.error("Bootstrap din DB a eșuat. Verifică conexiunea la Postgres.")
        else:
            st.info("Nu există context indexat disponibil.")

        if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
            stats = st.session_state.document_manager.get_library_stats()
            if stats.get('total_documents', 0) > 0:
                st.info("Biblioteca locală are documente. Rulează sync manual pentru a le indexa în DB.")
                api_key = os.getenv('GOOGLE_API_KEY', '')
                if api_key and st.button("Sync biblioteca în DB", type="primary"):
                    with st.spinner("Sincronizez biblioteca în DB..."):
                        sync_result = sync_library_to_db(api_key)
                    if sync_result.get("success"):
                        st.success(sync_result.get("message", "Sync finalizat."))
                    else:
                        st.error(sync_result.get("message", "Sync eșuat."))
                    st.rerun()
            else:
                st.warning("📚 Biblioteca este goală. Încarcă documente pentru a începe conversația.")
        else:
            st.warning("📚 Încarcă documente pentru a începe conversația.")

        return
    
    st.title("🧠 APCI - Chat")

    chat_manager = get_chat_manager()
    active_chat_id = st.session_state.get("active_chat_id", "")
    active_chat_data = chat_manager.load_session(active_chat_id) if chat_manager and active_chat_id else None
    if active_chat_data:
        topic_label = active_chat_data.get("topic_collection") or "fără topic"
        st.caption(
            f"Chat activ: **{active_chat_data.get('title', DEFAULT_CHAT_TITLE)}** · "
            f"Topic: **{topic_label}**"
        )

    # Configurare mod interogare + topic activ
    mode_options = list(QUERY_MODE_LABELS.keys())
    current_mode = st.session_state.get("query_mode", "topic_general")
    mode_index = mode_options.index(current_mode) if current_mode in mode_options else mode_options.index("topic_general")

    col_mode, col_topic = st.columns([1, 1])
    with col_mode:
        query_mode = st.selectbox(
            "Mod interogare",
            options=mode_options,
            index=mode_index,
            format_func=lambda mode: QUERY_MODE_LABELS.get(mode, mode)
        )
    st.session_state.query_mode = query_mode

    topic_collections = []
    if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
        collections = st.session_state.document_manager.get_collections()
        topic_collections = sorted([name for name in collections.keys() if not is_general_collection(name)])

    with col_topic:
        if query_mode in {"topic", "topic_general"}:
            if topic_collections:
                current_topic = st.session_state.get("active_topic_collection", "")
                default_topic_index = topic_collections.index(current_topic) if current_topic in topic_collections else 0
                selected_topic = st.selectbox(
                    "Topic activ",
                    options=topic_collections,
                    index=default_topic_index,
                    help="Topicul folosit pentru filtrarea documentelor tematice."
                )
                st.session_state.active_topic_collection = selected_topic
            else:
                st.session_state.active_topic_collection = ""
                st.warning("Nu există colecții tematice. Creează un topic prin încărcare într-o colecție nouă.")

    persist_active_chat_state(
        topic_collection=st.session_state.get("active_topic_collection", ""),
        query_mode=st.session_state.get("query_mode", "topic_general"),
    )

    col_actions1, _ = st.columns([1, 2])
    with col_actions1:
        if st.button("🗑️ Curăță chat activ", use_container_width=True):
            st.session_state.chat_history = []
            if chat_manager and active_chat_id:
                chat_manager.clear_session(active_chat_id)
            st.rerun()

    if st.session_state.apci_system and hasattr(st.session_state.apci_system, "get_second_brain_status"):
        second_brain_status = st.session_state.apci_system.get_second_brain_status()
        with st.expander("Second Brain status", expanded=False):
            db_status = second_brain_status.get("db", {})
            local_status = second_brain_status.get("local", {})
            col_sb1, col_sb2, col_sb3 = st.columns(3)
            with col_sb1:
                st.metric("DB task-uri deschise", int(db_status.get("tasks_open_count", 0) or 0))
                st.metric("DB decizii", int(db_status.get("decisions_count", 0) or 0))
            with col_sb2:
                st.metric("Local task-uri", int(local_status.get("open_tasks", 0) or 0))
                st.metric("Local memorii", int(local_status.get("episodes", 0) or 0))
            with col_sb3:
                st.caption(f"Consolidare memorie: {'ON' if second_brain_status.get('memory_consolidation_enabled') else 'OFF'}")
                st.caption(f"Memory decay: {second_brain_status.get('memory_decay_days', 0)} zile")
                st.caption(f"Ultima consolidare: {second_brain_status.get('last_memory_consolidation_at') or 'N/A'}")

    if st.session_state.apci_system and hasattr(st.session_state.apci_system, "list_tasks"):
        active_collection_for_tasks = st.session_state.get("active_topic_collection", "") or None
        task_items = st.session_state.apci_system.list_tasks(
            active_collection=active_collection_for_tasks,
            limit=15,
        )
        with st.expander("Task-uri si remindere", expanded=False):
            if not task_items:
                st.caption("Nu exista task-uri active.")
            else:
                for task in task_items:
                    task_id = task.get("id")
                    title = task.get("title", "Task")
                    status_value = task.get("status", "open")
                    topic_value = task.get("topic_collection", "") or "general"
                    st.write(f"**{title}**")
                    st.caption(f"status={status_value} - topic={topic_value}")
                    col_task1, col_task2, col_task3 = st.columns(3)
                    with col_task1:
                        if st.button("Open", key=f"task_open_{task_id}", use_container_width=True):
                            if st.session_state.apci_system.update_task_status(task_id, "open"):
                                st.rerun()
                    with col_task2:
                        if st.button("In progress", key=f"task_progress_{task_id}", use_container_width=True):
                            if st.session_state.apci_system.update_task_status(task_id, "in_progress"):
                                st.rerun()
                    with col_task3:
                        if st.button("Done", key=f"task_done_{task_id}", use_container_width=True):
                            if st.session_state.apci_system.update_task_status(task_id, "done"):
                                st.rerun()
                    st.divider()

    st.subheader("💬 Conversație")
    if not st.session_state.chat_history:
        st.info("Chatul este gol. Pune prima întrebare pentru acest topic.")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["response"])

            answer_origin = chat.get("answer_origin", "internal")
            if answer_origin == "external":
                st.warning("Răspuns extern: sursele interne au fost insuficiente.")

            if chat.get("sources"):
                with st.expander("📚 Surse"):
                    for j, source in enumerate(chat["sources"], 1):
                        filename = source.get("filename", f"Document {j}")
                        collection = source.get("collection", GENERAL_COLLECTION_NAME)
                        st.write(f"{j}. **{filename}** ({collection})")

            if chat.get("graph_sources"):
                with st.expander("🕸️ Surse Graph"):
                    for j, source in enumerate(chat["graph_sources"], 1):
                        concept = source.get("concept", f"Concept {j}")
                        filename = source.get("filename", "N/A")
                        citation = source.get("citation", f"G{j}")
                        st.write(f"{citation}. **{concept}** -> {filename}")

            if chat.get("memory_hits"):
                with st.expander("🧠 Memory Hits"):
                    for j, item in enumerate(chat["memory_hits"], 1):
                        title = item.get("title", f"Memory {j}")
                        topic = item.get("topic_collection", "")
                        memory_type = item.get("memory_type", "semantic")
                        st.write(f"{j}. **[{memory_type}] {title}** ({topic})")

            if chat.get("provenance"):
                with st.expander("🔎 Proveniență"):
                    for entry in chat["provenance"]:
                        citation = entry.get("citation", "?")
                        kind = entry.get("kind", "")
                        score = entry.get("score", 0.0)
                        st.write(f"{citation} · {kind} · scor={score:.3f}")

            if chat.get("external_sources"):
                with st.expander("🌐 Surse Externe"):
                    for j, source in enumerate(chat["external_sources"], 1):
                        title = source.get("title", f"Sursă externă {j}")
                        url = source.get("url", "")
                        if url:
                            st.markdown(f"{j}. [{title}]({url})")
                        else:
                            st.write(f"{j}. {title}")

            if chat.get("response_time"):
                response_time = chat["response_time"]
                cached = "💾 (cache)" if chat.get("cached") else "🆕 (nou)"
                route_used = chat.get("route_used", "vector")
                st.caption(f"⏱️ {response_time:.2f}s {cached} · route={route_used}")
                router_reason = chat.get("router_reason", "")
                answer_origin_value = chat.get("answer_origin", "internal")
                if router_reason:
                    st.caption(f"origin={answer_origin_value} - reason={router_reason}")
                else:
                    st.caption(f"origin={answer_origin_value}")

    user_question = st.chat_input("Pune o întrebare despre sursele tale...")
    if user_question and user_question.strip():
        process_question(user_question.strip())

def library_interface():
    """Interfața pentru biblioteca de documente"""
    if not DOCUMENT_MANAGER_AVAILABLE or not st.session_state.document_manager:
        st.error("Document Manager nu este disponibil")
        return
    
    st.title("📚 Biblioteca de Documente")
    
    # Statistici generale
    stats = st.session_state.document_manager.get_library_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Documente", stats.get("total_documents", 0))
    with col2:
        st.metric("📁 Colecții", stats.get("total_collections", 0))
    with col3:
        st.metric("✅ Indexate", stats.get("indexed_documents", 0))
    with col4:
        size_mb = stats.get("total_size_bytes", 0) / (1024 * 1024)
        st.metric("💾 Dimensiune", f"{size_mb:.1f} MB")
    
    # Tabs pentru diferite vizualizări
    tab1, tab2, tab3 = st.tabs(["📋 Toate Documentele", "📁 Pe Colecții", "🔍 Căutare"])
    
    with tab1:
        show_all_documents()
    
    with tab2:
        show_documents_by_collection()
    
    with tab3:
        show_search_interface()

def show_all_documents():
    """Afișează toate documentele"""
    docs = st.session_state.document_manager.get_all_documents()
    
    if not docs:
        st.info("📭 Biblioteca este goală. Încarcă câteva documente pentru a începe!")
        return
    
    # Header pentru tabel
    col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 1, 1, 1, 1.5])
    with col1:
        st.write("**Nume Document**")
    with col2:
        st.write("**Colecție**")
    with col3:
        st.write("**Tip**")
    with col4:
        st.write("**Dimensiune**")
    with col5:
        st.write("**Indexat**")
    with col6:
        st.write("**Selectează**")
    
    st.divider()
    
    # Inițializare pentru selecție
    if f"selected_docs_library" not in st.session_state:
        st.session_state.selected_docs_library = set()
    
    # Butoane de control pentru selecție
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("✅ Selectează Tot"):
            st.session_state.selected_docs_library = set(docs.keys())
            st.rerun()
    
    with col_btn2:
        if st.button("❌ Deselectează Tot"):
            st.session_state.selected_docs_library = set()
            st.rerun()
    
    with col_btn3:
        selected_count = len(st.session_state.selected_docs_library)
        if selected_count > 0:
            api_key = os.getenv('GOOGLE_API_KEY', '')
            if st.button(f"🚀 Încarcă Selectate ({selected_count})", type="primary"):
                if api_key:
                    load_documents_from_library(list(st.session_state.selected_docs_library), api_key)
                    st.session_state.selected_docs_library = set()  # Reset după încărcare
                    st.rerun()
                else:
                    st.error("API Key necesar!")
    
    st.divider()
    
    # Lista documentelor
    for doc_id, doc_info in docs.items():
        col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 1, 1, 1, 1.5])
        
        with col1:
            st.write(f"📄 {doc_info['original_name']}")
            if doc_info.get('description'):
                st.caption(doc_info['description'])
        
        with col2:
            st.write(f"📁 {doc_info.get('collection', 'default')}")
        
        with col3:
            st.write(doc_info.get('file_type', 'N/A').upper())
        
        with col4:
            size_kb = doc_info.get('file_size', 0) / 1024
            st.write(f"{size_kb:.1f} KB")
        
        with col5:
            if doc_info.get('indexed', False):
                st.write("✅")
            else:
                st.write("❌")
        
        with col6:
            # Checkbox pentru selecție multiplă
            selected = st.checkbox("", key=f"select_{doc_id}", label_visibility="collapsed")
            if f"selected_docs_library" not in st.session_state:
                st.session_state.selected_docs_library = set()
            
            if selected:
                st.session_state.selected_docs_library.add(doc_id)
            elif doc_id in st.session_state.selected_docs_library:
                st.session_state.selected_docs_library.remove(doc_id)
            
            # Buton pentru ștergere
            if st.button("🗑️", key=f"delete_{doc_id}", help="Șterge document"):
                if st.session_state.document_manager.remove_document(doc_id):
                    st.success(f"Document {doc_info['original_name']} șters!")
                    st.rerun()
                else:
                    st.error("Eroare la ștergere!")
        
        st.divider()

def show_documents_by_collection():
    """Afișează documentele grupate pe colecții"""
    collections = st.session_state.document_manager.get_collections()
    
    if not collections:
        st.info("📁 Nu există colecții. Documentele vor fi adăugate în colecția 'default'.")
        return
    
    # Selector pentru colecție
    collection_names = list(collections.keys())
    selected_collection = st.selectbox("📁 Selectează Colecția", collection_names)
    
    if selected_collection:
        st.subheader(f"📁 Colecția: {selected_collection}")
        
        docs = st.session_state.document_manager.get_documents_by_collection(selected_collection)
        
        if not docs:
            st.info(f"📭 Colecția '{selected_collection}' este goală.")
            return
        
        # Opțiuni pentru colecție
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"🚀 Încarcă Colecția ({len(docs)} documente)", type="primary"):
                api_key = os.getenv('GOOGLE_API_KEY', '')
                if api_key:
                    doc_ids = [doc['id'] for doc in docs]
                    load_documents_from_library(doc_ids, api_key)
                else:
                    st.error("API Key necesar!")
        
        with col2:
            if st.button("✅ Selectează toată colecția"):
                if f"selected_docs_library" not in st.session_state:
                    st.session_state.selected_docs_library = set()
                for doc in docs:
                    st.session_state.selected_docs_library.add(doc['id'])
                st.rerun()
        
        with col3:
            if st.button("❌ Deselectează colecția"):
                if f"selected_docs_library" not in st.session_state:
                    st.session_state.selected_docs_library = set()
                for doc in docs:
                    st.session_state.selected_docs_library.discard(doc['id'])
                st.rerun()
        
        st.divider()
        
        # Lista documentelor din colecție
        for doc in docs:
            col1, col2, col3 = st.columns([5, 1, 1])
            
            with col1:
                st.write(f"📄 **{doc['original_name']}**")
                if doc.get('description'):
                    st.caption(doc['description'])
                
                # Info suplimentare
                indexed_status = "✅ Indexat" if doc.get('indexed', False) else "❌ Neindexat"
                size_kb = doc.get('file_size', 0) / 1024
                st.caption(f"{doc.get('file_type', 'N/A').upper()} • {size_kb:.1f} KB • {indexed_status}")
            
            with col2:
                # Checkbox pentru selecție
                if f"selected_docs_library" not in st.session_state:
                    st.session_state.selected_docs_library = set()
                
                selected = st.checkbox("", key=f"select_coll_{doc['id']}", 
                                     value=doc['id'] in st.session_state.selected_docs_library,
                                     label_visibility="collapsed")
                
                if selected:
                    st.session_state.selected_docs_library.add(doc['id'])
                elif doc['id'] in st.session_state.selected_docs_library:
                    st.session_state.selected_docs_library.remove(doc['id'])
            
            with col3:
                if st.button("🗑️", key=f"delete_coll_{doc['id']}", help="Șterge document"):
                    if st.session_state.document_manager.remove_document(doc['id']):
                        st.success(f"Document {doc['original_name']} șters!")
                        st.rerun()
                    else:
                        st.error("Eroare la ștergere!")
            
            st.divider()

def show_search_interface():
    """Interfața de căutare în bibliotecă"""
    st.subheader("🔍 Caută în Bibliotecă")
    
    # Input pentru căutare
    search_query = st.text_input(
        "Caută documente după nume sau descriere:",
        placeholder="De exemplu: machine learning, AI, tutorial..."
    )
    
    if search_query:
        results = st.session_state.document_manager.search_documents(search_query)
        
        if results:
            st.write(f"🎯 Găsite {len(results)} rezultate:")
            
            # Butoane pentru selecție în bulk în rezultatele căutării
            col_search1, col_search2, col_search3 = st.columns(3)
            with col_search1:
                if st.button("✅ Selectează toate rezultatele"):
                    if f"selected_docs_library" not in st.session_state:
                        st.session_state.selected_docs_library = set()
                    for doc in results:
                        st.session_state.selected_docs_library.add(doc['id'])
                    st.rerun()
            
            with col_search2:
                if st.button("❌ Deselectează rezultatele"):
                    if f"selected_docs_library" not in st.session_state:
                        st.session_state.selected_docs_library = set()
                    for doc in results:
                        st.session_state.selected_docs_library.discard(doc['id'])
                    st.rerun()
            
            st.divider()
            
            for doc in results:
                col1, col2, col3 = st.columns([5, 1, 1])
                
                with col1:
                    st.write(f"📄 **{doc['original_name']}**")
                    if doc.get('description'):
                        st.caption(doc['description'])
                    
                    # Highlight search term în numele documentului
                    if search_query.lower() in doc['original_name'].lower():
                        st.caption("🎯 Găsit în nume")
                    elif search_query.lower() in doc.get('description', '').lower():
                        st.caption("🎯 Găsit în descriere")
                    
                    st.caption(f"📁 Colecție: {doc.get('collection', 'default')} • {doc.get('file_type', 'N/A').upper()}")
                
                with col2:
                    # Checkbox pentru selecție
                    if f"selected_docs_library" not in st.session_state:
                        st.session_state.selected_docs_library = set()
                    
                    selected = st.checkbox("", key=f"select_search_{doc['id']}", 
                                         value=doc['id'] in st.session_state.selected_docs_library,
                                         label_visibility="collapsed")
                    
                    if selected:
                        st.session_state.selected_docs_library.add(doc['id'])
                    elif doc['id'] in st.session_state.selected_docs_library:
                        st.session_state.selected_docs_library.remove(doc['id'])
                
                with col3:
                    if st.button("🗑️", key=f"search_delete_{doc['id']}", help="Șterge document"):
                        if st.session_state.document_manager.remove_document(doc['id']):
                            st.success(f"Document {doc['original_name']} șters!")
                            st.rerun()
                        else:
                            st.error("Eroare la ștergere!")
                
                st.divider()
        else:
            st.info(f"❌ Nu s-au găsit documente pentru '{search_query}'")

def main_interface():
    """Interfața principală simplificată."""
    if not DOCUMENT_MANAGER_AVAILABLE:
        main_chat_interface()
        return

    nav_options = ["💬 Chat", "📚 Biblioteca"]
    current_tab = st.session_state.get("current_tab", "💬 Chat")
    if current_tab not in nav_options:
        current_tab = "💬 Chat"

    selected_tab = st.radio(
        "Navigare",
        options=nav_options,
        index=nav_options.index(current_tab),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.current_tab = selected_tab

    if selected_tab == "📚 Biblioteca":
        library_interface()
    else:
        main_chat_interface()

def process_question(question: str):
    """Procesează o întrebare și afișează răspunsul"""
    try:
        with st.spinner("🤔 Gândesc..."):
            start_time = time.time()
            
            # Obține răspunsul de la APCI
            result = st.session_state.apci_system.query(
                question,
                retrieval_mode=st.session_state.get("query_mode", "topic_general"),
                active_collection=st.session_state.get("active_topic_collection", "") or None,
                general_collection=GENERAL_COLLECTION_NAME,
                include_memory=True,
            )
            
            end_time = time.time()
            
            # Adaugă în istoric
            chat_entry = {
                "question": question,
                "response": result.get("response", "Nu am putut genera un răspuns."),
                "sources": result.get("sources", []),
                "graph_sources": result.get("graph_sources", []),
                "memory_hits": result.get("memory_hits", []),
                "tasks": result.get("tasks", []),
                "provenance": result.get("provenance", []),
                "external_sources": result.get("external_sources", []),
                "response_time": result.get("response_time", end_time - start_time),
                "cached": result.get("cached", False),
                "model_used": result.get("model_used", "Unknown"),
                "answer_origin": result.get("answer_origin", "internal"),
                "retrieval_mode": result.get("retrieval_mode", st.session_state.get("query_mode", "topic_general")),
                "route_used": result.get("route_used", "vector"),
                "router_reason": result.get("router_reason", ""),
            }
            
            st.session_state.chat_history.append(chat_entry)

            chat_manager = get_chat_manager()
            active_chat_id = st.session_state.get("active_chat_id", "")
            if chat_manager and active_chat_id:
                chat_manager.append_exchange(active_chat_id, chat_entry)
                persisted_chat = chat_manager.load_session(active_chat_id) or {}
                st.session_state.chat_title_draft = persisted_chat.get("title", DEFAULT_CHAT_TITLE)
                chat_manager.save_session(
                    active_chat_id,
                    persisted_chat.get("messages", []),
                    topic_collection=st.session_state.get("active_topic_collection", ""),
                    query_mode=st.session_state.get("query_mode", "topic_general"),
                )
             
            # Actualizează statusul
            update_system_status()
            
            # Rerun pentru a afișa noul mesaj
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Eroare la procesarea întrebării: {str(e)}")
        logger.error(f"Eroare la procesarea întrebării: {e}")

def check_api_key_setup():
    """Verifică și configurează API key-ul"""
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        st.error("🔑 **API Key Google AI nu este configurat!**")
        
        with st.expander("🔧 **Configurare API Key**", expanded=True):
            st.markdown("""
            ### Cum obții un API Key Google AI:
            
            1. **Mergi la** [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. **Fă clic pe** "Create API Key" 
            3. **Copiază** cheia generată
            4. **Introdu-o mai jos** sau setează variabila de mediu `GOOGLE_API_KEY`
            """)
            
            # Input pentru API key
            api_key_input = st.text_input(
                "🔑 Introdu API Key-ul tău:", 
                type="password",
                placeholder="AIzaSy..."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Salvează în .env", type="primary"):
                    if api_key_input:
                        try:
                            env_path = Path(__file__).parent.parent / ".env"
                            
                            # Citește .env existent
                            env_content = []
                            if env_path.exists():
                                with open(env_path, 'r') as f:
                                    env_content = [line for line in f.readlines() 
                                                 if not line.startswith('GOOGLE_API_KEY')]
                            
                            # Adaugă noul API key
                            env_content.insert(0, f"GOOGLE_API_KEY={api_key_input}\n")
                            
                            # Salvează
                            with open(env_path, 'w') as f:
                                f.writelines(env_content)
                            
                            st.success("✅ API Key salvat! Reîncarcă pagina.")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Eroare la salvare: {e}")
                    else:
                        st.warning("⚠️ Introdu API key-ul mai întâi!")
            
            with col2:
                if st.button("🧪 Testează Conexiunea"):
                    if api_key_input:
                        with st.spinner("Testez conexiunea..."):
                            try:
                                # Setează temporar API key-ul
                                os.environ['GOOGLE_API_KEY'] = api_key_input
                                
                                # Testează conexiunea
                                from google import genai
                                client = genai.Client(api_key=api_key_input)

                                response = client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents="Spune doar 'Test reușit!'"
                                )

                                if response and getattr(response, "text", ""):
                                    st.success("✅ Conexiunea funcționează!")
                                else:
                                    st.error("❌ Test eșuat - verifică API key-ul")
                                    
                            except Exception as e:
                                st.error(f"❌ Eroare la test: {str(e)[:100]}...")
                    else:
                        st.warning("⚠️ Introdu API key-ul mai întâi!")
            
            st.info("💡 **Sau** setează variabila de mediu: `set GOOGLE_API_KEY=your_key_here`")
        
        return False
    
    return True

def main():
    """Funcția principală"""
    try:
        # Verifică API key-ul înainte de orice
        if not check_api_key_setup():
            st.stop()
        
        # Inițializează starea sesiunii
        initialize_session_state()
        
        # Startup rapid din DB (fără ingestie automată la refresh).
        if not st.session_state.get("startup_bootstrap_done", False):
            bootstrap_documents_from_storage(os.getenv('GOOGLE_API_KEY', ''))
        
        # Configurează bara laterală
        setup_sidebar()
        
        # Interfața principală
        main_interface()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            🧠 APCI - Asistentul Personalizat de Cercetare și Învățare<br>
            Dezvoltat cu ❤️ folosind Streamlit și Google Gemini 2.5 Flash
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Eroare critică în aplicație: {str(e)}")
        logger.error(f"Eroare critică: {e}")

if __name__ == "__main__":
    main()
