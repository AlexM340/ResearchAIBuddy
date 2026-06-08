"""CerebrumAI - Asistentul Personalizat de Cercetare si Invatare.

Punct de intrare Streamlit. Orchestreaza:
  - bootstrap-ul sistemului CerebrumAI (config, API key, indexare initiala);
  - bara laterala minimalista (status + setari avansate);
  - navigarea pe 3 tab-uri principale: Second Brain / Notebooks / Graph.

Logica per vedere traieste in modulele `views/*`.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Suprima zgomotul de la transformers (scaneaza la import procesoare optionale
# de imagine/video pentru care lipsesc dependintele - torchvision etc.).
# Trebuie setate INAINTE de orice import indirect al `transformers`.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import streamlit as st

# Incarca .env din root-ul proiectului inainte de orice altceva.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CerebrumAI - Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports module locale (cu detectie de erori pentru mesaje clare)
# ---------------------------------------------------------------------------

try:
    from rag_module_flash import create_apci_system  # noqa: F401
    RAG_MODULE_AVAILABLE = True
except ImportError as exc:
    RAG_MODULE_AVAILABLE = False
    st.error(f"Modulul RAG nu este disponibil: {exc}")

try:
    from document_manager import DocumentManager
    DOCUMENT_MANAGER_AVAILABLE = True
except ImportError as exc:
    DOCUMENT_MANAGER_AVAILABLE = False
    st.error(f"Document Manager nu este disponibil: {exc}")

try:
    from chat_session_manager import ChatSessionManager
    CHAT_SESSION_MANAGER_AVAILABLE = True
except ImportError as exc:
    CHAT_SESSION_MANAGER_AVAILABLE = False
    st.error(f"Chat Session Manager nu este disponibil: {exc}")

from views import _cache
from views._documents import run_full_reindex, sync_library_to_db, update_system_status
from views._shared import (
    DEFAULT_CHAT_TITLE,
    get_chat_manager,
    get_local_chat_manager,
)
from views.graph import render_graph
from views.notebooks import render_notebooks
from views.notes import render_notes
from views.retrieval_logs import render_retrieval_logs
from views.second_brain import render_second_brain
from views.tasks import render_tasks
from views.timeline import render_timeline


# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------


def load_config() -> Dict[str, Any]:
    """Incarca configuratia din `config.json` cu fallback la valori implicite."""
    default_config = _default_config()
    config_path = Path("config.json")
    if not config_path.exists():
        return default_config
    try:
        import json
        with open(config_path, "r", encoding="utf-8") as fh:
            file_config = json.load(fh)
    except Exception as exc:
        st.error(f"Eroare la incarcarea configuratiei: {exc}")
        return default_config

    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in default_config.items()}
    for key, value in file_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _default_config() -> Dict[str, Any]:
    return {
        "models": {
            "primary_llm": "gemini-2.5-flash",
            "fallback_llm": "gemini-2.0-flash-exp",
        },
        "gemini": {
            "temperature": 0.1,
            "max_tokens": 65536,
            "rate_limits": {"requests_per_minute": 30, "requests_per_day": 3000},
        },
        "chunking": {"chunk_size": 1000, "chunk_overlap": 200, "max_context_docs": 5},
        "rag": {
            "retrieval_candidates": 20,
            "rerank_top_k": 8,
            "neighbor_window": 1,
            "hybrid_alpha": 0.7,
        },
        "storage": {"postgres_dsn": "", "pgvector_embedding_dim": 384},
        "graph": {"neo4j_uri": "", "neo4j_user": "", "neo4j_password": ""},
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
            "ingestion_max_workers": 4,
        },
        "retrieval": {
            "vector_top_k": 8,
            "graph_top_k_paths": 6,
            "hybrid_rerank_top_k": 8,
        },
    }


def build_apci_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Construieste configuratia CerebrumAI folosita la initializarea sistemului."""
    rag_cfg = config.get("rag", {})
    sb_cfg = config.get("second_brain", {})
    retrieval_cfg = config.get("retrieval", {})
    quality_cfg = config.get("retrieval_quality", {})
    web_cfg = config.get("web_search", {})
    storage_cfg = config.get("storage", {})
    graph_cfg = config.get("graph", {})

    tavily_key = os.getenv("TAVILY_API_KEY", config.get("tavily_api_key", "")).strip()

    postgres_dsn = os.getenv("POSTGRES_DSN", storage_cfg.get("postgres_dsn", "")).strip()
    neo4j_uri = os.getenv("NEO4J_URI", graph_cfg.get("neo4j_uri", "")).strip()
    neo4j_user = os.getenv("NEO4J_USER", graph_cfg.get("neo4j_user", "")).strip()
    neo4j_password = os.getenv("NEO4J_PASSWORD", graph_cfg.get("neo4j_password", "")).strip()

    return {
        "model_name": config["models"]["primary_llm"],
        "fallback_model": config["models"]["fallback_llm"],
        "temperature": config["gemini"]["temperature"],
        "max_tokens": config["gemini"]["max_tokens"],
        "max_context_docs": config["chunking"]["max_context_docs"],
        "chunk_size": config["chunking"]["chunk_size"],
        "chunk_overlap": config["chunking"]["chunk_overlap"],
        "cache_enabled": True,
        "rate_limit_rpm": config["gemini"]["rate_limits"]["requests_per_minute"],
        "rate_limit_rpd": config["gemini"]["rate_limits"]["requests_per_day"],
        "retrieval_candidates": rag_cfg.get("retrieval_candidates", 20),
        "rerank_top_k": rag_cfg.get("rerank_top_k", 8),
        "neighbor_window": rag_cfg.get("neighbor_window", 1),
        "hybrid_alpha": rag_cfg.get("hybrid_alpha", 0.7),
        "vector_top_k": retrieval_cfg.get("vector_top_k", 8),
        "graph_top_k_paths": retrieval_cfg.get("graph_top_k_paths", 6),
        "hybrid_rerank_top_k": retrieval_cfg.get("hybrid_rerank_top_k", 8),
        "enable_graph_rag": sb_cfg.get("enable_graph_rag", True),
        "router_mode": sb_cfg.get("router_mode", "rule_based"),
        "vector_confidence_threshold": sb_cfg.get("vector_confidence_threshold", 0.35),
        "context_budget_tokens": sb_cfg.get("context_budget_tokens", 8000),
        "vector_budget_ratio": sb_cfg.get("vector_budget_ratio", 0.65),
        "graph_budget_ratio": sb_cfg.get("graph_budget_ratio", 0.35),
        "decision_extraction_enabled": sb_cfg.get("decision_extraction_enabled", True),
        "graph_max_hops": sb_cfg.get("graph_max_hops", 2),
        "graph_min_confidence": sb_cfg.get("graph_min_confidence", 0.70),
        "graph_extraction_mode": sb_cfg.get("graph_extraction_mode", "heuristic_fallback"),
        "memory_consolidation_enabled": sb_cfg.get("memory_consolidation_enabled", True),
        "memory_decay_days": sb_cfg.get("memory_decay_days", 90),
        "db_primary_strict": sb_cfg.get("db_primary_strict", True),
        "ingestion_embedding_batch_size": sb_cfg.get("ingestion_embedding_batch_size", 64),
        "ingestion_max_workers": sb_cfg.get("ingestion_max_workers", 4),
        "enable_reranker": quality_cfg.get("enable_reranker", True),
        "reranker_model": quality_cfg.get("reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        "reranker_input_k": quality_cfg.get("reranker_input_k", 20),
        "enable_hyde": quality_cfg.get("enable_hyde", False),
        "hyde_max_tokens": quality_cfg.get("hyde_max_tokens", 200),
        "tavily_api_key": tavily_key,
        "enable_web_fallback": web_cfg.get("enable_web_fallback", True),
        "web_fallback_max_results": web_cfg.get("max_results", 5),
        "web_fallback_search_depth": web_cfg.get("search_depth", "advanced"),
        "postgres_dsn": postgres_dsn,
        "pgvector_embedding_dim": storage_cfg.get("pgvector_embedding_dim", 384),
        "neo4j_uri": neo4j_uri,
        "neo4j_user": neo4j_user,
        "neo4j_password": neo4j_password,
    }


# ---------------------------------------------------------------------------
# Session state + bootstrap
# ---------------------------------------------------------------------------


def initialize_session_state() -> None:
    """Initializeaza valorile implicite din `st.session_state`."""
    defaults: Dict[str, Any] = {
        "apci_system": None,
        "documents_loaded": False,
        "chat_history": [],
        "active_chat_id": "",
        "chat_title_draft": DEFAULT_CHAT_TITLE,
        "system_stats": None,
        "active_notebook": "",
        "startup_bootstrap_done": False,
        "documents_bootstrap_source": "unknown",
        "last_sync_summary": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "chat_manager" not in st.session_state and CHAT_SESSION_MANAGER_AVAILABLE:
        st.session_state.chat_manager = ChatSessionManager()
    if "document_manager" not in st.session_state and DOCUMENT_MANAGER_AVAILABLE:
        st.session_state.document_manager = DocumentManager()


def ensure_apci_system(api_key: str):
    """Initializeaza CerebrumAI o singura data si il returneaza."""
    if not RAG_MODULE_AVAILABLE:
        return None
    if st.session_state.apci_system:
        return st.session_state.apci_system
    config = load_config()
    config_dict = build_apci_config_dict(config)
    st.session_state.apci_system = create_apci_system(api_key, config_dict)
    return st.session_state.apci_system


def bootstrap_documents_from_storage(api_key: str) -> None:
    """Initializeaza sistemul si citeste statusul indexului din DB la pornire."""
    if st.session_state.get("startup_bootstrap_done"):
        return
    st.session_state.startup_bootstrap_done = True

    try:
        apci_system = ensure_apci_system(api_key)
        if not apci_system:
            st.session_state.documents_bootstrap_source = "apci_unavailable"
            return

        storage_status = apci_system.get_storage_index_status()
        indexed_chunks = int(storage_status.get("indexed_chunks", 0) or 0)
        postgres_enabled = bool(storage_status.get("postgres_enabled", False))

        st.session_state.last_sync_summary = {
            "indexed_documents": int(storage_status.get("indexed_documents", 0) or 0),
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
            st.session_state.documents_bootstrap_source = "db_empty"
        else:
            st.session_state.documents_bootstrap_source = "local_pending"

        update_system_status()
    except Exception as exc:
        logger.error("Eroare la bootstrap din storage: %s", exc)
        st.session_state.documents_bootstrap_source = "bootstrap_error"


# ---------------------------------------------------------------------------
# Migrare chat-uri locale -> Postgres (utility expus in sidebar)
# ---------------------------------------------------------------------------


def migrate_local_chats_to_db() -> Dict[str, Any]:
    """Migreaza chat-urile locale (JSON) in Postgres, daca DB-ul e disponibil."""
    result: Dict[str, Any] = {"success": False, "migrated": 0, "failed": 0, "message": ""}

    local_manager = get_local_chat_manager()
    apci_system = st.session_state.get("apci_system")
    repository = getattr(apci_system, "repository", None) if apci_system else None

    if not local_manager:
        result["message"] = "Chat manager local indisponibil."
        return result
    if not repository or not getattr(repository, "enabled", False):
        result["message"] = "Storage Postgres nu este activ."
        return result
    try:
        if not repository.is_healthy():
            result["message"] = "Conexiunea la Postgres nu este disponibila."
            return result
    except Exception as exc:
        result["message"] = f"Conexiunea la Postgres a esuat: {exc}"
        return result

    try:
        candidates = local_manager.list_sessions(include_migrated=False)
        if not candidates:
            result["success"] = True
            result["message"] = "Nu exista chat-uri locale de migrat."
            return result

        failed_ids: List[str] = []
        for summary in candidates:
            session_id = summary.get("id", "")
            if not session_id:
                continue
            data = local_manager.load_session(session_id)
            if not data:
                failed_ids.append(session_id)
                continue
            created = repository.create_session(
                title=data.get("title", DEFAULT_CHAT_TITLE),
                topic_collection=data.get("topic_collection", ""),
                query_mode=data.get("query_mode", "topic_general"),
            )
            db_id = created.get("id") if created else ""
            if not db_id:
                failed_ids.append(session_id)
                continue

            ok = True
            for exchange in data.get("messages", []):
                if not repository.append_exchange(db_id, exchange):
                    ok = False
                    break
            if ok:
                local_manager.mark_session_migrated(session_id, db_session_id=db_id)
                result["migrated"] += 1
            else:
                repository.delete_session(db_id)
                failed_ids.append(session_id)

        result["failed"] = len(failed_ids)
        result["success"] = result["failed"] == 0
        result["message"] = (
            f"Migrare completa: {result['migrated']} chat-uri."
            if result["success"]
            else f"Migrare partiala: {result['migrated']} ok, {result['failed']} esuate."
        )
        return result
    except Exception as exc:
        logger.error("Eroare la migrarea chat-urilor: %s", exc)
        result["message"] = f"Eroare: {exc}"
        return result


# ---------------------------------------------------------------------------
# API key onboarding
# ---------------------------------------------------------------------------


def check_api_key_setup() -> bool:
    """Verifica si configureaza API key-ul. Returneaza True daca este setat."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        return True

    st.title("Bun venit in CerebrumAI")
    st.markdown(
        "Ai nevoie de un **API key Google Gemini** pentru a porni asistentul. "
        "Mergi la [Google AI Studio](https://makersuite.google.com/app/apikey) si genereaza unul."
    )

    api_key_input = st.text_input("API key Gemini", type="password", placeholder="AIzaSy...")
    if st.button("Salveaza", type="primary"):
        if not api_key_input:
            st.warning("Introdu mai intai key-ul.")
            return False
        try:
            env_path = Path(__file__).parent.parent / ".env"
            existing: List[str] = []
            if env_path.exists():
                with open(env_path, "r") as fh:
                    existing = [line for line in fh.readlines() if not line.startswith("GOOGLE_API_KEY")]
            existing.insert(0, f"GOOGLE_API_KEY={api_key_input}\n")
            with open(env_path, "w") as fh:
                fh.writelines(existing)
            os.environ["GOOGLE_API_KEY"] = api_key_input
            st.success("API key salvat. Reincarca pagina.")
        except Exception as exc:
            st.error(f"Eroare la salvare: {exc}")
        return False
    return False


# ---------------------------------------------------------------------------
# Sidebar slim
# ---------------------------------------------------------------------------


def render_sidebar_brand() -> None:
    """Antetul barei laterale (sus de tot, inainte de continutul paginii)."""
    with st.sidebar:
        st.markdown("### 🧠 CerebrumAI")
        st.caption("Asistent personalizat de cercetare si invatare")


def render_sidebar_global() -> None:
    """Utilitare globale + status + setari, sub sectiunea specifica paginii.

    Se randeaza DUPA pagina, ca sectiunile paginii (ex. Chat/Unelte din Second
    Brain) sa apara sus, iar plumbing-ul (status, setari, intretinere) jos.
    """
    with st.sidebar:
        st.divider()
        _render_quick_actions()
        st.divider()
        _render_status_pill()
        with st.expander("⚙️ Setari", expanded=False):
            _render_settings_section()
        with st.expander("🔧 Zona tehnica", expanded=False):
            st.caption("Operatii de intretinere — pot dura sau rescrie indexul. Foloseste cu grija.")
            _render_maintenance_section()


def _render_status_pill() -> None:
    """Status condensat: o pastila de sanatate + avertismente; detalii la cerere."""
    summary = st.session_state.get("last_sync_summary", {}) or {}
    apci_system = st.session_state.get("apci_system")

    postgres_ok = False
    neo4j_ok = False
    web_search_ok = False
    storage_status: Dict[str, Any] = {}
    if apci_system:
        try:
            # Cached: fans out to 3 DB queries; partajat cu restul sidebar-ului.
            storage_status = _cache.storage_index_status(apci_system) or {}
            postgres_ok = bool(storage_status.get("postgres_enabled", False))
        except Exception:
            storage_status = {}
        neo4j_client = getattr(apci_system, "neo4j_client", None)
        neo4j_ok = bool(neo4j_client and getattr(neo4j_client, "enabled", False))
        try:
            web_search_ok = bool(getattr(apci_system, "web_search_available", False))
        except Exception:
            web_search_ok = False

    migrations = (storage_status.get("migrations") or {}) if storage_status else {}
    pending = migrations.get("pending") or []

    # Pastila de sanatate (o singura linie).
    if postgres_ok:
        st.caption("🟢 Conectat")
    else:
        st.caption("🟡 Mod local (fara Postgres)")

    # Avertismentele actionabile raman vizibile.
    if not postgres_ok:
        st.warning("Mod limitat: Second Brain complet necesita Postgres + pgvector.")
    if pending:
        st.warning(f"Migratii DB pending: {len(pending)} — vezi 🔧 Zona tehnica.")

    # Detaliile tehnice, la cerere.
    with st.expander("Detalii sistem", expanded=False):
        st.caption(f"Postgres + pgvector: {'conectat' if postgres_ok else 'indisponibil (local fallback)'}")
        st.caption(f"Neo4j (graph): {'conectat' if neo4j_ok else 'indisponibil'}")
        st.caption(f"Tavily (web search): {'conectat' if web_search_ok else 'indisponibil (TAVILY_API_KEY lipsa)'}")
        counts = (storage_status.get("memory_artifact_counts") or {}) if storage_status else {}
        if counts:
            st.caption(f"Memory artifacts: {sum(int(v or 0) for v in counts.values())}")
        indexed_docs = int(summary.get("indexed_documents", 0) or 0)
        indexed_chunks = int(summary.get("indexed_chunks", 0) or 0)
        st.caption(f"Documente indexate: {indexed_docs} ({indexed_chunks} chunks)")
        last_sync = summary.get("last_sync_at")
        if last_sync:
            st.caption(f"Ultimul sync: {last_sync}")


def _render_quick_actions() -> None:
    document_manager = st.session_state.get("document_manager")
    if not document_manager:
        return

    if st.button("+ Adauga documente", use_container_width=True, type="primary"):
        nav_pages = st.session_state.get("_nav_pages_by_url") or {}
        notebooks_page = nav_pages.get("notebooks")
        if notebooks_page is not None:
            st.switch_page(notebooks_page)

    _render_quick_note_capture()

    lib_stats = document_manager.get_library_stats()
    if lib_stats.get("total_documents", 0) > 0:
        if st.button("Sync biblioteca → DB", use_container_width=True):
            with st.spinner("Sincronizez..."):
                result = sync_library_to_db()
            if result.get("success"):
                st.success(result.get("message", "Sync ok."))
            else:
                st.error(result.get("message", "Sync esuat."))
            st.rerun()


def _render_quick_note_capture() -> None:
    """Quick-capture widget: prinde idei oricand, fara sa parasesti tab-ul curent.

    Mereu accesibil in sidebar — esenta primitivei de Second Brain.
    """
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return
    repo = getattr(apci_system, "repository", None)
    if not repo or not getattr(repo, "enabled", False):
        return

    with st.expander("💡 Captureaza o idee", expanded=False):
        content = st.text_area(
            "Idee",
            placeholder="Scrie rapid o idee, observatie, intrebare...",
            height=100,
            key="sidebar_note_content",
            label_visibility="collapsed",
        )
        col_save, col_open = st.columns([1, 1])
        with col_save:
            if st.button(
                "Salveaza",
                key="sidebar_note_save",
                type="primary",
                use_container_width=True,
                disabled=not (content and content.strip()),
            ):
                note_id = apci_system.create_note(content=content.strip())
                if note_id:
                    st.success(f"Nota #{note_id} salvata.")
                    st.session_state.sidebar_note_content = ""
                    st.rerun()
                else:
                    st.error("Salvare esuata.")
        with col_open:
            if st.button("Vezi notele", key="sidebar_open_notes", use_container_width=True):
                nav_pages = st.session_state.get("_nav_pages_by_url") or {}
                notes_page = nav_pages.get("notes")
                if notes_page is not None:
                    st.switch_page(notes_page)


def _render_settings_section() -> None:
    """Setari user-facing: web search implicit, cache, cheia API."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        st.caption("Sistemul nu este initializat.")
        return

    # Web fallback implicit (diferit de 'Forteaza web' din chat).
    st.markdown("**Web search (Tavily)**")
    current = bool(getattr(apci_system.config, "enable_web_fallback", True))
    new_value = st.toggle(
        "Auto web search cand lipseste context intern",
        value=current,
        key="sidebar_web_fallback_toggle",
        help="Comportamentul implicit. 'Forteaza web' din chat tine doar de urmatoarea intrebare.",
    )
    if new_value != current:
        try:
            apci_system.config.enable_web_fallback = bool(new_value)
            st.success("Setare actualizata.")
        except Exception:
            st.error("Nu am putut actualiza setarea.")

    # Cache embeddings.
    try:
        cache_info = apci_system.get_cache_info() or {}
        embeddings_cache = cache_info.get("embeddings_cache") or {}
        if embeddings_cache:
            st.divider()
            st.caption(
                f"Cache embeddings: {embeddings_cache.get('cached_files', 0)} fisiere "
                f"({embeddings_cache.get('cache_size_mb', 0)} MB)"
            )
            if st.button("Curata cache embeddings", use_container_width=True):
                apci_system.clear_embeddings_cache()
                st.success("Cache curatat.")
                st.rerun()
    except Exception:
        pass

    # API key (read-only display).
    current_key = os.getenv("GOOGLE_API_KEY", "")
    if current_key:
        masked = current_key[:6] + "…" + current_key[-4:] if len(current_key) > 12 else "configurat"
        st.divider()
        st.caption(f"API Key: `{masked}`")


def _render_maintenance_section() -> None:
    """Operatii de intretinere/admin (potential lente sau destructive)."""
    # Migrare chat-uri locale -> Postgres
    chat_manager = get_chat_manager()
    local_manager = get_local_chat_manager()
    if chat_manager and hasattr(chat_manager, "is_healthy") and local_manager:
        pending = local_manager.list_sessions(include_migrated=False)
        if pending:
            st.caption(f"{len(pending)} chat-uri locale nemigrate.")
            if st.button("Migreaza in Postgres", use_container_width=True):
                with st.spinner("Migrez..."):
                    res = migrate_local_chats_to_db()
                if res.get("success"):
                    st.success(res.get("message", "Migrare ok."))
                else:
                    st.warning(res.get("message", "Migrare partiala."))
                st.rerun()
            st.divider()

    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return

    st.markdown("**Memorie canonica**")
    try:
        # Cached storage status (evita 3 query-uri DB la fiecare rerun).
        storage_status = _cache.storage_index_status(apci_system) or {}
        migrations = storage_status.get("migrations", {}) or {}
        pending_migrations = migrations.get("pending") or []
        if pending_migrations:
            st.warning("Migratii pending: " + ", ".join(pending_migrations))
            if st.button("Aplica migratiile DB", use_container_width=True, key="sidebar_run_migrations"):
                repo = getattr(apci_system, "repository", None)
                ok = bool(repo and hasattr(repo.client, "run_migrations") and repo.client.run_migrations())
                if ok:
                    st.success("Migratii aplicate.")
                else:
                    st.error("Migratiile nu au putut fi aplicate.")
                st.rerun()
        else:
            st.caption("Schema DB: la zi.")
    except Exception:
        pass

    if st.button("Backfill memory artifacts", use_container_width=True, key="sidebar_backfill_memory"):
        with st.spinner("Construiesc memory_artifacts pentru date existente..."):
            result = apci_system.backfill_memory_artifacts(limit=1000)
        if result.get("success"):
            st.success(f"Backfill: {result.get('created', 0)} artifact-uri create.")
        else:
            st.warning(f"Backfill partial: {result}")
        st.rerun()

    if st.button("Reconstruieste Neo4j din Postgres", use_container_width=True, key="sidebar_rebuild_graph"):
        with st.spinner("Reconstruiesc proiectia Neo4j..."):
            result = apci_system.rebuild_graph_projection()
        if result.get("error"):
            st.warning(result["error"])
        else:
            st.success(
                f"Graph rebuild: {result.get('chunks', 0)} chunks, "
                f"{result.get('artifacts', 0)} artifacts, {result.get('relations', 0)} relatii."
            )
        st.rerun()

    st.divider()
    st.caption("Reindexare completa: sterge toti vectorii si ii reconstruieste cu modelul curent.")
    if st.button("Reindexeaza toate documentele", use_container_width=True, key="sidebar_reindex_btn"):
        with st.spinner("Reindexez... poate dura cateva minute."):
            result = run_full_reindex()
        if result.get("success"):
            st.success(result.get("message", "Reindexare completa."))
        else:
            st.error(result.get("message", "Reindexare esuata."))
        st.rerun()


# ---------------------------------------------------------------------------
# Top navigation + routing
# ---------------------------------------------------------------------------


def _render_reindex_banner() -> None:
    """Afiseaza avertisment global cand embedding model s-a schimbat."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system or not hasattr(apci_system, "is_reindex_recommended"):
        return
    try:
        if not apci_system.is_reindex_recommended():
            return
    except Exception:
        return

    with st.container(border=True):
        st.warning(
            "Modelul de embeddings a fost schimbat fata de ultima indexare. "
            "Vectorii existenti sunt incompatibili cu noul model. "
            "Reindexeaza pentru retrieval optim (inclusiv cross-lingual RO↔EN)."
        )
        col_info, col_action = st.columns([3, 1])
        with col_info:
            st.caption(
                "Reindexarea sterge chunks + embeddings existente si le reconstruieste "
                "cu modelul curent. Documentele din biblioteca raman intacte."
            )
        with col_action:
            if st.button("Reindexeaza acum", type="primary", use_container_width=True, key="reindex_banner_btn"):
                with st.spinner("Reindexez toate documentele... poate dura cateva minute."):
                    result = run_full_reindex()
                if result.get("success"):
                    st.success(result.get("message", "Reindexare completa."))
                else:
                    st.error(result.get("message", "Reindexare esuata."))
                st.rerun()


def main_interface() -> None:
    """Top nav nativ via st.navigation(position='top').

    Streamlit randeaza nav-ul intr-un container DOM dedicat (sora cu stMain),
    independent de stacking context-urile din block-container. Ramane vizibil
    indiferent cat scrollezi — la fel cum sta jos chat_input.

    Ofera URL routing automat (fiecare view are url_path propriu) si suport
    nativ pentru iconite. Navigarea programatica (din alte view-uri) se face
    cu helper-ul `navigate_to(url_path)` din views._shared.
    """
    pages = [
        st.Page(render_second_brain, title="Second Brain", icon="🧠",
                url_path="second-brain", default=True),
        st.Page(render_notebooks, title="Notebooks", icon="📚",
                url_path="notebooks"),
        st.Page(render_notes, title="Notes", icon="💡",
                url_path="notes"),
        st.Page(render_tasks, title="Tasks", icon="✅",
                url_path="tasks"),
        st.Page(render_timeline, title="Timeline", icon="📅",
                url_path="timeline"),
        st.Page(render_graph, title="Knowledge Graph", icon="🕸️",
                url_path="graph"),
        st.Page(render_retrieval_logs, title="Logs", icon="📈",
                url_path="logs"),
    ]

    # expune pages pentru navigare programatica (st.switch_page) din alte view-uri
    st.session_state["_nav_pages_by_url"] = {p.url_path: p for p in pages}

    pg = st.navigation(pages, position="top")
    _render_reindex_banner()
    pg.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        if not check_api_key_setup():
            st.stop()

        initialize_session_state()

        if not st.session_state.get("startup_bootstrap_done"):
            bootstrap_documents_from_storage(os.getenv("GOOGLE_API_KEY", ""))

        # Sidebar: brand sus -> sectiunea paginii (ex. Chat/Unelte) -> utilitare globale jos.
        render_sidebar_brand()
        main_interface()
        render_sidebar_global()
    except Exception as exc:
        st.error(f"Eroare critica in aplicatie: {exc}")
        logger.exception("Eroare critica: %s", exc)


if __name__ == "__main__":
    main()
