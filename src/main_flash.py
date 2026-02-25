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
            for key in ['models', 'gemini', 'chunking', 'rag', 'ui']:
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
        'hybrid_alpha': rag_config.get("hybrid_alpha", 0.7)
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
    
    # Auto-loading activat implicit pentru experiență production-ready
    if 'auto_load_enabled' not in st.session_state:
        st.session_state.auto_load_enabled = True
    
    if 'auto_load_attempted' not in st.session_state:
        st.session_state.auto_load_attempted = False

    if 'query_mode' not in st.session_state:
        st.session_state.query_mode = "topic_general"

    if 'active_topic_collection' not in st.session_state:
        st.session_state.active_topic_collection = ""

    if 'upload_collection' not in st.session_state:
        st.session_state.upload_collection = GENERAL_COLLECTION_NAME

def get_chat_manager() -> Optional["ChatSessionManager"]:
    """Return chat session manager if available."""
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

def auto_load_library_documents():
    """
    Încarcă automat toate documentele din bibliotecă la pornirea aplicației
    pentru o experiență production-ready și intuitivă
    """
    if not DOCUMENT_MANAGER_AVAILABLE or not st.session_state.document_manager:
        return False
    
    # Verifică dacă auto-loading este activat și nu a fost deja încercat
    if not st.session_state.get('auto_load_enabled', True):
        return False
        
    if st.session_state.get('auto_load_attempted', False):
        return False
    
    # Verifică dacă există API key
    api_key = os.getenv('GOOGLE_API_KEY', '')
    if not api_key:
        return False
    
    # Verifică dacă există documente în bibliotecă
    stats = st.session_state.document_manager.get_library_stats()
    if stats.get('total_documents', 0) == 0:
        st.session_state.auto_load_attempted = True
        return False
    
    # Marcăm că am încercat auto-loading-ul
    st.session_state.auto_load_attempted = True
    
    try:
        # Obține toate documentele din bibliotecă
        all_docs = st.session_state.document_manager.get_all_documents()
        doc_ids = list(all_docs.keys())
        
        if not doc_ids:
            return False
        
        with st.spinner("Auto-loading: Încarcă bibliotecă completă pentru experiență optimă..."):
            # Inițializează sistemul APCI dacă nu există
            if not st.session_state.apci_system:
                config = load_config()
                config_dict = build_apci_config_dict(config)
                st.session_state.apci_system = create_apci_system(api_key, config_dict)
            
            # Obține căile fișierelor din bibliotecă
            file_paths = []
            loaded_docs = []
            metadata_by_path = {}
            
            for doc_id in doc_ids:
                file_path = st.session_state.document_manager.get_document_file_path(doc_id)
                doc_info = st.session_state.document_manager.get_document_info(doc_id)
                
                if file_path and file_path.exists() and doc_info:
                    resolved_path = str(file_path.resolve())
                    file_paths.append(resolved_path)
                    loaded_docs.append(doc_info['original_name'])
                    metadata_by_path[resolved_path] = build_metadata_entry(doc_id, doc_info, file_path)
            
            if file_paths:
                # Procesează documentele
                success = st.session_state.apci_system.load_documents(
                    file_paths,
                    metadata_by_path=metadata_by_path
                )
                
                if success:
                    st.session_state.documents_loaded = True
                    
                    # Marchează documentele ca indexate în bibliotecă
                    for doc_id in doc_ids:
                        st.session_state.document_manager.mark_document_indexed(doc_id)
                    
                    # Actualizează statusul
                    update_system_status()
                    
                    # Afișează notificare de succes
                    st.success(f"Auto-loading complet! {len(file_paths)} documente din bibliotecă sunt acum disponibile pentru chat.")
                    return True
                else:
                    st.warning("Auto-loading parțial: Unele documente nu au putut fi procesate")
                    return False
            else:
                st.info("Biblioteca este goală - documentele vor fi încărcate automat când vor fi adăugate")
                return False
                
    except Exception as e:
        st.error(f"Eroare la auto-loading: {e}")
        return False

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
            
            # Indicator stare auto-loading
            if st.session_state.get('auto_load_enabled', True):
                if st.session_state.get('documents_loaded', False):
                    st.success("Auto-loading: Biblioteca încărcată!")
                elif stats.get("total_documents", 0) > 0:
                    st.info("Auto-loading: Gata să încarce...")
                else:
                    st.info("Auto-loading: Bibliotecă goală")
            else:
                st.warning("Auto-loading dezactivat")
            
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
            
            # Auto-loading Settings
            st.subheader("Auto-Loading Bibliotecă")
            
            auto_load_enabled = st.checkbox(
                "Încărcare automată la pornire",
                value=st.session_state.get('auto_load_enabled', True),
                help="Încarcă automat toate documentele din bibliotecă la pornirea aplicației pentru o experiență production-ready"
            )
            
            if auto_load_enabled != st.session_state.get('auto_load_enabled', True):
                st.session_state.auto_load_enabled = auto_load_enabled
                st.session_state.auto_load_attempted = False  # Reset pentru a permite re-loading
            
            if not auto_load_enabled and st.session_state.get('documents_loaded', False):
                st.info("Auto-loading dezactivat. Documentele rămân încărcate în sesiunea curentă.")
            
            # Manual trigger pentru auto-loading
            if not st.session_state.get('documents_loaded', False) and api_key:
                if st.button("Forțează Auto-Loading Acum", type="secondary"):
                    st.session_state.auto_load_attempted = False
                    auto_load_library_documents()
                    st.rerun()
            
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
        # Inițializează sistemul APCI doar dacă nu există
        if not st.session_state.apci_system:
            with st.spinner("🔄 Inițializez sistemul APCI..."):
                # Configurație pentru sistem - folosește configurația globală
                config = load_config()
                config_dict = build_apci_config_dict(config)
                
                # Creează sistemul APCI
                st.session_state.apci_system = create_apci_system(api_key, config_dict)
        else:
            st.info("📚 Adaug documente noi la biblioteca existentă...")
        
        # Salvează fișierele temporar
        temp_dir = Path("./data/uploaded_docs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        doc_ids = []
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
                        doc_ids.append(doc_id)
                        doc_info = st.session_state.document_manager.get_document_info(doc_id)
                        library_file_path = st.session_state.document_manager.get_document_file_path(doc_id)

                        if library_file_path and doc_info:
                            resolved_path = str(library_file_path.resolve())
                            if resolved_path not in unique_paths:
                                unique_paths.add(resolved_path)
                                file_paths.append(resolved_path)
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
        
        # Procesează documentele
        with st.spinner("🧠 Procesez și indexez documentele..."):
            success = st.session_state.apci_system.load_documents(
                file_paths,
                metadata_by_path=metadata_by_path
            )
            
            if success:
                st.session_state.documents_loaded = True
                st.success(f"✅ {len(file_paths)} documente procesate cu succes!")
                
                # Marchează documentele ca indexate în bibliotecă
                if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
                    for doc_id in doc_ids:
                        st.session_state.document_manager.mark_document_indexed(doc_id)
                
                # Actualizează statusul
                update_system_status()
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
        # Inițializează sistemul APCI doar dacă nu există
        if not st.session_state.apci_system:
            with st.spinner("🔄 Inițializez sistemul APCI..."):
                # Configurație pentru sistem - folosește configurația globală
                config = load_config()
                config_dict = build_apci_config_dict(config)
                
                # Creează sistemul APCI
                st.session_state.apci_system = create_apci_system(api_key, config_dict)
        else:
            st.info("📚 Adaug documente din bibliotecă la sistema curentă...")
        
        # Obține căile fișierelor din bibliotecă
        file_paths = []
        doc_names = []
        metadata_by_path = {}
        unique_paths = set()
        
        with st.spinner("📚 Încarcă din bibliotecă..."):
            for doc_id in doc_ids:
                file_path = st.session_state.document_manager.get_document_file_path(doc_id)
                doc_info = st.session_state.document_manager.get_document_info(doc_id)
                
                if file_path and file_path.exists() and doc_info:
                    resolved_path = str(file_path.resolve())
                    if resolved_path not in unique_paths:
                        unique_paths.add(resolved_path)
                        file_paths.append(resolved_path)
                    doc_names.append(doc_info['original_name'])
                    metadata_by_path[resolved_path] = build_metadata_entry(doc_id, doc_info, file_path)
                    st.write(f"✅ Găsit: {doc_info['original_name']}")
                else:
                    st.warning(f"⚠️ Documentul {doc_id} nu a fost găsit")
        
        if not file_paths:
            st.error("❌ Nu s-au găsit documente valide")
            return
        
        # Procesează documentele
        with st.spinner("🧠 Procesez și indexez documentele..."):
            success = st.session_state.apci_system.load_documents(
                file_paths,
                metadata_by_path=metadata_by_path
            )
            
            if success:
                st.session_state.documents_loaded = True
                st.success(f"✅ {len(file_paths)} documente încărcate din bibliotecă!")
                
                # Marchează documentele ca indexate în bibliotecă
                for doc_id in doc_ids:
                    st.session_state.document_manager.mark_document_indexed(doc_id)
                
                # Actualizează statusul
                update_system_status()
                st.rerun()
            else:
                st.error("❌ Eroare la procesarea documentelor")
    
    except Exception as e:
        st.error(f"❌ Eroare: {str(e)}")
        logger.error(f"Eroare la încărcarea din bibliotecă: {e}")

def update_system_status():
    """Actualizează statusul sistemului"""
    if st.session_state.apci_system:
        try:
            st.session_state.system_stats = st.session_state.apci_system.get_system_status()
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

def main_chat_interface():
    """Interfața principală de chat"""
    # Verifică dacă sistemul este gata
    if not st.session_state.apci_system:
        st.title("🧠 APCI - Asistentul Personalizat de Cercetare și Învățare")
        
        # Verifică dacă există documente în bibliotecă
        if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
            stats = st.session_state.document_manager.get_library_stats()
            if stats.get('total_documents', 0) > 0:
                st.info("🚀 Configurează API key-ul în bara laterală. Documentele din bibliotecă vor fi încărcate automat!")
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
                        st.info("Configurează API key-ul mai întâi. Documentele din bibliotecă vor fi încărcate automat!")
                    else:
                        st.info("Încarcă documente mai întâi pentru a primi răspunsuri personalizate!")
                else:
                    st.info("Încarcă documente mai întâi pentru a primi răspunsuri personalizate!")
        
        return
    
    if not st.session_state.documents_loaded:
        st.title("🧠 APCI - Asistentul Personalizat de Cercetare și Învățare")
        
        # Verifică dacă auto-loading este activat
        if st.session_state.get('auto_load_enabled', True):
            if DOCUMENT_MANAGER_AVAILABLE and st.session_state.document_manager:
                stats = st.session_state.document_manager.get_library_stats()
                if stats.get('total_documents', 0) > 0:
                    st.info("🔄 Auto-loading în progres... Documentele din bibliotecă vor fi disponibile în curând!")
                else:
                    st.warning("📚 Biblioteca este goală. Încarcă documente pentru a începe conversația!")
            else:
                st.warning("📚 Încarcă documente pentru a începe conversația!")
        else:
            st.warning("📚 Auto-loading dezactivat. Încarcă manual documente pentru a începe conversația!")
        
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
                st.caption(f"⏱️ {response_time:.2f}s {cached}")

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
                general_collection=GENERAL_COLLECTION_NAME
            )
            
            end_time = time.time()
            
            # Adaugă în istoric
            chat_entry = {
                "question": question,
                "response": result.get("response", "Nu am putut genera un răspuns."),
                "sources": result.get("sources", []),
                "external_sources": result.get("external_sources", []),
                "response_time": result.get("response_time", end_time - start_time),
                "cached": result.get("cached", False),
                "model_used": result.get("model_used", "Unknown"),
                "answer_origin": result.get("answer_origin", "internal"),
                "retrieval_mode": result.get("retrieval_mode", st.session_state.get("query_mode", "topic_general"))
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
        
        # Auto-loading pentru experiență production-ready
        # Încarcă automat toate documentele din bibliotecă
        if (st.session_state.get('auto_load_enabled', True) and 
            not st.session_state.get('auto_load_attempted', False) and
            not st.session_state.get('documents_loaded', False)):
            auto_load_library_documents()
        
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
