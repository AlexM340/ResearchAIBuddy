"""Helper-uri partajate intre vederile CerebrumAI.

Tot ce este folosit de mai multe view-uri (constante, session helpers,
process_question, render-ul de mesaje) traieste aici pentru a evita
import-uri circulare intre `main_flash.py` si modulele `views/*`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constante UI
# ---------------------------------------------------------------------------

GENERAL_COLLECTION_NAME = "general"
DEFAULT_CHAT_TITLE = "Chat nou"

QUERY_MODE_LABELS = {
    "topic": "Doar Notebook",
    "topic_general": "Notebook + Memorie globala",
    "all": "Toate Sursele",
}

VIEW_SECOND_BRAIN = "second_brain"
VIEW_NOTEBOOKS = "notebooks"
VIEW_GRAPH = "graph"
VIEW_TASKS = "tasks"
VIEW_LOGS = "logs"


def navigate_to(url_path: str) -> None:
    """Programmatic navigation between views via st.switch_page.

    Pages dict is exposed by `main_interface()` in session_state under
    `_nav_pages_by_url` (keyed by st.Page url_path). Falls back to no-op
    if the dict isn't populated (e.g. during early bootstrap).
    """
    nav_pages = st.session_state.get("_nav_pages_by_url") or {}
    page = nav_pages.get(url_path)
    if page is not None:
        st.switch_page(page)

# ---------------------------------------------------------------------------
# Helper-uri pentru collection / metadata
# ---------------------------------------------------------------------------


def normalize_collection_name(name: str) -> str:
    """Normalizeaza numele colectiei si aplica fallback pentru general."""
    cleaned = (name or "").strip()
    return cleaned if cleaned else GENERAL_COLLECTION_NAME


def is_general_collection(name: str) -> bool:
    """Verifica daca o colectie este colectie generala."""
    normalized = normalize_collection_name(name).lower()
    return normalized in {GENERAL_COLLECTION_NAME, "default"}


def build_metadata_entry(doc_id: str, doc_info: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    """Construieste metadata standardizata pentru indexare si filtrare la query."""
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
        "source_path": resolved_path,
    }


# ---------------------------------------------------------------------------
# Chat manager helpers
# ---------------------------------------------------------------------------


def get_chat_manager():
    """Returneaza chat manager-ul DB cand storage e healthy, altfel cel local."""
    apci_system = st.session_state.get("apci_system")
    repository = getattr(apci_system, "repository", None) if apci_system else None
    if repository and getattr(repository, "enabled", False):
        try:
            if repository.is_healthy():
                return repository
        except Exception:
            pass

    return st.session_state.get("chat_manager")


def get_local_chat_manager():
    """Returneaza chat manager-ul local file-based daca exista."""
    return st.session_state.get("chat_manager")


def format_chat_label(session_info: Dict[str, Any]) -> str:
    """Construieste un label compact pentru selectorul de chat-uri."""
    title = session_info.get("title", DEFAULT_CHAT_TITLE)
    topic = session_info.get("topic_collection", "")
    count = session_info.get("message_count", 0)
    topic_label = topic if topic else "global"
    return f"{title} [{topic_label}] ({count})"


# ---------------------------------------------------------------------------
# Chat session lifecycle (scoped pe view)
# ---------------------------------------------------------------------------


def _list_sessions_for_scope(manager, topic_collection: Optional[str]) -> List[Dict[str, Any]]:
    """Filtreaza sesiunile de chat in functie de scope-ul activ.

    - topic_collection == "" sau None  -> Second Brain (chat-uri globale)
    - topic_collection == "Nume"        -> Notebook (chat-uri pe acel notebook)
    """
    sessions = manager.list_sessions()
    target = (topic_collection or "").strip()
    if target:
        return [s for s in sessions if (s.get("topic_collection") or "").strip() == target]
    return [s for s in sessions if not (s.get("topic_collection") or "").strip()]


def ensure_active_chat_for_scope(
    topic_collection: Optional[str],
    default_query_mode: str,
) -> Optional[str]:
    """Garanteaza un chat activ pentru scope-ul curent (Second Brain sau Notebook).

    Returneaza id-ul chat-ului activ. Daca nu exista nici unul in scope, creeaza unul.
    """
    manager = get_chat_manager()
    if not manager:
        return None

    scope_sessions = _list_sessions_for_scope(manager, topic_collection)
    active_chat_id = st.session_state.get("active_chat_id", "")
    scope_ids = {s["id"] for s in scope_sessions}

    if active_chat_id in scope_ids:
        # chat-ul curent e deja in scope, doar reincarca starea
        active_session = manager.load_session(active_chat_id) or {}
        st.session_state.chat_history = active_session.get("messages", [])
        st.session_state.chat_title_draft = active_session.get("title", DEFAULT_CHAT_TITLE)
        return active_chat_id

    if scope_sessions:
        latest = scope_sessions[0]
        st.session_state.active_chat_id = latest["id"]
        loaded = manager.load_session(latest["id"]) or {}
        st.session_state.chat_history = loaded.get("messages", [])
        st.session_state.chat_title_draft = loaded.get("title", DEFAULT_CHAT_TITLE)
        return latest["id"]

    # nici un chat in scope -> creeaza unul nou
    created = manager.create_session(
        title=DEFAULT_CHAT_TITLE,
        topic_collection=topic_collection or "",
        query_mode=default_query_mode,
    )
    if not created or not created.get("id"):
        return None
    st.session_state.active_chat_id = created["id"]
    st.session_state.chat_history = []
    st.session_state.chat_title_draft = created.get("title", DEFAULT_CHAT_TITLE)
    return created["id"]


def switch_active_chat(session_id: str) -> None:
    """Incarca un chat selectat in session state."""
    manager = get_chat_manager()
    if not manager or not session_id:
        return

    session_data = manager.load_session(session_id)
    if not session_data:
        return

    st.session_state.active_chat_id = session_id
    st.session_state.chat_history = session_data.get("messages", [])
    st.session_state.chat_title_draft = session_data.get("title", DEFAULT_CHAT_TITLE)


def create_chat_in_scope(
    topic_collection: Optional[str],
    query_mode: str,
    title: str = DEFAULT_CHAT_TITLE,
) -> Optional[str]:
    """Creeaza un chat nou si il face activ in scope-ul curent."""
    manager = get_chat_manager()
    if not manager:
        return None
    created = manager.create_session(
        title=title,
        topic_collection=topic_collection or "",
        query_mode=query_mode,
    )
    if not created or not created.get("id"):
        return None
    switch_active_chat(created["id"])
    return created["id"]


def delete_active_chat(topic_collection: Optional[str], default_query_mode: str) -> None:
    """Sterge chat-ul activ si pune unul valid in loc."""
    manager = get_chat_manager()
    if not manager:
        return
    active_chat_id = st.session_state.get("active_chat_id", "")
    if not active_chat_id:
        return
    manager.delete_session(active_chat_id)
    st.session_state.active_chat_id = ""
    ensure_active_chat_for_scope(topic_collection, default_query_mode)


def clear_active_chat() -> None:
    """Curata mesajele chat-ului activ, fara a-l sterge."""
    manager = get_chat_manager()
    active_chat_id = st.session_state.get("active_chat_id", "")
    st.session_state.chat_history = []
    if manager and active_chat_id:
        manager.clear_session(active_chat_id)


def rename_active_chat(new_title: str) -> bool:
    """Redenumeste chat-ul activ."""
    manager = get_chat_manager()
    active_chat_id = st.session_state.get("active_chat_id", "")
    if not manager or not active_chat_id or not new_title.strip():
        return False
    manager.rename_session(active_chat_id, new_title.strip())
    st.session_state.chat_title_draft = new_title.strip()
    return True


# ---------------------------------------------------------------------------
# Procesare intrebare + persistenta
# ---------------------------------------------------------------------------


def process_question(
    question: str,
    *,
    retrieval_mode: str,
    active_collection: Optional[str],
    force_web: bool = False,
) -> None:
    """Trimite intrebarea catre CerebrumAI si actualizeaza chat-ul activ.

    force_web=True ocoleste retrievalul intern si interogheaza direct Tavily.
    """
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        st.error("Sistemul CerebrumAI nu este initializat.")
        return

    spinner_msg = "Caut pe web..." if force_web else "Caut in baza ta de cunostinte..."

    try:
        with st.spinner(spinner_msg):
            start_time = time.time()
            result = apci_system.query(
                question,
                retrieval_mode=retrieval_mode,
                active_collection=active_collection or None,
                general_collection=GENERAL_COLLECTION_NAME,
                include_memory=True,
                force_web=force_web,
            )
            end_time = time.time()

        chat_entry = {
            "question": question,
            "response": result.get("response", "Nu am putut genera un raspuns."),
            "sources": result.get("sources", []),
            "graph_sources": result.get("graph_sources", []),
            "memory_hits": result.get("memory_hits", []),
            "note_sources": result.get("note_sources", []),
            "tasks": result.get("tasks", []),
            "provenance": result.get("provenance", []),
            "external_sources": result.get("external_sources", []),
            "response_time": result.get("response_time", end_time - start_time),
            "cached": result.get("cached", False),
            "model_used": result.get("model_used", "Unknown"),
            "answer_origin": result.get("answer_origin", "internal"),
            "retrieval_mode": result.get("retrieval_mode", retrieval_mode),
            "route_used": result.get("route_used", "vector"),
            "router_reason": result.get("router_reason", ""),
            "auto_captured": result.get("auto_captured", {}),
            "proposed_artifacts": result.get("proposed_artifacts", {}),
        }

        st.session_state.chat_history.append(chat_entry)

        manager = get_chat_manager()
        active_chat_id = st.session_state.get("active_chat_id", "")
        if manager and active_chat_id:
            manager.append_exchange(active_chat_id, chat_entry)
            persisted = manager.load_session(active_chat_id) or {}
            manager.save_session(
                active_chat_id,
                persisted.get("messages", []),
                topic_collection=active_collection or "",
                query_mode=retrieval_mode,
            )
            st.session_state.chat_title_draft = persisted.get("title", DEFAULT_CHAT_TITLE)

        st.rerun()
    except Exception as exc:
        logger.error("Eroare la procesarea intrebarii: %s", exc)
        st.error(f"Eroare la procesarea intrebarii: {exc}")


# ---------------------------------------------------------------------------
# Render mesaje chat
# ---------------------------------------------------------------------------


def render_chat_history(
    empty_message: str = "Pune prima intrebare ca sa incepem.",
    target_collection: Optional[str] = None,
) -> None:
    """Randeaza istoricul chat-ului activ, identic in toate view-urile.

    target_collection: colectia in care se vor adauga sursele externe la cerere.
    None = "general" (Second Brain). Notebook-urile pasează numele lor.
    """
    history = st.session_state.get("chat_history", [])
    if not history:
        st.info(empty_message)
        return

    added_urls: set = st.session_state.setdefault("chat_added_urls", set())

    for chat_idx, chat in enumerate(history):
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["response"])

            answer_origin = chat.get("answer_origin", "internal")
            if answer_origin == "external":
                st.warning("Raspuns extern: sursele interne au fost insuficiente.")

            _render_auto_captured_badges(chat)
            _render_proposed_artifacts(chat, chat_idx)

            if chat.get("sources"):
                with st.expander("Surse documente"):
                    for j, source in enumerate(chat["sources"], 1):
                        filename = source.get("filename", f"Document {j}")
                        collection = source.get("collection", GENERAL_COLLECTION_NAME)
                        st.write(f"{j}. **{filename}** ({collection})")

            if chat.get("graph_sources"):
                with st.expander("Surse din knowledge graph"):
                    for j, source in enumerate(chat["graph_sources"], 1):
                        concept = source.get("concept", f"Concept {j}")
                        filename = source.get("filename", "N/A")
                        citation = source.get("citation", f"G{j}")
                        st.write(f"{citation}. **{concept}** -> {filename}")

            if chat.get("memory_hits"):
                with st.expander("Memorie personala relevanta"):
                    for j, item in enumerate(chat["memory_hits"], 1):
                        title = item.get("title", f"Memory {j}")
                        topic = item.get("topic_collection", "")
                        memory_type = item.get("memory_type", "semantic")
                        st.write(f"{j}. **[{memory_type}] {title}** ({topic})")

            if chat.get("note_sources"):
                with st.expander("Notele tale relevante"):
                    for j, note in enumerate(chat["note_sources"], 1):
                        note_title = note.get("title") or f"Nota #{note.get('id', j)}"
                        note_topic = note.get("topic_collection", "") or "global"
                        note_content = note.get("content", "")
                        similarity = float(note.get("similarity", 0.0) or 0.0)
                        preview = note_content[:250] + ("..." if len(note_content) > 250 else "")
                        st.markdown(f"**[N{j}] {note_title}** · {note_topic} · sim={similarity:.2f}")
                        st.caption(preview)

            if chat.get("provenance"):
                with st.expander("Provenienta detaliata"):
                    for entry in chat["provenance"]:
                        citation = entry.get("citation", "?")
                        kind = entry.get("kind", "")
                        score = entry.get("score", 0.0)
                        st.write(f"{citation} | {kind} | scor={score:.3f}")

            if chat.get("external_sources"):
                with st.expander("Surse externe (web)"):
                    target_name = target_collection or GENERAL_COLLECTION_NAME
                    st.caption(f"Adaugarea trimite documentul in colectia '{target_name}'.")
                    for j, source in enumerate(chat["external_sources"], 1):
                        title = source.get("title", f"Sursa externa {j}")
                        url = source.get("url", "")

                        col_link, col_btn = st.columns([4, 1])
                        with col_link:
                            if url:
                                st.markdown(f"{j}. [{title}]({url})")
                            else:
                                st.write(f"{j}. {title}")
                        with col_btn:
                            if not url:
                                continue
                            if url in added_urls:
                                st.caption("✓ Adăugat")
                            else:
                                btn_key = f"chat_add_ext_{chat_idx}_{j}_{abs(hash(url))}"
                                if st.button("Adaugă", key=btn_key, use_container_width=True):
                                    from views._documents import add_web_url_as_document
                                    with st.spinner(f"Fetch {title[:40]}..."):
                                        res = add_web_url_as_document(url, title, target_name)
                                    if res.get("success"):
                                        added_urls.add(url)
                                        st.session_state.chat_added_urls = added_urls
                                        st.success(res["message"])
                                        st.rerun()
                                    else:
                                        st.error(res["message"])

            if chat.get("response_time"):
                response_time = chat["response_time"]
                cached_label = "(cache)" if chat.get("cached") else "(nou)"
                route_used = chat.get("route_used", "vector")
                st.caption(f"{response_time:.2f}s {cached_label} | route={route_used}")
                router_reason = chat.get("router_reason", "")
                origin = chat.get("answer_origin", "internal")
                if router_reason:
                    st.caption(f"origin={origin} | reason={router_reason}")
                else:
                    st.caption(f"origin={origin}")


# ---------------------------------------------------------------------------
# Auto-captured artifacts + proposed confirmation cards
# ---------------------------------------------------------------------------


_PROPOSAL_DISMISSED_KEY = "chat_proposal_dismissed"


def _render_auto_captured_badges(chat: Dict[str, Any]) -> None:
    """Badge verde cu ce a captat sistemul automat dintr-un schimb."""
    captured = chat.get("auto_captured") or {}
    if not captured:
        return

    parts: List[str] = []
    notes = captured.get("notes") or []
    tasks = captured.get("tasks") or []
    decisions = captured.get("decisions") or []
    prefs = captured.get("preferences") or []

    if notes:
        for n in notes:
            title = n.get("title") or f"#{n.get('id')}"
            parts.append(f"💡 nota: *{title}*")
    if tasks:
        for t in tasks:
            title = t.get("title", "task")
            parts.append(f"📌 task: *{title}*")
    if decisions:
        for d in decisions:
            title = d.get("title", "decizie")
            parts.append(f"⚙️ decizie: *{title}*")
    if prefs:
        for p in prefs:
            key = p.get("preference_key", "preferinta")
            parts.append(f"🎯 preferinta: *{key}*")

    if not parts:
        return

    with st.container(border=True):
        st.caption("✓ Captat automat din aceasta conversatie:")
        for p in parts:
            st.markdown(f"- {p}")


def _render_proposed_artifacts(chat: Dict[str, Any], chat_idx: int) -> None:
    """Cards de confirmare pentru artifact-uri detectate dar nesigure.

    Apar cand confidence-ul detectiei e in zona MEDIUM (0.5-0.85). User-ul
    confirma sau respinge per card. Dismissarile sunt volatile (session state)
    asa ca daca reload-ul chat-ului afiseaza din nou, user-ul poate reconsidera.
    """
    proposed = chat.get("proposed_artifacts") or {}
    if not proposed:
        return

    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return

    dismissed: set = st.session_state.setdefault(_PROPOSAL_DISMISSED_KEY, set())

    notes = proposed.get("notes") or []
    visible_notes = [
        n for j, n in enumerate(notes)
        if (chat_idx, "note", j) not in dismissed
    ]
    if not visible_notes:
        return

    with st.container(border=True):
        st.caption("💡 Am detectat ceva ce ar putea merita salvat — vrei sa-l adaug?")

        for j, note in enumerate(notes):
            if (chat_idx, "note", j) in dismissed:
                continue

            content = note.get("content", "")
            title = note.get("title", content[:60])
            topic = note.get("topic_collection", "") or "global"
            conf = float(note.get("confidence", 0.0) or 0.0)

            st.markdown(f"**{title}**")
            st.caption(f"📌 {topic} · confidence {conf:.0%}")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            st.write(preview)

            col_save, col_skip = st.columns(2)
            with col_save:
                if st.button(
                    "Salveaza ca nota",
                    key=f"propose_save_{chat_idx}_{j}",
                    type="primary",
                    use_container_width=True,
                ):
                    note_id = apci_system.create_note(
                        content=content,
                        title=title,
                        topic_collection=topic if topic != "global" else "",
                    )
                    if note_id:
                        dismissed.add((chat_idx, "note", j))
                        st.session_state[_PROPOSAL_DISMISSED_KEY] = dismissed
                        st.success(f"Salvat ca nota #{note_id}.")
                        st.rerun()
                    else:
                        st.error("Salvare esuata.")
            with col_skip:
                if st.button(
                    "Skip",
                    key=f"propose_skip_{chat_idx}_{j}",
                    use_container_width=True,
                ):
                    dismissed.add((chat_idx, "note", j))
                    st.session_state[_PROPOSAL_DISMISSED_KEY] = dismissed
                    st.rerun()


# ---------------------------------------------------------------------------
# Common pre-checks pentru view-uri
# ---------------------------------------------------------------------------


def system_ready() -> bool:
    """True daca sistemul CerebrumAI e initializat."""
    return bool(st.session_state.get("apci_system"))


def has_indexed_documents() -> bool:
    """True daca exista documente indexate (DB sau local)."""
    if st.session_state.get("documents_loaded"):
        return True
    summary = st.session_state.get("last_sync_summary", {})
    return int(summary.get("indexed_chunks", 0) or 0) > 0
