"""Vederea Notes: primitiva Second Brain pentru capturarea gandirii proprii.

Spre deosebire de documente (surse externe) si chat history (conversatii),
notele sunt **scrise direct de tine** — idei, observatii, concluzii personale.
Intra in retrieval ca surse "[N#]" alaturi de documente si memorie.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from views._shared import (
    GENERAL_COLLECTION_NAME,
    is_general_collection,
    normalize_collection_name,
    system_ready,
)


def render_notes() -> None:
    """Punct de intrare pentru tab-ul Notes."""
    st.title("Notes")
    st.caption(
        "Idei proprii, observatii, concluzii. Tot ce gandesti tu, capturat rapid si "
        "regasibil din chat alaturi de documentele tale."
    )

    if not system_ready():
        st.info("Configureaza API key-ul in bara laterala pentru a folosi notele.")
        return

    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        st.error("Sistem indisponibil.")
        return

    repo = getattr(apci_system, "repository", None)
    if not repo or not getattr(repo, "enabled", False):
        st.warning("Persistenta notelor necesita Postgres conectat.")
        return

    _render_capture(apci_system)
    st.divider()
    _render_filters_and_list(apci_system)


# ---------------------------------------------------------------------------
# Capture form
# ---------------------------------------------------------------------------


def _available_collections() -> List[str]:
    """Returneaza colectiile pentru selector (general + notebook-uri)."""
    document_manager = st.session_state.get("document_manager")
    cols = [GENERAL_COLLECTION_NAME]
    if document_manager:
        all_cols = document_manager.get_collections() or {}
        topic_cols = sorted(
            name for name in all_cols.keys() if not is_general_collection(name)
        )
        cols.extend(topic_cols)
    return cols


def _render_capture(apci_system) -> None:
    """Form pentru creare nota."""
    st.subheader("Captureaza o idee noua")

    with st.form("note_capture_form", clear_on_submit=True):
        title = st.text_input(
            "Titlu (opțional)",
            placeholder="ex: Ideea principala pentru capitolul 3",
            key="note_capture_title",
        )
        content = st.text_area(
            "Conținut",
            placeholder="Scrie ideea ta aici. Poate fi orice — o observatie, o intrebare, o concluzie...",
            height=150,
            key="note_capture_content",
        )
        col_topic, col_tags = st.columns([1, 2])
        with col_topic:
            collections = _available_collections()
            topic = st.selectbox(
                "Topic",
                options=collections,
                index=0,
                key="note_capture_topic",
                help="In ce notebook sa stocam nota. 'general' = vizibila peste tot.",
            )
        with col_tags:
            tags_raw = st.text_input(
                "Tag-uri (separate prin virgula)",
                placeholder="ex: idee, todo, important",
                key="note_capture_tags",
            )

        submitted = st.form_submit_button("Salveaza nota", type="primary", use_container_width=True)

        if submitted:
            if not content or not content.strip():
                st.warning("Adauga macar conținut pentru nota.")
                return
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
            topic_clean = "" if is_general_collection(topic) else normalize_collection_name(topic)
            with st.spinner("Indexez nota..."):
                note_id = apci_system.create_note(
                    content=content.strip(),
                    title=title.strip() if title else "",
                    topic_collection=topic_clean,
                    tags=tags,
                )
            if note_id:
                st.success(f"Nota salvata (#{note_id}) si indexata in retrieval.")
                st.rerun()
            else:
                st.error("Salvarea notei a esuat.")


# ---------------------------------------------------------------------------
# Filters + list
# ---------------------------------------------------------------------------


def _render_filters_and_list(apci_system) -> None:
    """Filtre (topic + search) si lista notelor."""
    st.subheader("Notele tale")

    col_search, col_topic = st.columns([2, 1])
    with col_search:
        search_query = st.text_input(
            "Cauta in note",
            placeholder="full-text search peste titlu + conținut",
            key="notes_search_query",
            label_visibility="collapsed",
        )
    with col_topic:
        collections = ["(toate)"] + _available_collections()
        topic_filter = st.selectbox(
            "Filtreaza topic",
            options=collections,
            index=0,
            key="notes_topic_filter",
            label_visibility="collapsed",
        )

    topic_arg: Optional[str]
    if topic_filter == "(toate)":
        topic_arg = None
    elif is_general_collection(topic_filter):
        topic_arg = ""
    else:
        topic_arg = normalize_collection_name(topic_filter)

    notes = apci_system.list_notes(
        topic_collection=topic_arg,
        text_query=search_query if search_query and search_query.strip() else None,
        limit=200,
    )

    if not notes:
        if search_query or topic_filter != "(toate)":
            st.info("Nicio nota nu corespunde filtrelor.")
        else:
            st.info("Inca nu ai capturat nicio nota. Foloseste formularul de mai sus.")
        return

    st.caption(f"{len(notes)} note")

    for note in notes:
        _render_note_card(apci_system, note)


def _render_note_card(apci_system, note: Dict[str, Any]) -> None:
    note_id = note["id"]
    edit_key = f"note_edit_{note_id}"
    confirm_key = f"note_del_{note_id}"

    with st.container(border=True):
        if st.session_state.get(edit_key):
            _render_note_edit_form(apci_system, note, edit_key)
            return

        title_display = note.get("title") or f"Nota #{note_id}"
        topic = note.get("topic_collection") or "global"
        created = (note.get("created_at") or "")[:10]
        updated = (note.get("updated_at") or "")[:10]
        tags = note.get("tags") or []

        st.markdown(f"### {title_display}")

        meta_parts = [f"📌 {topic}", f"creat: {created}"]
        if updated and updated != created:
            meta_parts.append(f"editat: {updated}")
        if tags:
            meta_parts.append(" ".join(f"`{t}`" for t in tags))
        st.caption(" · ".join(meta_parts))

        content = note.get("content", "")
        if len(content) > 500:
            with st.expander("Conținut complet"):
                st.write(content)
            st.write(content[:500] + "...")
        else:
            st.write(content)

        col_edit, col_del = st.columns(2)
        with col_edit:
            if st.button("Editeaza", key=f"note_edit_btn_{note_id}", use_container_width=True):
                st.session_state[edit_key] = True
                st.rerun()
        with col_del:
            if st.session_state.get(confirm_key):
                if st.button(
                    "Confirma stergere",
                    key=f"note_del_ok_{note_id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    if apci_system.delete_note(note_id):
                        st.session_state.pop(confirm_key, None)
                        st.success(f"Nota #{note_id} stearsa.")
                        st.rerun()
                    else:
                        st.error("Stergerea a esuat.")
            else:
                if st.button("Sterge", key=f"note_del_btn_{note_id}", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()


def _render_note_edit_form(apci_system, note: Dict[str, Any], edit_key: str) -> None:
    note_id = note["id"]

    with st.form(f"note_edit_form_{note_id}"):
        new_title = st.text_input(
            "Titlu",
            value=note.get("title", ""),
            key=f"note_edit_title_{note_id}",
        )
        new_content = st.text_area(
            "Conținut",
            value=note.get("content", ""),
            height=200,
            key=f"note_edit_content_{note_id}",
        )

        collections = _available_collections()
        current_topic = note.get("topic_collection") or GENERAL_COLLECTION_NAME
        topic_index = collections.index(current_topic) if current_topic in collections else 0

        col_topic, col_tags = st.columns([1, 2])
        with col_topic:
            new_topic = st.selectbox(
                "Topic",
                options=collections,
                index=topic_index,
                key=f"note_edit_topic_{note_id}",
            )
        with col_tags:
            new_tags_raw = st.text_input(
                "Tag-uri",
                value=", ".join(note.get("tags") or []),
                key=f"note_edit_tags_{note_id}",
            )

        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("Salveaza", type="primary", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Renunta", use_container_width=True)

        if cancel:
            st.session_state.pop(edit_key, None)
            st.rerun()

        if save:
            new_tags = [t.strip() for t in new_tags_raw.split(",") if t.strip()] if new_tags_raw else []
            topic_clean = "" if is_general_collection(new_topic) else normalize_collection_name(new_topic)
            with st.spinner("Reindexez nota..."):
                ok = apci_system.update_note(
                    note_id=note_id,
                    title=new_title.strip(),
                    content=new_content.strip(),
                    topic_collection=topic_clean,
                    tags=new_tags,
                )
            if ok:
                st.session_state.pop(edit_key, None)
                st.success("Nota actualizata.")
                st.rerun()
            else:
                st.error("Actualizarea a esuat.")
