"""Vederea Notebooks: spatii de lucru focusate pe topic.

- Lista de notebooks (cards) cand niciunul nu este deschis.
- View detaliat (surse + chat) cand un notebook este activ.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from views._documents import process_uploaded_files, save_text_as_document
from views._shared import (
    DEFAULT_CHAT_TITLE,
    clear_active_chat,
    create_chat_in_scope,
    delete_active_chat,
    ensure_active_chat_for_scope,
    format_chat_label,
    get_chat_manager,
    is_general_collection,
    normalize_collection_name,
    process_question,
    rename_active_chat,
    render_chat_history,
    switch_active_chat,
    system_ready,
)
from views._suggestions import render_source_suggestions
from views.graph import render_mini_graph


NOTEBOOK_DEFAULT_QUERY_MODE = "topic_general"
NOTEBOOK_FOCUS_QUERY_MODE = "topic"


def render_notebooks() -> None:
    """Punct de intrare pentru tab-ul Notebooks."""
    document_manager = st.session_state.get("document_manager")
    if not document_manager:
        st.error("Document Manager nu este disponibil.")
        return

    if not system_ready():
        st.info("Configureaza API key-ul in bara laterala pentru a folosi notebook-urile.")
        return

    active_notebook = (st.session_state.get("active_notebook") or "").strip()
    if active_notebook:
        _render_open_notebook(active_notebook)
    else:
        _render_notebooks_list()


# ---------------------------------------------------------------------------
# Lista notebook-uri
# ---------------------------------------------------------------------------


def _render_notebooks_list() -> None:
    st.title("Notebooks")
    st.caption(
        "Spatii de lucru focusate pe un subiect. Fiecare notebook are sursele si "
        "chat-ul lui, dar tot ce inveti aici contribuie si la Second Brain."
    )

    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    notebook_names = sorted(
        [name for name in collections.keys() if not is_general_collection(name)]
    )

    _render_notebook_creator()
    st.divider()

    if not notebook_names:
        st.info(
            "Nu ai inca niciun notebook. Creeaza primul notebook mai sus, apoi "
            "incarca documentele care apartin acelui subiect."
        )
        return

    columns_per_row = 3
    for row_start in range(0, len(notebook_names), columns_per_row):
        row_names = notebook_names[row_start : row_start + columns_per_row]
        cols = st.columns(columns_per_row)
        for col, name in zip(cols, row_names):
            with col:
                _render_notebook_card(name, collections.get(name, {}))


def _render_notebook_creator() -> None:
    with st.form("create_notebook_form", clear_on_submit=True):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            new_name = st.text_input(
                "Nume notebook nou",
                key="new_notebook_name",
                placeholder="ex: Licenta, RAG Papers, SQL Notes...",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Creeaza notebook", use_container_width=True)
        if submitted:
            cleaned = normalize_collection_name(new_name)
            if is_general_collection(cleaned):
                st.warning("Numele 'general' este rezervat pentru sursele globale.")
                return
            document_manager = st.session_state.get("document_manager")
            if document_manager and document_manager.create_collection(cleaned):
                st.session_state.active_notebook = cleaned
                st.success(f"Notebook-ul '{cleaned}' a fost creat.")
                st.rerun()
            else:
                st.error("Nu am putut crea notebook-ul.")


def _render_notebook_card(name: str, info: Dict[str, Any]) -> None:
    document_manager = st.session_state.get("document_manager")
    docs = document_manager.get_documents_by_collection(name) if document_manager else []
    indexed_docs = sum(1 for d in docs if d.get("indexed", False))

    chat_count = _count_chats_for_notebook(name)

    confirm_key = f"nb_delete_confirm_{name}"

    with st.container(border=True):
        st.markdown(f"### {name}")
        st.caption(f"{len(docs)} surse · {indexed_docs} indexate · {chat_count} chat-uri")

        col_open, col_del = st.columns([3, 1])
        with col_open:
            if st.button("Deschide", key=f"open_nb_{name}", use_container_width=True, type="primary"):
                st.session_state.active_notebook = name
                st.rerun()
        with col_del:
            if st.session_state.get(confirm_key):
                if st.button("Confirma", key=f"nb_delete_ok_{name}", use_container_width=True, type="secondary"):
                    dm = st.session_state.get("document_manager")
                    if dm and dm.delete_collection(name):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
            else:
                if st.button("Sterge", key=f"nb_delete_{name}", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()


def _count_chats_for_notebook(name: str) -> int:
    manager = get_chat_manager()
    if not manager:
        return 0
    try:
        sessions = manager.list_sessions()
    except Exception:
        return 0
    target = name.strip()
    return sum(1 for s in sessions if (s.get("topic_collection") or "").strip() == target)


# ---------------------------------------------------------------------------
# View notebook deschis
# ---------------------------------------------------------------------------


def _render_open_notebook(notebook_name: str) -> None:
    document_manager = st.session_state.get("document_manager")
    docs = document_manager.get_documents_by_collection(notebook_name) if document_manager else []

    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.title(notebook_name)
        st.caption(f"{len(docs)} surse in acest notebook")
    with header_right:
        if st.button("Inapoi", use_container_width=True, key="nb_back_to_list"):
            st.session_state.active_notebook = ""
            st.rerun()

    _render_notebook_synthesis(notebook_name)
    _render_notebook_analysis(notebook_name)

    sources_col, chat_col = st.columns([1, 2])

    with sources_col:
        _render_notebook_sources(notebook_name, docs)

    with chat_col:
        _render_notebook_chat(notebook_name)

    with st.expander("Graf de cunostinte (notebook)", expanded=False):
        render_mini_graph(notebook_name)


def _render_notebook_synthesis(notebook_name: str) -> None:
    """Expander cu sinteza topicului — agregare structurata a cunostintelor.

    Buton de generare via Gemini + display markdown + buton de salvare ca
    document indexat in notebook (inchide bucla: sinteza devine sursa noua).
    """
    apci_system = st.session_state.get("apci_system")
    if not apci_system or not hasattr(apci_system, "synthesize_topic"):
        return

    synth_key = f"nb_synth_data_{notebook_name}"

    with st.expander(f"📋 Sintetizeaza tot ce stii despre '{notebook_name}'", expanded=False):
        st.caption(
            "Agregheaza documente + notele tale + decizii + preferinte intr-un outline "
            "structurat. Util inainte de o lucrare sau ca sa identifici gap-uri."
        )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            if st.button(
                "Genereaza sinteza",
                key=f"nb_synth_gen_{notebook_name}",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Agreg surse si sintetizez... poate dura cateva secunde."):
                    data = apci_system.synthesize_topic(notebook_name)
                st.session_state[synth_key] = data
                st.rerun()
        with col_clear:
            if st.session_state.get(synth_key):
                if st.button(
                    "Curata",
                    key=f"nb_synth_clear_{notebook_name}",
                    use_container_width=True,
                ):
                    st.session_state.pop(synth_key, None)
                    st.rerun()

        synth_data = st.session_state.get(synth_key)
        if synth_data is None:
            return

        if synth_data.get("error"):
            st.warning(synth_data["error"])
            return

        outline = synth_data.get("outline", "")
        if not outline:
            st.info("Sinteza goala — Gemini nu a generat continut.")
            return

        st.markdown("---")
        st.markdown(outline)

        sources = synth_data.get("sources_used", {}) or {}
        counts_parts: list[str] = []
        if sources.get("docs"):
            counts_parts.append(f"{len(sources['docs'])} fragmente documente")
        if sources.get("notes"):
            counts_parts.append(f"{len(sources['notes'])} note")
        if sources.get("decisions"):
            counts_parts.append(f"{len(sources['decisions'])} decizii")
        if sources.get("preferences"):
            counts_parts.append(f"{len(sources['preferences'])} preferinte")
        if counts_parts:
            st.caption("Bazat pe: " + " · ".join(counts_parts))

        st.markdown("---")
        col_save, col_dl = st.columns([1, 1])
        full_content = f"# Sinteza: {notebook_name}\n\n{outline}\n"
        with col_save:
            if st.button(
                "Salveaza ca document in notebook",
                key=f"nb_synth_save_{notebook_name}",
                use_container_width=True,
                help="Devine sursa indexata — apare in viitoare raspunsuri ca [D#].",
            ):
                with st.spinner("Salvez si indexez..."):
                    res = save_text_as_document(
                        content=full_content,
                        title=f"Sinteza {notebook_name}",
                        target_collection=notebook_name,
                        description="Sinteza auto-generata din docs + note + decizii + preferinte",
                        file_extension="md",
                        filename_prefix="sinteza",
                    )
                if res.get("success"):
                    st.success(res.get("message", "Salvat."))
                    st.rerun()
                else:
                    st.error(res.get("message", "Salvare esuata."))
        with col_dl:
            st.download_button(
                "Descarca .md",
                data=full_content,
                file_name=f"sinteza_{notebook_name}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"nb_synth_dl_{notebook_name}",
            )


def _render_notebook_analysis(notebook_name: str) -> None:
    """Taxonomie si gap analysis pentru notebook."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return

    taxonomy_key = f"nb_taxonomy_data_{notebook_name}"
    gaps_key = f"nb_gaps_data_{notebook_name}"

    with st.expander(f"Taxonomie si gap analysis pentru '{notebook_name}'", expanded=False):
        col_tax, col_gap, col_clear = st.columns([1, 1, 1])
        with col_tax:
            if st.button("Genereaza taxonomie", key=f"nb_tax_gen_{notebook_name}", use_container_width=True):
                if not hasattr(apci_system, "analyze_topic_taxonomy"):
                    st.error("Taxonomia nu este disponibila in backend.")
                else:
                    with st.spinner("Construiesc taxonomia..."):
                        st.session_state[taxonomy_key] = apci_system.analyze_topic_taxonomy(notebook_name)
                    st.rerun()
        with col_gap:
            if st.button("Genereaza gap analysis", key=f"nb_gap_gen_{notebook_name}", use_container_width=True):
                if not hasattr(apci_system, "analyze_topic_gaps"):
                    st.error("Gap analysis nu este disponibil in backend.")
                else:
                    with st.spinner("Analizez acoperirea si golurile..."):
                        st.session_state[gaps_key] = apci_system.analyze_topic_gaps(notebook_name)
                    st.rerun()
        with col_clear:
            if st.button("Curata analize", key=f"nb_analysis_clear_{notebook_name}", use_container_width=True):
                st.session_state.pop(taxonomy_key, None)
                st.session_state.pop(gaps_key, None)
                st.rerun()

        for label, state_key, prefix in (
            ("Taxonomie", taxonomy_key, "taxonomie"),
            ("Gap analysis", gaps_key, "gap_analysis"),
        ):
            data = st.session_state.get(state_key)
            if not data:
                continue
            if data.get("error"):
                st.warning(data["error"])
                continue
            markdown = data.get("markdown", "")
            if not markdown:
                st.info(f"{label}: rezultat gol.")
                continue
            st.markdown(f"### {label}")
            st.markdown(markdown)
            sources = data.get("sources_used") or {}
            if sources:
                st.caption("Surse: " + " | ".join(f"{k}={v}" for k, v in sources.items()))
            if data.get("citation_invalid"):
                st.warning("Au fost eliminate citatii fara provenance: " + ", ".join(data["citation_invalid"]))

            content = f"# {label}: {notebook_name}\n\n{markdown}\n"
            col_save, col_dl = st.columns([1, 1])
            with col_save:
                if st.button(
                    f"Salveaza {label.lower()} ca document",
                    key=f"nb_{prefix}_save_{notebook_name}",
                    use_container_width=True,
                ):
                    with st.spinner("Salvez si indexez analiza..."):
                        res = save_text_as_document(
                            content=content,
                            title=f"{label} {notebook_name}",
                            target_collection=notebook_name,
                            description=f"{label} auto-generata din Second Brain",
                            file_extension="md",
                            filename_prefix=prefix,
                        )
                    if res.get("success"):
                        st.success(res.get("message", "Salvat."))
                        st.rerun()
                    else:
                        st.error(res.get("message", "Salvare esuata."))
            with col_dl:
                st.download_button(
                    f"Descarca {label.lower()}",
                    data=content,
                    file_name=f"{prefix}_{notebook_name}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"nb_{prefix}_dl_{notebook_name}",
                )


_SORT_OPTIONS = ["Nume", "Data adaugarii", "Marime", "Tip"]

_SORT_KEY_MAP = {
    "Nume": lambda d: (d.get("original_name") or "").lower(),
    "Data adaugarii": lambda d: d.get("added_date") or "",
    "Marime": lambda d: d.get("file_size") or 0,
    "Tip": lambda d: (d.get("file_type") or "").lower(),
}


def _render_notebook_sources(notebook_name: str, docs: List[Dict[str, Any]]) -> None:
    st.subheader("Surse")

    if not docs:
        st.info("Acest notebook nu are surse inca.")
    else:
        # --- Sort & filter controls ---
        file_types = sorted(
            {(d.get("file_type") or "").upper().lstrip(".") for d in docs if d.get("file_type")}
        )

        ctrl_sort, ctrl_type = st.columns(2)
        with ctrl_sort:
            sort_by = st.selectbox(
                "Sorteaza dupa",
                options=_SORT_OPTIONS,
                index=0,
                key=f"nb_sort_{notebook_name}",
                label_visibility="collapsed",
            )
        with ctrl_type:
            filter_types = st.multiselect(
                "Filtreaza tip",
                options=file_types,
                default=[],
                placeholder="Toate tipurile",
                key=f"nb_filter_type_{notebook_name}",
                label_visibility="collapsed",
            )

        show_indexed_only = st.checkbox(
            "Doar indexate",
            value=False,
            key=f"nb_filter_indexed_{notebook_name}",
        )

        # Apply filters then sort
        filtered = list(docs)
        if filter_types:
            filtered = [
                d for d in filtered
                if (d.get("file_type") or "").upper().lstrip(".") in filter_types
            ]
        if show_indexed_only:
            filtered = [d for d in filtered if d.get("indexed", False)]
        filtered.sort(key=_SORT_KEY_MAP[sort_by])

        total = len(docs)
        shown = len(filtered)
        st.caption(f"{shown} din {total} surse")

        for doc in filtered:
            with st.container(border=True):
                st.write(f"**{doc.get('original_name', 'fara nume')}**")
                size_kb = doc.get("file_size", 0) / 1024
                indexed_label = "indexat" if doc.get("indexed", False) else "neindexat"
                file_type = (doc.get("file_type") or "").upper().lstrip(".")
                added = (doc.get("added_date") or "")[:10]
                st.caption(f"{file_type} · {size_kb:.1f} KB · {indexed_label} · {added}")

                if st.button("Sterge", key=f"nb_del_{doc['id']}", use_container_width=True):
                    document_manager = st.session_state.get("document_manager")
                    if document_manager and document_manager.remove_document(doc["id"]):
                        st.success("Document sters.")
                        st.rerun()

    st.divider()
    st.markdown("**Adauga surse**")
    uploaded = st.file_uploader(
        "Incarca fisiere in acest notebook",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        key=f"nb_upload_{notebook_name}",
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("Proceseaza", type="primary", key=f"nb_process_{notebook_name}"):
            with st.spinner("Procesez si indexez..."):
                result = process_uploaded_files(uploaded, target_collection=notebook_name)
            if result.get("success"):
                st.success(result.get("message", "Done."))
                st.rerun()
            else:
                st.error(result.get("message", "Eroare."))

    st.divider()
    render_source_suggestions(topic_collection=notebook_name)


def _render_notebook_chat(notebook_name: str) -> None:
    st.subheader("Conversatie")

    manager = get_chat_manager()
    if not manager:
        st.warning("Persistenta chat-urilor nu este disponibila.")
        return

    focus_only = st.checkbox(
        "Doar surse din acest notebook (ignora memoria globala)",
        value=False,
        key=f"nb_focus_{notebook_name}",
        help="Daca e activ, raspunsul se bazeaza strict pe documentele din notebook.",
    )
    query_mode = NOTEBOOK_FOCUS_QUERY_MODE if focus_only else NOTEBOOK_DEFAULT_QUERY_MODE

    ensure_active_chat_for_scope(notebook_name, query_mode)

    sessions = [
        s for s in manager.list_sessions()
        if (s.get("topic_collection") or "").strip() == notebook_name.strip()
    ]
    session_map = {s["id"]: s for s in sessions}
    session_ids = list(session_map.keys())
    active_chat_id = st.session_state.get("active_chat_id", "")

    col_select, col_new, col_del = st.columns([4, 1, 1])
    with col_select:
        if session_ids:
            default_index = session_ids.index(active_chat_id) if active_chat_id in session_ids else 0
            selected_id = st.selectbox(
                "Chat",
                options=session_ids,
                index=default_index,
                format_func=lambda chat_id: format_chat_label(session_map[chat_id]),
                key=f"nb_chat_select_{notebook_name}",
                label_visibility="collapsed",
            )
            if selected_id != active_chat_id:
                switch_active_chat(selected_id)
                st.rerun()
        else:
            st.caption("Nu exista chat-uri in acest notebook.")
    with col_new:
        if st.button("Chat nou", use_container_width=True, key=f"nb_new_chat_{notebook_name}"):
            create_chat_in_scope(notebook_name, query_mode)
            st.rerun()
    with col_del:
        if st.button("Sterge", use_container_width=True, key=f"nb_del_chat_{notebook_name}"):
            delete_active_chat(notebook_name, query_mode)
            st.rerun()

    with st.expander("Optiuni chat", expanded=False):
        col_title, col_save, col_clear = st.columns([3, 1, 1])
        with col_title:
            new_title = st.text_input(
                "Titlu chat",
                value=st.session_state.get("chat_title_draft", DEFAULT_CHAT_TITLE),
                key=f"nb_chat_title_{notebook_name}",
                label_visibility="collapsed",
            )
        with col_save:
            if st.button("Salveaza titlu", use_container_width=True, key=f"nb_rename_{notebook_name}"):
                if rename_active_chat(new_title):
                    st.success("Titlu actualizat.")
                    st.rerun()
        with col_clear:
            if st.button("Curata mesaje", use_container_width=True, key=f"nb_clear_{notebook_name}"):
                clear_active_chat()
                st.rerun()

    st.divider()
    render_chat_history(
        empty_message="Pune prima intrebare despre sursele din acest notebook.",
        target_collection=notebook_name,
    )

    apci_system = st.session_state.get("apci_system")
    web_available = bool(getattr(apci_system, "web_search_available", False)) if apci_system else False

    if web_available:
        force_web = st.checkbox(
            "Cauta pe web pentru urmatoarea intrebare (Tavily)",
            value=False,
            key=f"nb_force_web_{notebook_name}",
            help="Daca e bifat, urmatoarea intrebare merge direct catre Tavily, ignorand sursele notebook-ului.",
        )
    else:
        force_web = False

    user_question = st.chat_input(f"Intreaba despre '{notebook_name}'...")
    if user_question and user_question.strip():
        process_question(
            user_question.strip(),
            retrieval_mode=query_mode,
            active_collection=notebook_name,
            force_web=force_web,
        )
