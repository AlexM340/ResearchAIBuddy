"""Vederea principala: Second Brain.

Chat global care interogheaza intregul corpus + memoria personala,
indiferent de notebook. Aceasta este pagina de pornire a aplicatiei.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from views._shared import (
    DEFAULT_CHAT_TITLE,
    clear_active_chat,
    create_chat_in_scope,
    delete_active_chat,
    ensure_active_chat_for_scope,
    format_chat_label,
    get_chat_manager,
    has_indexed_documents,
    navigate_to,
    process_question,
    rename_active_chat,
    render_chat_history,
    switch_active_chat,
    system_ready,
)
from views._suggestions import render_source_suggestions


GLOBAL_QUERY_MODE = "all"
SCOPE_TOPIC: str = ""  # global = fara topic_collection


def render_second_brain() -> None:
    """Randeaza pagina Second Brain (chat global) cu layout in 2 tab-uri.

    - Conversatie: chat controls + sugestii (collapsed) + istoric chat + chat input
    - Insights & Tools: stats + panel-uri memorie/digest/contradictii
    """
    _render_header()

    if not system_ready():
        st.info("Configureaza API key-ul in bara laterala pentru a activa Second Brain.")
        return

    _render_pinned_reminders()

    if not has_indexed_documents():
        _render_empty_state()
        return

    apci_system = st.session_state.get("apci_system")
    web_available = bool(getattr(apci_system, "web_search_available", False)) if apci_system else False

    chat_tab, tools_tab = st.tabs(["💬 Conversație", "🛠️ Insights & Tools"])

    with chat_tab:
        _render_chat_controls()
        render_source_suggestions(topic_collection=None)
        st.divider()
        render_chat_history(
            empty_message="Intreaba orice. Caut peste tot ce am invatat impreuna.",
            target_collection=None,
        )
        if web_available:
            st.checkbox(
                "Cauta pe web pentru urmatoarea intrebare (Tavily)",
                value=False,
                key="sb_force_web_next",
                help="Daca e bifat, urmatoarea intrebare este trimisa direct catre Tavily, ignorand sursele interne.",
            )

    with tools_tab:
        with st.expander("📊 Statistici sistem", expanded=False):
            _render_quick_stats()
        _render_second_brain_panels()

    user_question = st.chat_input("Intreaba Second Brain...")
    if user_question and user_question.strip():
        process_question(
            user_question.strip(),
            retrieval_mode=GLOBAL_QUERY_MODE,
            active_collection=None,
            force_web=bool(st.session_state.get("sb_force_web_next", False)) if web_available else False,
        )


# ---------------------------------------------------------------------------
# Sectiuni interne
# ---------------------------------------------------------------------------


def _render_header() -> None:
    st.title("Second Brain")
    st.caption(
        "Asistentul tau personal cu memorie peste **toate** sursele si conversatiile tale. "
        "Intreaba orice, indiferent de notebook."
    )


def _render_empty_state() -> None:
    st.warning("Inca nu ai surse indexate.")
    st.markdown(
        "Mergi la tab-ul **Notebooks** pentru a-ti crea primul notebook si a incarca documente. "
        "Pe masura ce adaugi surse, Second Brain le va invata pe toate."
    )


def _render_quick_stats() -> None:
    apci_system = st.session_state.get("apci_system")
    document_manager = st.session_state.get("document_manager")
    if not apci_system:
        return

    cols = st.columns(4)

    if document_manager:
        lib_stats = document_manager.get_library_stats()
        cols[0].metric("Documente", lib_stats.get("total_documents", 0))
        cols[1].metric("Notebooks", lib_stats.get("total_collections", 0))
    else:
        cols[0].metric("Documente", "N/A")
        cols[1].metric("Notebooks", "N/A")

    sb_status = {}
    if hasattr(apci_system, "get_second_brain_status"):
        try:
            sb_status = apci_system.get_second_brain_status() or {}
        except Exception:
            sb_status = {}

    db_stats = sb_status.get("db", {}) if sb_status else {}
    decisions = int(db_stats.get("decisions_count", 0) or 0)
    open_tasks = int(db_stats.get("tasks_open_count", 0) or 0)
    cols[2].metric("Decizii memorate", decisions)
    cols[3].metric("Task-uri active", open_tasks)


def _render_chat_controls() -> None:
    """Selector chat global + actiuni (nou/sterge/redenumire/curata)."""
    manager = get_chat_manager()
    if not manager:
        st.warning("Persistenta chat-urilor nu este disponibila.")
        return

    ensure_active_chat_for_scope(SCOPE_TOPIC, GLOBAL_QUERY_MODE)

    sessions = [
        s for s in manager.list_sessions()
        if not (s.get("topic_collection") or "").strip()
    ]
    session_map = {s["id"]: s for s in sessions}
    session_ids = list(session_map.keys())
    active_chat_id = st.session_state.get("active_chat_id", "")

    if not session_ids:
        st.caption("Nu exista chat-uri globale inca.")
        return

    default_index = session_ids.index(active_chat_id) if active_chat_id in session_ids else 0

    col_select, col_new, col_del = st.columns([4, 1, 1])

    with col_select:
        selected_id = st.selectbox(
            "Chat global activ",
            options=session_ids,
            index=default_index,
            format_func=lambda chat_id: format_chat_label(session_map[chat_id]),
            key="sb_chat_selector",
            label_visibility="collapsed",
        )
        if selected_id != active_chat_id:
            switch_active_chat(selected_id)
            st.rerun()

    with col_new:
        if st.button("Chat nou", use_container_width=True, key="sb_chat_new"):
            create_chat_in_scope(SCOPE_TOPIC, GLOBAL_QUERY_MODE)
            st.rerun()

    with col_del:
        if st.button("Sterge", use_container_width=True, key="sb_chat_delete"):
            delete_active_chat(SCOPE_TOPIC, GLOBAL_QUERY_MODE)
            st.rerun()

    with st.expander("Optiuni chat", expanded=False):
        col_title, col_save, col_clear = st.columns([3, 1, 1])
        with col_title:
            new_title = st.text_input(
                "Titlu chat",
                value=st.session_state.get("chat_title_draft", DEFAULT_CHAT_TITLE),
                key="sb_chat_title_input",
                label_visibility="collapsed",
            )
        with col_save:
            if st.button("Salveaza titlu", use_container_width=True, key="sb_chat_rename"):
                if rename_active_chat(new_title):
                    st.success("Titlu actualizat.")
                    st.rerun()
        with col_clear:
            if st.button("Curata mesaje", use_container_width=True, key="sb_chat_clear"):
                clear_active_chat()
                st.rerun()


# ---------------------------------------------------------------------------
# Pinned reminders (P1: task assistant)
# ---------------------------------------------------------------------------


def _render_pinned_reminders() -> None:
    """Banner cu task-uri scadente azi sau expirate."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system or not hasattr(apci_system, "list_due_soon_tasks"):
        return

    try:
        due_soon = apci_system.list_due_soon_tasks(within_days=1, limit=5) or []
    except Exception:
        return

    if not due_soon:
        return

    overdue = [t for t in due_soon if (t.get("days_until_due") or 0) < 0]
    today = [t for t in due_soon if 0 <= (t.get("days_until_due") or 0) < 1]

    if not overdue and not today:
        return

    with st.container(border=True):
        col_msg, col_action = st.columns([4, 1])
        with col_msg:
            parts: list[str] = []
            if overdue:
                parts.append(f"**{len(overdue)} task-uri expirate**")
            if today:
                parts.append(f"**{len(today)} scadente azi**")
            st.warning(" · ".join(parts))
            preview_titles = [t.get("title", "") for t in (overdue + today)[:3]]
            st.caption(" · ".join(f"\"{t}\"" for t in preview_titles if t))
        with col_action:
            if st.button("Vezi tasks", use_container_width=True, type="primary", key="sb_goto_tasks"):
                navigate_to("tasks")


# ---------------------------------------------------------------------------
# E4: Second Brain Polish — contradictii, decision drift, memory surfacing
# ---------------------------------------------------------------------------


def _render_second_brain_panels() -> None:
    """Randeaza panel-urile E4 + P2 (weekly digest + memory editor)."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return

    _render_weekly_digest(apci_system)

    panel_left, panel_right = st.columns(2)

    with panel_left:
        _render_memory_surfacing(apci_system)

    with panel_right:
        _render_decision_drift(apci_system)

    _render_memory_editor(apci_system)
    _render_contradictions(apci_system)


def _render_memory_surfacing(apci_system) -> None:
    """Carduri proactive: decizii recente, task-uri active, preferinte de top."""
    with st.expander("Memorie surfaced", expanded=False):
        st.caption("Ce am invatat recent despre tine si ce ai de facut.")

        recent_decisions: list = []
        open_tasks: list = []
        top_prefs: list = []
        try:
            recent_decisions = apci_system.list_recent_decisions(limit=5) or []
        except Exception:
            pass
        try:
            open_tasks = apci_system.list_open_tasks(limit=5) or []
        except Exception:
            pass
        try:
            top_prefs = apci_system.list_top_preferences(limit=5) or []
        except Exception:
            pass

        if not recent_decisions and not open_tasks and not top_prefs:
            st.info("Inca nu am acumulat memorie. Pune intrebari ca sa invat preferintele tale.")
            return

        st.markdown("**Decizii recente**")
        if recent_decisions:
            for decision in recent_decisions:
                topic = decision.get("topic_collection") or "global"
                created = (decision.get("created_at") or "")[:10]
                conf = decision.get("confidence", 0.0)
                st.write(f"- **{decision.get('title', 'fara titlu')}** · {topic} · {created} · conf={conf:.2f}")
        else:
            st.caption("Nu exista decizii memorate inca.")

        st.markdown("**Task-uri active**")
        if open_tasks:
            for task in open_tasks:
                topic = task.get("topic_collection") or "global"
                status = task.get("status", "open")
                priority = task.get("priority", "normal")
                st.write(f"- **{task.get('title', 'fara titlu')}** · {topic} · [{status}/{priority}]")
        else:
            st.caption("Nu ai task-uri deschise.")

        st.markdown("**Preferinte top**")
        if top_prefs:
            for pref in top_prefs:
                key = pref.get("preference_key", "")
                value = pref.get("preference_value", "")
                topic = pref.get("topic_collection") or "global"
                conf = pref.get("confidence", 0.0)
                st.write(f"- **{key}**: {value} · {topic} · conf={conf:.2f}")
        else:
            st.caption("Nu am extras preferinte inca.")


def _render_decision_drift(apci_system) -> None:
    """Decizii vechi, candidat pentru review."""
    with st.expander("Decision drift", expanded=False):
        st.caption("Decizii mai vechi care ar putea fi depasite — verifica daca mai sunt valide.")

        threshold_days = st.slider(
            "Vechime minima (zile)",
            min_value=7,
            max_value=180,
            value=30,
            step=7,
            key="sb_drift_threshold",
        )

        try:
            aged = apci_system.list_aged_decisions(min_age_days=threshold_days, limit=10) or []
        except Exception:
            aged = []

        if not aged:
            st.success(f"Nicio decizie mai veche de {threshold_days} zile. Memorie proaspata.")
            return

        st.warning(f"{len(aged)} decizii mai vechi de {threshold_days} zile.")
        for decision in aged:
            topic = decision.get("topic_collection") or "global"
            age = decision.get("age_days", 0)
            conf = decision.get("confidence", 0.0)
            with st.container(border=True):
                st.markdown(f"**{decision.get('title', 'fara titlu')}**")
                st.caption(f"{topic} · {age} zile · conf={conf:.2f}")
                rationale = decision.get("rationale", "")
                if rationale:
                    st.write(rationale[:300] + ("..." if len(rationale) > 300 else ""))


def _render_contradictions(apci_system) -> None:
    """Tensiuni detectate in graful de cunostinte (relatii CONTRADICTS)."""
    with st.expander("Contradictii in cunostinte", expanded=False):
        neo4j_client = getattr(apci_system, "neo4j_client", None)
        if not neo4j_client or not getattr(neo4j_client, "enabled", False):
            st.caption("Neo4j nu este disponibil — contradictii nu pot fi detectate.")
            return

        st.caption("Concepte care se contrazic, extrase din relatiile CONTRADICTS din graf.")

        col_show, _ = st.columns([1, 3])
        with col_show:
            include_dismissed = st.checkbox(
                "Arata si dismissed",
                value=False,
                key="sb_contradictions_show_dismissed",
            )

        try:
            contradictions = apci_system.get_contradictions(
                limit=20,
                include_dismissed=include_dismissed,
            ) or []
        except TypeError:
            # Backward-compat daca system wrapper nu accepta include_dismissed.
            try:
                contradictions = apci_system.get_contradictions(limit=20) or []
            except Exception:
                contradictions = []
        except Exception:
            contradictions = []

        if not contradictions:
            st.success("Nicio contradictie activa.")
            return

        active = [c for c in contradictions if not c.get("dismissed", False)]
        dismissed = [c for c in contradictions if c.get("dismissed", False)]

        if active:
            st.warning(f"{len(active)} contradictii active.")
            for idx, item in enumerate(active):
                _render_contradiction_card(apci_system, item, dismissed=False, card_index=idx)

        if dismissed and include_dismissed:
            st.caption(f"{len(dismissed)} contradictii dismissed (rezolvate manual).")
            for idx, item in enumerate(dismissed):
                _render_contradiction_card(apci_system, item, dismissed=True, card_index=idx)


def _render_contradiction_card(apci_system, item: Dict[str, Any], dismissed: bool, card_index: int) -> None:
    source = item.get("source", "?")
    target = item.get("target", "?")
    topic = item.get("topic", "general")
    conf = item.get("confidence", 0.0)
    documents = item.get("documents", [])
    badge = "✓ dismissed · " if dismissed else ""

    with st.container(border=True):
        st.markdown(f"{badge}**{source}** ⟷ **{target}**")
        st.caption(f"{topic} · conf={conf:.2f}")
        if documents:
            docs_label = ", ".join(documents[:3])
            if len(documents) > 3:
                docs_label += f" (+{len(documents) - 3})"
            st.caption(f"Surse: {docs_label}")

        if dismissed:
            if st.button(
                "Restaureaza",
                key=f"contr_restore_{card_index}_{source}_{target}",
                use_container_width=True,
            ):
                if apci_system.restore_contradiction(source=source, target=target):
                    st.rerun()
                else:
                    st.error("Restore esuat.")
        else:
            if st.button(
                "Dismiss (rezolvat manual)",
                key=f"contr_dismiss_{card_index}_{source}_{target}",
                use_container_width=True,
            ):
                if apci_system.dismiss_contradiction(source=source, target=target):
                    st.rerun()
                else:
                    st.error("Dismiss esuat.")


# ---------------------------------------------------------------------------
# P2: Weekly digest
# ---------------------------------------------------------------------------


def _render_weekly_digest(apci_system) -> None:
    """Sumar al activitatii din ultima fereastra de N zile."""
    with st.expander("Digest", expanded=False):
        col_label, col_window = st.columns([3, 1])
        with col_label:
            st.caption("Ce am invatat si ce ai facut in ultima perioada.")
        with col_window:
            window = st.selectbox(
                "Zile",
                options=[1, 3, 7, 14, 30],
                index=2,
                key="sb_digest_window",
                label_visibility="collapsed",
            )

        try:
            summary = apci_system.get_weekly_summary(days=int(window)) or {}
        except Exception:
            summary = {}

        if not summary:
            st.info("Persistenta DB indisponibila — nu pot calcula digest-ul.")
            return

        cols = st.columns(4)
        cols[0].metric("Decizii noi", summary.get("decisions_added", 0))
        cols[1].metric("Task-uri noi", summary.get("tasks_added", 0))
        cols[2].metric("Task-uri inchise", summary.get("tasks_closed", 0))
        cols[3].metric("Documente noi", summary.get("documents_added", 0))

        st.caption(
            f"Conversatii: {summary.get('messages_count', 0)} mesaje · "
            f"Task-uri ramase deschise: {summary.get('tasks_currently_open', 0)}"
        )

        recent_decisions = summary.get("recent_decisions") or []
        if recent_decisions:
            st.markdown("**Decizii ultime**")
            for d in recent_decisions:
                topic = d.get("topic_collection") or "global"
                created = (d.get("created_at") or "")[:10]
                st.write(f"- {d.get('title', 'fara titlu')} · {topic} · {created}")

        recent_closed = summary.get("recent_closed_tasks") or []
        if recent_closed:
            st.markdown("**Task-uri inchise recent**")
            for t in recent_closed:
                topic = t.get("topic_collection") or "global"
                status = t.get("status", "done")
                updated = (t.get("updated_at") or "")[:10]
                st.write(f"- {t.get('title', 'fara titlu')} · {topic} · [{status}] · {updated}")

        if not recent_decisions and not recent_closed:
            st.caption("Nicio activitate notabila in fereastra selectata.")


# ---------------------------------------------------------------------------
# P2: Memory editor (CRUD pentru decizii si preferinte)
# ---------------------------------------------------------------------------


def _render_memory_editor(apci_system) -> None:
    """Editor cu CRUD pentru decizii si preferinte memorate."""
    with st.expander("Editor memorie", expanded=False):
        st.caption("Editeaza ce am invatat despre tine. Corecteaza, sterge sau ajusteaza incrementul de incredere.")

        tab_decisions, tab_preferences = st.tabs(["Decizii", "Preferinte"])

        with tab_decisions:
            _render_decisions_editor(apci_system)

        with tab_preferences:
            _render_preferences_editor(apci_system)


def _render_decisions_editor(apci_system) -> None:
    try:
        decisions = apci_system.list_all_decisions(limit=200) or []
    except Exception:
        decisions = []

    if not decisions:
        st.info("Nicio decizie memorata inca.")
        return

    st.caption(f"{len(decisions)} decizii memorate.")

    for decision in decisions:
        decision_id = decision["id"]
        edit_key = f"edit_decision_{decision_id}"
        confirm_key = f"del_decision_{decision_id}"

        with st.container(border=True):
            if st.session_state.get(edit_key):
                _render_decision_edit_form(apci_system, decision, edit_key)
                continue

            topic = decision.get("topic_collection") or "global"
            created = (decision.get("created_at") or "")[:10]
            conf = decision.get("confidence", 0.0)
            st.markdown(f"**{decision.get('title', 'fara titlu')}**")
            st.caption(f"{topic} · {created} · conf={conf:.2f}")
            rationale = decision.get("rationale", "")
            if rationale:
                st.write(rationale[:300] + ("..." if len(rationale) > 300 else ""))

            cols = st.columns(2)
            with cols[0]:
                if st.button("Edit", key=f"d_edit_btn_{decision_id}", use_container_width=True):
                    st.session_state[edit_key] = True
                    st.rerun()
            with cols[1]:
                if st.session_state.get(confirm_key):
                    if st.button(
                        "Confirma stergere",
                        key=f"d_del_ok_{decision_id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        apci_system.delete_decision(decision_id)
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                else:
                    if st.button("Sterge", key=f"d_del_{decision_id}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()


def _render_decision_edit_form(apci_system, decision: Dict[str, Any], edit_key: str) -> None:
    decision_id = decision["id"]
    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    topic_options = ["(global)"] + sorted(
        name for name in collections.keys()
        if name.lower() not in {"general", "default"}
    )

    current_topic = decision.get("topic_collection") or ""
    topic_index = 0
    if current_topic and current_topic in topic_options:
        topic_index = topic_options.index(current_topic)

    with st.form(f"edit_decision_form_{decision_id}"):
        new_title = st.text_input("Titlu", value=decision.get("title", ""), key=f"d_f_title_{decision_id}")
        new_rationale = st.text_area(
            "Rationale",
            value=decision.get("rationale", ""),
            key=f"d_f_rat_{decision_id}",
            height=100,
        )
        col_topic, col_conf = st.columns(2)
        with col_topic:
            new_topic = st.selectbox(
                "Notebook",
                options=topic_options,
                index=topic_index,
                key=f"d_f_topic_{decision_id}",
            )
        with col_conf:
            new_conf = st.slider(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(decision.get("confidence", 0.5)),
                step=0.05,
                key=f"d_f_conf_{decision_id}",
            )

        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("Salveaza", type="primary", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Anuleaza", use_container_width=True)

        if save:
            topic_value = "" if new_topic == "(global)" else new_topic
            ok = apci_system.update_decision(
                decision_id=decision_id,
                title=new_title,
                rationale=new_rationale,
                topic_collection=topic_value,
                confidence=new_conf,
            )
            if ok:
                st.session_state.pop(edit_key, None)
                st.rerun()
            else:
                st.error("Update esuat.")
        if cancel:
            st.session_state.pop(edit_key, None)
            st.rerun()


def _render_preferences_editor(apci_system) -> None:
    try:
        preferences = apci_system.list_all_preferences(limit=200) or []
    except Exception:
        preferences = []

    if not preferences:
        st.info("Nicio preferinta memorata inca.")
        return

    st.caption(f"{len(preferences)} preferinte memorate.")

    for pref in preferences:
        pref_id = pref["id"]
        edit_key = f"edit_pref_{pref_id}"
        confirm_key = f"del_pref_{pref_id}"

        with st.container(border=True):
            if st.session_state.get(edit_key):
                _render_preference_edit_form(apci_system, pref, edit_key)
                continue

            topic = pref.get("topic_collection") or "global"
            conf = pref.get("confidence", 0.0)
            updated = (pref.get("updated_at") or "")[:10]
            st.markdown(f"**{pref.get('preference_key', '?')}**: {pref.get('preference_value', '')}")
            st.caption(f"{topic} · {updated} · conf={conf:.2f}")

            cols = st.columns(2)
            with cols[0]:
                if st.button("Edit", key=f"p_edit_btn_{pref_id}", use_container_width=True):
                    st.session_state[edit_key] = True
                    st.rerun()
            with cols[1]:
                if st.session_state.get(confirm_key):
                    if st.button(
                        "Confirma stergere",
                        key=f"p_del_ok_{pref_id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        apci_system.delete_preference(pref_id)
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                else:
                    if st.button("Sterge", key=f"p_del_{pref_id}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()


def _render_preference_edit_form(apci_system, pref: Dict[str, Any], edit_key: str) -> None:
    pref_id = pref["id"]
    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    topic_options = ["(global)"] + sorted(
        name for name in collections.keys()
        if name.lower() not in {"general", "default"}
    )

    current_topic = pref.get("topic_collection") or ""
    topic_index = 0
    if current_topic and current_topic in topic_options:
        topic_index = topic_options.index(current_topic)

    with st.form(f"edit_pref_form_{pref_id}"):
        st.caption(f"Cheia preferintei: **{pref.get('preference_key', '?')}** (read-only)")
        new_value = st.text_area(
            "Valoare",
            value=pref.get("preference_value", ""),
            key=f"p_f_val_{pref_id}",
            height=80,
        )
        col_topic, col_conf = st.columns(2)
        with col_topic:
            new_topic = st.selectbox(
                "Notebook",
                options=topic_options,
                index=topic_index,
                key=f"p_f_topic_{pref_id}",
            )
        with col_conf:
            new_conf = st.slider(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(pref.get("confidence", 0.5)),
                step=0.05,
                key=f"p_f_conf_{pref_id}",
            )

        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("Salveaza", type="primary", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Anuleaza", use_container_width=True)

        if save:
            topic_value = "" if new_topic == "(global)" else new_topic
            ok = apci_system.update_preference(
                preference_id=pref_id,
                preference_value=new_value,
                confidence=new_conf,
                topic_collection=topic_value,
            )
            if ok:
                st.session_state.pop(edit_key, None)
                st.rerun()
            else:
                st.error("Update esuat.")
        if cancel:
            st.session_state.pop(edit_key, None)
            st.rerun()
