"""Vederea principala: Second Brain.

Chat global care interogheaza intregul corpus + memoria personala,
indiferent de notebook. Aceasta este pagina de pornire a aplicatiei.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from views import _cache
from views._cache import invalidate_reads
from views._documents import save_text_as_document
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

_EXAMPLE_PROMPTS = [
    "Rezuma tot ce am salvat recent",
    "Care sunt deciziile mele recente?",
    "Ce am de facut saptamana asta?",
]

def render_second_brain() -> None:
    """Pagina Second Brain: chat-first.

    Zona principala e doar conversatia. Managementul chat-ului (selector, chat
    nou, optiuni) si uneltele traiesc in bara laterala nativa din stanga
    (pliabila). Fiecare unealta se deschide intr-un dialog modal, la cerere.
    Utilizator nou fara surse: ghidaj in 3 pasi in loc de un ecran gol.
    """
    _render_header()

    if not system_ready():
        st.info("Configureaza API key-ul in bara laterala pentru a activa Second Brain.")
        return

    _render_pinned_reminders()

    # Utilizator nou, fara surse indexate: ghideaza-l pas cu pas.
    if not has_indexed_documents():
        _render_guided_start()
        return

    apci_system = st.session_state.get("apci_system")
    web_available = bool(getattr(apci_system, "web_search_available", False)) if apci_system else False

    # Chat management + unelte -> bara laterala nativa (stanga). Main = doar chat.
    _render_sidebar_panel(apci_system)

    pending_question: Optional[str] = None
    if not st.session_state.get("chat_history"):
        pending_question = _render_welcome_examples()
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
    typed_question = st.chat_input("Intreaba Second Brain...")
    question = (typed_question or "").strip() or (pending_question or "")
    if question:
        process_question(
            question,
            retrieval_mode=GLOBAL_QUERY_MODE,
            active_collection=None,
            force_web=bool(st.session_state.get("sb_force_web_next", False)) if web_available else False,
        )


# ---------------------------------------------------------------------------
# Onboarding + zona principala (chat-first)
# ---------------------------------------------------------------------------


def _render_guided_start() -> None:
    """Ghidaj in 3 pasi pentru utilizatorii fara surse indexate."""
    st.markdown("### Hai sa incepem 🚀")
    st.caption("In 3 pasi simpli, Second Brain devine asistentul tau personal cu memorie.")

    step1, step2, step3 = st.columns(3)
    with step1:
        with st.container(border=True):
            st.markdown("#### 1 · 📚 Adauga o sursa")
            st.write("Incarca un PDF/TXT/MD sau adauga un link intr-un notebook.")
            if st.button(
                "Deschide Notebooks",
                use_container_width=True,
                type="primary",
                key="guided_goto_notebooks",
            ):
                navigate_to("notebooks")
    with step2:
        with st.container(border=True):
            st.markdown("#### 2 · 💬 Pune o intrebare")
            st.write("Dupa indexare, intreaba orice in caseta de chat. Iti raspund cu surse.")
            st.caption("Caseta de chat apare dupa ce ai cel putin o sursa.")
    with step3:
        with st.container(border=True):
            st.markdown("#### 3 · 🧠 Vezi ce am invatat")
            st.write("Decizii, preferinte si task-uri apar in panoul **Unelte** din bara laterala.")

    st.divider()
    st.info("Inca nu ai surse indexate. Incepe cu pasul 1.")


def _render_welcome_examples() -> Optional[str]:
    """Mesaj de bun venit + intrebari-exemplu. Returneaza intrebarea aleasa."""
    st.markdown("#### Bine ai venit la Second Brain 👋")
    st.caption(
        "Iti raspund pe baza a tot ce ai salvat. Incearca una din intrebarile de mai jos "
        "sau scrie a ta in casuta de jos."
    )

    chosen: Optional[str] = None
    cols = st.columns(len(_EXAMPLE_PROMPTS))
    for idx, (col, prompt) in enumerate(zip(cols, _EXAMPLE_PROMPTS)):
        with col:
            if st.button(prompt, key=f"sb_example_{idx}", use_container_width=True):
                chosen = prompt
    st.divider()
    return chosen


# ---------------------------------------------------------------------------
# Bara laterala: chat management + unelte (dialoguri la cerere)
# ---------------------------------------------------------------------------


def _render_sidebar_panel(apci_system) -> None:
    """Randeaza chat management + meniul de unelte in bara laterala nativa."""
    requested_tool: Optional[str] = None
    with st.sidebar:
        st.divider()
        st.markdown("### 💬 Chat")
        _render_sidebar_chat_controls()
        st.divider()
        st.markdown("### 🛠️ Unelte")
        requested_tool = _render_sidebar_tools(apci_system)

    # Modalul e la nivel de pagina: il deschidem in afara contextului sidebar.
    _open_tool_dialog(requested_tool, apci_system)


def _render_sidebar_chat_controls() -> None:
    """Selector chat global + chat nou + optiuni (redenumire/curata/sterge)."""
    manager = get_chat_manager()
    if not manager:
        st.warning("Persistenta chat-urilor nu este disponibila.")
        return

    # Dupa o actiune programatica (Nou/Sterge) aplicam selectia in asteptare INAINTE
    # de a instantia selectbox-ul; altfel valoarea veche ramasa in widget ar suprascrie
    # chat-ul activ la rerun (si "Nou" nu te-ar duce pe chat-ul nou).
    if "_sb_pending_chat" in st.session_state:
        pending_chat = st.session_state.pop("_sb_pending_chat")
        if pending_chat:
            st.session_state["sb_chat_selector"] = pending_chat
        else:
            st.session_state.pop("sb_chat_selector", None)

    ensure_active_chat_for_scope(SCOPE_TOPIC, GLOBAL_QUERY_MODE)

    sessions = [
        s for s in _cache.chat_sessions(manager, type(manager).__name__)
        if not (s.get("topic_collection") or "").strip()
    ]
    session_map = {s["id"]: s for s in sessions}
    session_ids = list(session_map.keys())
    active_chat_id = st.session_state.get("active_chat_id", "")

    if session_ids:
        default_index = session_ids.index(active_chat_id) if active_chat_id in session_ids else 0
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
    else:
        st.caption("Nu exista chat-uri globale inca.")

    col_new, col_opts = st.columns(2)
    with col_new:
        if st.button("➕ Nou", use_container_width=True, key="sb_chat_new"):
            new_id = create_chat_in_scope(SCOPE_TOPIC, GLOBAL_QUERY_MODE)
            if new_id:
                st.session_state["_sb_pending_chat"] = new_id
            st.rerun()
    with col_opts:
        with st.popover("⚙️ Optiuni", use_container_width=True):
            new_title = st.text_input(
                "Titlu chat",
                value=st.session_state.get("chat_title_draft", DEFAULT_CHAT_TITLE),
                key="sb_chat_title_input",
            )
            if st.button("Salveaza titlu", use_container_width=True, key="sb_chat_rename"):
                if rename_active_chat(new_title):
                    st.success("Titlu actualizat.")
                    st.rerun()
            if st.button("🧹 Curata mesaje", use_container_width=True, key="sb_chat_clear"):
                clear_active_chat()
                st.rerun()
            st.divider()
            if st.button("🗑️ Sterge chat", use_container_width=True, key="sb_chat_delete"):
                delete_active_chat(SCOPE_TOPIC, GLOBAL_QUERY_MODE)
                # Sincronizeaza selectbox-ul cu noul chat activ ales de ensure_*.
                st.session_state["_sb_pending_chat"] = st.session_state.get("active_chat_id") or None
                st.rerun()


def _render_sidebar_tools(apci_system) -> Optional[str]:
    """Meniu de unelte. Returneaza cheia uneltei cerute (dialogul se deschide dupa)."""
    try:
        proposals = _cache.memory_proposals(apci_system, "pending", 30) or []
    except Exception:
        proposals = []
    inbox_label = f"📥 Inbox ({len(proposals)})" if proposals else "📥 Inbox"

    requested: Optional[str] = None
    if st.button(inbox_label, use_container_width=True, key="sb_tool_inbox"):
        requested = "inbox"
    if st.button("📊 Insights", use_container_width=True, key="sb_tool_insights"):
        requested = "insights"
    if st.button("✏️ Editor memorie", use_container_width=True, key="sb_tool_editor"):
        requested = "editor"
    if st.button("⚠️ Contradictii", use_container_width=True, key="sb_tool_contradictions"):
        requested = "contradictions"
    if st.button("🔭 Analize globale", use_container_width=True, key="sb_tool_analysis"):
        requested = "analysis"
    if st.button("🔎 Sugestii surse", use_container_width=True, key="sb_tool_suggestions"):
        requested = "suggestions"
    return requested


def _open_tool_dialog(tool: Optional[str], apci_system) -> None:
    if tool == "inbox":
        _dialog_inbox(apci_system)
    elif tool == "insights":
        _dialog_insights(apci_system)
    elif tool == "editor":
        _dialog_editor(apci_system)
    elif tool == "contradictions":
        _dialog_contradictions(apci_system)
    elif tool == "analysis":
        _dialog_analysis(apci_system)
    elif tool == "suggestions":
        _dialog_suggestions()


@st.dialog("📥 Inbox memorie", width="large")
def _dialog_inbox(apci_system) -> None:
    _render_memory_inbox(apci_system)


@st.dialog("📊 Insights", width="large")
def _dialog_insights(apci_system) -> None:
    _render_quick_stats()
    st.divider()
    st.markdown("#### 📅 Digest")
    _render_weekly_digest(apci_system)
    st.markdown("#### 🧠 Memorie recenta")
    _render_memory_surfacing(apci_system)
    st.markdown("#### 🌫️ Decizii de revizuit")
    _render_decision_drift(apci_system)


@st.dialog("✏️ Editor memorie", width="large")
def _dialog_editor(apci_system) -> None:
    _render_memory_editor(apci_system)


@st.dialog("⚠️ Contradictii in cunostinte", width="large")
def _dialog_contradictions(apci_system) -> None:
    _render_contradictions(apci_system)


@st.dialog("🔭 Analize globale", width="large")
def _dialog_analysis(apci_system) -> None:
    _render_global_analysis(apci_system)


@st.dialog("🔎 Sugestii surse", width="large")
def _dialog_suggestions() -> None:
    render_source_suggestions(topic_collection=None)


# ---------------------------------------------------------------------------
# Sectiuni interne
# ---------------------------------------------------------------------------


def _render_header() -> None:
    st.title("Second Brain")
    st.caption(
        "Asistentul tau personal cu memorie peste **toate** sursele si conversatiile tale. "
        "Intreaba orice, indiferent de notebook."
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
            sb_status = _cache.second_brain_status(apci_system) or {}
        except Exception:
            sb_status = {}

    db_stats = sb_status.get("db", {}) if sb_status else {}
    decisions = int(db_stats.get("decisions_count", 0) or 0)
    open_tasks = int(db_stats.get("tasks_open_count", 0) or 0)
    cols[2].metric("Decizii memorate", decisions)
    cols[3].metric("Task-uri active", open_tasks)


# ---------------------------------------------------------------------------
# Pinned reminders (P1: task assistant)
# ---------------------------------------------------------------------------


def _render_pinned_reminders() -> None:
    """Banner cu task-uri scadente azi sau expirate."""
    apci_system = st.session_state.get("apci_system")
    if not apci_system or not hasattr(apci_system, "list_due_soon_tasks"):
        return

    try:
        due_soon = _cache.due_soon_tasks(apci_system, 1, 5) or []
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


def _render_memory_inbox(apci_system) -> None:
    """Persistent inbox for medium-confidence memory proposals."""
    if not hasattr(apci_system, "list_memory_proposals"):
        return

    try:
        proposals = _cache.memory_proposals(apci_system, "pending", 30) or []
    except Exception:
        proposals = []

    with st.container():
        st.caption("Propuneri detectate din conversatii. Accepta doar elementele corecte.")
        if not proposals:
            st.info("Nu exista propuneri pending.")
            return

        for proposal in proposals:
            _render_memory_proposal_card(apci_system, proposal)


def _render_memory_proposal_card(apci_system, proposal: Dict[str, Any]) -> None:
    proposal_id = int(proposal.get("id"))
    proposal_type = proposal.get("proposal_type", "memory")
    payload = dict(proposal.get("payload") or {})
    topic = payload.get("topic_collection", proposal.get("topic_collection", "")) or ""
    confidence = float(proposal.get("confidence", payload.get("confidence", 0.0)) or 0.0)

    with st.container(border=True):
        st.markdown(f"**{_proposal_title(proposal_type, payload)}**")
        st.caption(f"{proposal_type} | {topic or 'global'} | confidence={confidence:.0%}")

        with st.form(f"memory_proposal_form_{proposal_id}"):
            edited_payload: Dict[str, Any] = {"topic_collection": topic}

            if proposal_type == "note":
                edited_payload["title"] = st.text_input(
                    "Titlu",
                    value=payload.get("title", ""),
                    key=f"mp_title_{proposal_id}",
                )
                edited_payload["content"] = st.text_area(
                    "Continut",
                    value=payload.get("content", ""),
                    height=120,
                    key=f"mp_content_{proposal_id}",
                )
            elif proposal_type == "task":
                edited_payload["title"] = st.text_input(
                    "Titlu",
                    value=payload.get("title", ""),
                    key=f"mp_task_title_{proposal_id}",
                )
                edited_payload["details"] = st.text_area(
                    "Detalii",
                    value=payload.get("details", ""),
                    height=100,
                    key=f"mp_task_details_{proposal_id}",
                )
                col_priority, col_due = st.columns(2)
                with col_priority:
                    edited_payload["priority"] = st.selectbox(
                        "Prioritate",
                        options=["low", "normal", "high"],
                        index=["low", "normal", "high"].index(payload.get("priority", "normal"))
                        if payload.get("priority", "normal") in {"low", "normal", "high"} else 1,
                        key=f"mp_task_priority_{proposal_id}",
                    )
                with col_due:
                    edited_payload["due_at"] = st.text_input(
                        "Due date",
                        value=payload.get("due_at") or "",
                        key=f"mp_task_due_{proposal_id}",
                    ) or None
            elif proposal_type == "decision":
                edited_payload["title"] = st.text_input(
                    "Titlu",
                    value=payload.get("title", ""),
                    key=f"mp_decision_title_{proposal_id}",
                )
                edited_payload["rationale"] = st.text_area(
                    "Rationale",
                    value=payload.get("rationale", ""),
                    height=120,
                    key=f"mp_decision_rationale_{proposal_id}",
                )
            elif proposal_type == "preference":
                edited_payload["preference_key"] = st.text_input(
                    "Cheie",
                    value=payload.get("preference_key", "user_preference"),
                    key=f"mp_pref_key_{proposal_id}",
                )
                edited_payload["preference_value"] = st.text_area(
                    "Valoare",
                    value=payload.get("preference_value", ""),
                    height=100,
                    key=f"mp_pref_value_{proposal_id}",
                )
            else:
                st.json(payload)

            edited_payload["topic_collection"] = st.text_input(
                "Notebook/topic",
                value=topic,
                key=f"mp_topic_{proposal_id}",
            )

            col_accept, col_dismiss = st.columns(2)
            with col_accept:
                accept = st.form_submit_button("Accepta", type="primary", use_container_width=True)
            with col_dismiss:
                dismiss = st.form_submit_button("Dismiss", use_container_width=True)

        if accept:
            if apci_system.accept_memory_proposal(proposal_id, edited_payload):
                invalidate_reads()
                st.success("Propunere salvata.")
                st.rerun()
            else:
                st.error("Nu am putut salva propunerea.")
        if dismiss:
            if apci_system.dismiss_memory_proposal(proposal_id):
                invalidate_reads()
                st.rerun()
            else:
                st.error("Nu am putut marca propunerea ca dismissed.")


def _proposal_title(proposal_type: str, payload: Dict[str, Any]) -> str:
    if proposal_type == "preference":
        return f"Preferinta: {payload.get('preference_key', 'preferinta')}"
    return payload.get("title") or f"Propunere {proposal_type}"


def _render_global_analysis(apci_system) -> None:
    """Global synthesis, taxonomy and gap analysis for all Second Brain sources."""
    synth_key = "sb_global_synthesis"
    taxonomy_key = "sb_global_taxonomy"
    gaps_key = "sb_global_gaps"

    with st.container():
        col_synth, col_tax, col_gap, col_clear = st.columns([1, 1, 1, 1])
        with col_synth:
            if st.button("Sinteza globala", key="sb_global_synth_btn", use_container_width=True):
                with st.spinner("Sintetizez tot Second Brain-ul..."):
                    st.session_state[synth_key] = apci_system.synthesize_topic("")
                st.rerun()
        with col_tax:
            if st.button("Taxonomie globala", key="sb_global_tax_btn", use_container_width=True):
                with st.spinner("Construiesc taxonomia globala..."):
                    st.session_state[taxonomy_key] = apci_system.analyze_topic_taxonomy("")
                st.rerun()
        with col_gap:
            if st.button("Gap analysis global", key="sb_global_gap_btn", use_container_width=True):
                with st.spinner("Analizez gap-urile globale..."):
                    st.session_state[gaps_key] = apci_system.analyze_topic_gaps("")
                st.rerun()
        with col_clear:
            if st.button("Curata", key="sb_global_analysis_clear", use_container_width=True):
                for key in (synth_key, taxonomy_key, gaps_key):
                    st.session_state.pop(key, None)
                st.rerun()

        _render_analysis_result("Sinteza globala", st.session_state.get(synth_key), "sinteza_globala")
        _render_analysis_result("Taxonomie globala", st.session_state.get(taxonomy_key), "taxonomie_globala")
        _render_analysis_result("Gap analysis global", st.session_state.get(gaps_key), "gap_analysis_global")


def _render_analysis_result(label: str, data: Dict[str, Any] | None, filename_prefix: str) -> None:
    if not data:
        return
    if data.get("error"):
        st.warning(data["error"])
        return
    markdown = data.get("outline") or data.get("markdown") or ""
    if not markdown:
        st.info(f"{label}: rezultat gol.")
        return
    st.markdown(f"### {label}")
    st.markdown(markdown)
    sources = data.get("sources_used") or {}
    if sources:
        if isinstance(sources, dict) and all(isinstance(v, int) for v in sources.values()):
            st.caption("Surse: " + " | ".join(f"{k}={v}" for k, v in sources.items()))
    if data.get("citation_invalid"):
        st.warning("Au fost eliminate citatii fara provenance: " + ", ".join(data["citation_invalid"]))

    content = f"# {label}\n\n{markdown}\n"
    col_save, col_dl = st.columns([1, 1])
    with col_save:
        if st.button(f"Salveaza {label.lower()}", key=f"sb_save_{filename_prefix}", use_container_width=True):
            with st.spinner("Salvez si indexez analiza..."):
                res = save_text_as_document(
                    content=content,
                    title=label,
                    target_collection="general",
                    description=f"{label} auto-generata din Second Brain",
                    file_extension="md",
                    filename_prefix=filename_prefix,
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
            file_name=f"{filename_prefix}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"sb_dl_{filename_prefix}",
        )


def _render_memory_surfacing(apci_system) -> None:
    """Carduri proactive: decizii recente, task-uri active, preferinte de top."""
    with st.container():
        st.caption("Ce am invatat recent despre tine si ce ai de facut.")

        recent_decisions: list = []
        open_tasks: list = []
        top_prefs: list = []
        try:
            recent_decisions = _cache.recent_decisions(apci_system, 5) or []
        except Exception:
            pass
        try:
            open_tasks = _cache.open_tasks(apci_system, 5) or []
        except Exception:
            pass
        try:
            top_prefs = _cache.top_preferences(apci_system, 5) or []
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
    with st.container():
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
            aged = _cache.aged_decisions(apci_system, threshold_days, 10) or []
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
    with st.container():
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
            contradictions = _cache.contradictions(apci_system, 20, include_dismissed) or []
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
                    invalidate_reads()
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
                    invalidate_reads()
                    st.rerun()
                else:
                    st.error("Dismiss esuat.")


# ---------------------------------------------------------------------------
# P2: Weekly digest
# ---------------------------------------------------------------------------


def _render_weekly_digest(apci_system) -> None:
    """Sumar al activitatii din ultima fereastra de N zile."""
    with st.container():
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
            summary = _cache.weekly_summary(apci_system, int(window)) or {}
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
    with st.container():
        st.caption("Editeaza ce am invatat despre tine. Corecteaza, sterge sau ajusteaza incrementul de incredere.")

        tab_decisions, tab_preferences = st.tabs(["Decizii", "Preferinte"])

        with tab_decisions:
            _render_decisions_editor(apci_system)

        with tab_preferences:
            _render_preferences_editor(apci_system)


def _render_decisions_editor(apci_system) -> None:
    try:
        decisions = _cache.all_decisions(apci_system, 200) or []
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
                        invalidate_reads()
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
                invalidate_reads()
                st.session_state.pop(edit_key, None)
                st.rerun()
            else:
                st.error("Update esuat.")
        if cancel:
            st.session_state.pop(edit_key, None)
            st.rerun()


def _render_preferences_editor(apci_system) -> None:
    try:
        preferences = _cache.all_preferences(apci_system, 200) or []
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
                        invalidate_reads()
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
                invalidate_reads()
                st.session_state.pop(edit_key, None)
                st.rerun()
            else:
                st.error("Update esuat.")
        if cancel:
            st.session_state.pop(edit_key, None)
            st.rerun()
