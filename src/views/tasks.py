"""Vederea Tasks: capture rapid, kanban si reminders.

- Reminders pinned (due-soon + expirate) sus
- Quick capture: form scurt pentru un task nou
- Kanban: 3 coloane (Open / In Progress / Done) cu inline edit
- Filtre: topic, priority
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from views._shared import is_general_collection, system_ready


PRIORITY_OPTIONS = ["low", "normal", "high"]
PRIORITY_BADGE = {"low": "🟢", "normal": "🟡", "high": "🔴"}
STATUS_COLUMNS = [("open", "Open"), ("in_progress", "In progress"), ("done", "Done")]


def render_tasks() -> None:
    """Punct de intrare pentru tab-ul Tasks."""
    st.title("Tasks")
    st.caption("Quick capture, reminders si lifecycle pentru tot ce ai de facut.")

    if not system_ready():
        st.info("Configureaza API key-ul in bara laterala pentru a folosi tab-ul Tasks.")
        return

    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        st.error("Sistem indisponibil.")
        return

    repo = getattr(apci_system, "repository", None)
    if not repo or not getattr(repo, "enabled", False):
        st.warning("Persistenta tasks necesita Postgres conectat.")
        return

    _render_reminders(apci_system)
    st.divider()
    _render_quick_capture(apci_system)
    st.divider()
    _render_kanban(apci_system)


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------


def _render_reminders(apci_system) -> None:
    within = st.session_state.get("tasks_reminder_window", 7)

    col_label, col_window = st.columns([3, 1])
    with col_label:
        st.subheader("Reminders")
    with col_window:
        within = st.selectbox(
            "Fereastra (zile)",
            options=[1, 3, 7, 14, 30],
            index=[1, 3, 7, 14, 30].index(within) if within in [1, 3, 7, 14, 30] else 2,
            key="tasks_reminder_window",
            label_visibility="collapsed",
        )

    try:
        due_soon = apci_system.list_due_soon_tasks(within_days=int(within), limit=20) or []
    except Exception:
        due_soon = []

    if not due_soon:
        st.success(f"Nimic urgent in urmatoarele {within} zile.")
        return

    overdue = [t for t in due_soon if (t.get("days_until_due") or 0) < 0]
    upcoming = [t for t in due_soon if (t.get("days_until_due") or 0) >= 0]

    if overdue:
        st.error(f"{len(overdue)} task-uri expirate")
        for task in overdue:
            _render_reminder_card(apci_system, task, expired=True)

    if upcoming:
        st.warning(f"{len(upcoming)} task-uri scadente in curand")
        for task in upcoming:
            _render_reminder_card(apci_system, task, expired=False)


def _render_reminder_card(apci_system, task: Dict[str, Any], expired: bool) -> None:
    days = task.get("days_until_due") or 0.0
    if expired:
        when = f"expirat de {abs(int(days))} zile"
    elif days < 1:
        when = "scadent astazi"
    else:
        when = f"scadent in {int(days)} zile"

    topic = task.get("topic_collection") or "global"
    priority = task.get("priority", "normal")
    badge = PRIORITY_BADGE.get(priority, "")

    with st.container(border=True):
        cols = st.columns([5, 1, 1])
        with cols[0]:
            st.markdown(f"{badge} **{task.get('title', 'fara titlu')}**")
            st.caption(f"{when} · {topic} · {task.get('status', 'open')}")
        with cols[1]:
            if st.button("Done", key=f"rem_done_{task['id']}", use_container_width=True):
                apci_system.update_task_status(task["id"], "done")
                st.rerun()
        with cols[2]:
            if st.button("Snooze", key=f"rem_snooze_{task['id']}", use_container_width=True,
                         help="Muta scadenta cu +7 zile"):
                _snooze_task(apci_system, task, days=7)
                st.rerun()


def _snooze_task(apci_system, task: Dict[str, Any], days: int) -> None:
    """Avanseaza due_at cu N zile fata de azi."""
    from datetime import timedelta
    new_due = (datetime.now() + timedelta(days=days)).date().isoformat()
    apci_system.update_task(task["id"], due_at=new_due)


# ---------------------------------------------------------------------------
# Quick capture
# ---------------------------------------------------------------------------


def _render_quick_capture(apci_system) -> None:
    st.subheader("Adauga task rapid")

    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    notebook_options = ["(global)"] + sorted(
        name for name in collections.keys() if not is_general_collection(name)
    )

    with st.form("task_quick_capture", clear_on_submit=True):
        col_title, col_priority = st.columns([4, 1])
        with col_title:
            title = st.text_input(
                "Titlu",
                placeholder="ex: review draft licenta capitolul 3",
                label_visibility="collapsed",
                key="qc_title",
            )
        with col_priority:
            priority = st.selectbox(
                "Prioritate",
                options=PRIORITY_OPTIONS,
                index=1,
                key="qc_priority",
                label_visibility="collapsed",
            )

        col_topic, col_due = st.columns([2, 2])
        with col_topic:
            topic = st.selectbox(
                "Notebook",
                options=notebook_options,
                key="qc_topic",
            )
        with col_due:
            due = st.date_input(
                "Scadenta (optional)",
                value=None,
                key="qc_due",
            )

        details = st.text_area(
            "Detalii (optional)",
            placeholder="Note suplimentare...",
            key="qc_details",
            height=80,
        )

        submitted = st.form_submit_button("Adauga", type="primary", use_container_width=True)
        if submitted:
            clean_title = (title or "").strip()
            if not clean_title:
                st.warning("Titlul este obligatoriu.")
                return

            topic_value = "" if topic == "(global)" else topic
            due_value: Optional[str] = due.isoformat() if isinstance(due, date) else None

            new_id = apci_system.create_task_manual(
                title=clean_title,
                details=(details or "").strip(),
                topic_collection=topic_value,
                priority=priority,
                due_at=due_value,
            )
            if new_id:
                st.success(f"Task adaugat (#{new_id}).")
                st.rerun()
            else:
                st.error("Nu am putut salva task-ul.")


# ---------------------------------------------------------------------------
# Kanban
# ---------------------------------------------------------------------------


def _render_kanban(apci_system) -> None:
    st.subheader("Kanban")

    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    notebook_filter_options = ["Toate"] + sorted(
        name for name in collections.keys() if not is_general_collection(name)
    ) + ["(global)"]

    col_topic, col_priority = st.columns(2)
    with col_topic:
        topic_filter = st.selectbox(
            "Filtreaza dupa notebook",
            options=notebook_filter_options,
            key="kanban_topic_filter",
        )
    with col_priority:
        priority_filter = st.multiselect(
            "Filtreaza dupa prioritate",
            options=PRIORITY_OPTIONS,
            default=[],
            placeholder="Toate prioritatile",
            key="kanban_priority_filter",
        )

    try:
        all_tasks = apci_system.list_all_tasks(limit=500) or []
    except Exception:
        all_tasks = []

    filtered = _apply_kanban_filters(all_tasks, topic_filter, priority_filter)

    if not filtered:
        st.info("Nu exista task-uri care sa corespunda filtrelor.")
        return

    cols = st.columns(len(STATUS_COLUMNS))
    for col, (status_key, status_label) in zip(cols, STATUS_COLUMNS):
        with col:
            bucket = [t for t in filtered if t.get("status") == status_key]
            st.markdown(f"### {status_label} ({len(bucket)})")
            for task in bucket:
                _render_task_card(apci_system, task)


def _apply_kanban_filters(
    tasks: List[Dict[str, Any]],
    topic_filter: str,
    priority_filter: List[str],
) -> List[Dict[str, Any]]:
    filtered = tasks
    if topic_filter == "(global)":
        filtered = [t for t in filtered if not (t.get("topic_collection") or "").strip()]
    elif topic_filter != "Toate":
        filtered = [t for t in filtered if (t.get("topic_collection") or "") == topic_filter]
    if priority_filter:
        filtered = [t for t in filtered if t.get("priority", "normal") in priority_filter]
    return filtered


def _render_task_card(apci_system, task: Dict[str, Any]) -> None:
    task_id = task["id"]
    status = task.get("status", "open")
    priority = task.get("priority", "normal")
    badge = PRIORITY_BADGE.get(priority, "")
    topic = task.get("topic_collection") or "global"
    due_at = task.get("due_at") or ""
    edit_key = f"edit_task_{task_id}"
    confirm_del_key = f"del_confirm_{task_id}"

    with st.container(border=True):
        if st.session_state.get(edit_key):
            _render_task_edit_form(apci_system, task, edit_key)
            return

        st.markdown(f"{badge} **{task.get('title', 'fara titlu')}**")
        meta_parts = [topic]
        if due_at:
            meta_parts.append(f"scadent {due_at[:10]}")
        st.caption(" · ".join(meta_parts))

        details = task.get("details") or ""
        if details:
            st.caption(details[:160] + ("..." if len(details) > 160 else ""))

        # Action buttons: status transitions + edit + delete
        action_cols = st.columns(4)
        next_status = _next_status_for(status)
        with action_cols[0]:
            if next_status and st.button(
                _next_status_label(next_status),
                key=f"advance_{task_id}",
                use_container_width=True,
            ):
                apci_system.update_task_status(task_id, next_status)
                st.rerun()
        with action_cols[1]:
            if status != "open" and st.button("→ Open", key=f"reopen_{task_id}", use_container_width=True):
                apci_system.update_task_status(task_id, "open")
                st.rerun()
        with action_cols[2]:
            if st.button("Edit", key=f"edit_btn_{task_id}", use_container_width=True):
                st.session_state[edit_key] = True
                st.rerun()
        with action_cols[3]:
            if st.session_state.get(confirm_del_key):
                if st.button("Confirma", key=f"del_ok_{task_id}", use_container_width=True, type="secondary"):
                    apci_system.delete_task(task_id)
                    st.session_state.pop(confirm_del_key, None)
                    st.rerun()
            else:
                if st.button("Sterge", key=f"del_{task_id}", use_container_width=True):
                    st.session_state[confirm_del_key] = True
                    st.rerun()


def _next_status_for(status: str) -> Optional[str]:
    if status == "open":
        return "in_progress"
    if status == "in_progress":
        return "done"
    return None


def _next_status_label(next_status: str) -> str:
    return {"in_progress": "→ Start", "done": "→ Done"}.get(next_status, next_status)


def _render_task_edit_form(apci_system, task: Dict[str, Any], edit_key: str) -> None:
    task_id = task["id"]
    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    topic_options = ["(global)"] + sorted(
        name for name in collections.keys() if not is_general_collection(name)
    )

    current_topic = task.get("topic_collection") or ""
    topic_index = 0
    if current_topic and current_topic in topic_options:
        topic_index = topic_options.index(current_topic)

    current_priority = task.get("priority", "normal")
    priority_index = PRIORITY_OPTIONS.index(current_priority) if current_priority in PRIORITY_OPTIONS else 1

    current_due = task.get("due_at")
    parsed_due: Optional[date] = None
    if current_due:
        try:
            parsed_due = datetime.fromisoformat(current_due[:10]).date()
        except Exception:
            parsed_due = None

    with st.form(f"edit_task_form_{task_id}"):
        new_title = st.text_input("Titlu", value=task.get("title", ""), key=f"f_title_{task_id}")
        new_details = st.text_area("Detalii", value=task.get("details", ""), key=f"f_details_{task_id}", height=80)
        col_topic, col_pri, col_due = st.columns(3)
        with col_topic:
            new_topic = st.selectbox("Notebook", options=topic_options, index=topic_index, key=f"f_topic_{task_id}")
        with col_pri:
            new_priority = st.selectbox("Prioritate", options=PRIORITY_OPTIONS, index=priority_index, key=f"f_pri_{task_id}")
        with col_due:
            new_due = st.date_input("Scadenta", value=parsed_due, key=f"f_due_{task_id}")

        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("Salveaza", type="primary", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Anuleaza", use_container_width=True)

        if save:
            topic_value = "" if new_topic == "(global)" else new_topic
            due_value = new_due.isoformat() if isinstance(new_due, date) else ""
            ok = apci_system.update_task(
                task_id=task_id,
                title=new_title,
                details=new_details,
                priority=new_priority,
                due_at=due_value,
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
