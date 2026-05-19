"""Vederea Timeline: agregare cronologica a evenimentelor de cunostinte.

Sentimentul de "memorie care curge" — vezi ce s-a intamplat in Second Brain
in ultima saptamana, luna, etc.: note, decizii, task-uri, surse noi, chat-uri.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from views._shared import (
    GENERAL_COLLECTION_NAME,
    is_general_collection,
    normalize_collection_name,
    system_ready,
)


_EVENT_TYPE_LABELS = {
    "note_created": "💡 Note create",
    "note_updated": "✏️ Note editate",
    "decision": "⚙️ Decizii",
    "task_created": "📌 Task-uri create",
    "task_done": "✅ Task-uri inchise",
    "document_added": "📄 Documente adaugate",
    "chat_started": "💬 Chat-uri",
}

_WINDOW_OPTIONS = [
    ("Ultima zi", 1),
    ("Ultimele 3 zile", 3),
    ("Ultimele 7 zile", 7),
    ("Ultimele 14 zile", 14),
    ("Ultimele 30 zile", 30),
    ("Ultimele 90 zile", 90),
]


def render_timeline() -> None:
    """Punct de intrare pentru tab-ul Timeline."""
    st.title("Timeline")
    st.caption(
        "Cronologic: tot ce ai capturat, decis, finalizat si invatat in ultima perioada. "
        "Memoria ta personala vazuta in flux, nu fragmentata."
    )

    if not system_ready():
        st.info("Configureaza API key-ul in bara laterala pentru a folosi timeline-ul.")
        return

    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        st.error("Sistem indisponibil.")
        return

    repo = getattr(apci_system, "repository", None)
    if not repo or not getattr(repo, "enabled", False):
        st.warning("Timeline-ul necesita Postgres conectat.")
        return

    days, topic, selected_types = _render_filters()

    events = apci_system.get_timeline_events(
        topic_collection=topic,
        days=days,
        event_types=list(selected_types) if selected_types else None,
        limit=300,
    )

    st.divider()
    _render_summary_metrics(events)
    st.divider()

    if not events:
        st.info("Niciun eveniment in fereastra selectata. Schimba filtrele sau adauga note/documente.")
        return

    _render_timeline_groups(events)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _available_topics() -> List[str]:
    """Lista colectii pentru filtrul de topic."""
    document_manager = st.session_state.get("document_manager")
    topics = [GENERAL_COLLECTION_NAME]
    if document_manager:
        cols = document_manager.get_collections() or {}
        topics.extend(sorted(
            name for name in cols.keys() if not is_general_collection(name)
        ))
    return topics


def _render_filters() -> tuple[int, Optional[str], set[str]]:
    """Randeaza filtrele si returneaza (days, topic_or_None, selected_types_set)."""
    col_window, col_topic = st.columns([1, 1])

    with col_window:
        label_to_days = dict(_WINDOW_OPTIONS)
        labels = [opt[0] for opt in _WINDOW_OPTIONS]
        choice = st.selectbox(
            "Fereastra",
            options=labels,
            index=2,  # default: 7 zile
            key="timeline_window",
            label_visibility="collapsed",
        )
        days = label_to_days[choice]

    with col_topic:
        topics = ["(toate topicurile)"] + _available_topics()
        topic_choice = st.selectbox(
            "Topic",
            options=topics,
            index=0,
            key="timeline_topic_filter",
            label_visibility="collapsed",
        )
        if topic_choice == "(toate topicurile)":
            topic: Optional[str] = None
        elif is_general_collection(topic_choice):
            topic = ""
        else:
            topic = normalize_collection_name(topic_choice)

    # Multiselect pentru tipuri de evenimente
    all_types = list(_EVENT_TYPE_LABELS.keys())
    type_labels = [_EVENT_TYPE_LABELS[t] for t in all_types]
    default_labels = type_labels  # toate selectate initial
    selected_labels = st.multiselect(
        "Tipuri de evenimente",
        options=type_labels,
        default=default_labels,
        key="timeline_type_filter",
    )
    label_to_type = {v: k for k, v in _EVENT_TYPE_LABELS.items()}
    selected_types = {label_to_type[label] for label in selected_labels if label in label_to_type}

    return days, topic, selected_types


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def _render_summary_metrics(events: List[Dict[str, Any]]) -> None:
    """4 metrici pentru fereastra selectata."""
    counts: Dict[str, int] = {}
    for e in events:
        t = e.get("type", "")
        counts[t] = counts.get(t, 0) + 1

    cols = st.columns(4)
    cols[0].metric("Total evenimente", len(events))
    cols[1].metric(
        "Note (create + editate)",
        counts.get("note_created", 0) + counts.get("note_updated", 0),
    )
    cols[2].metric("Decizii", counts.get("decision", 0))
    cols[3].metric(
        "Task-uri (create + inchise)",
        counts.get("task_created", 0) + counts.get("task_done", 0),
    )


# ---------------------------------------------------------------------------
# Timeline rendering (grouped by day)
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Python iso8601 parser: poate accepta sau nu "Z" sufix
        ts_clean = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(ts_clean)
    except Exception:
        return None


def _human_date_label(date_iso: str, today_iso: str, yesterday_iso: str) -> str:
    if date_iso == today_iso:
        return f"📅 {date_iso} (azi)"
    if date_iso == yesterday_iso:
        return f"📅 {date_iso} (ieri)"
    return f"📅 {date_iso}"


def _render_timeline_groups(events: List[Dict[str, Any]]) -> None:
    """Grupeaza evenimentele pe zile si randeaza fiecare card."""
    from datetime import date as _date, timedelta as _td

    today = _date.today()
    today_iso = today.isoformat()
    yesterday_iso = (today - _td(days=1)).isoformat()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ts = e.get("timestamp", "")
        date_part = ts[:10] if len(ts) >= 10 else "unknown"
        grouped.setdefault(date_part, []).append(e)

    # Sortare zile descrescator
    for day_iso in sorted(grouped.keys(), reverse=True):
        day_events = grouped[day_iso]
        st.markdown(f"### {_human_date_label(day_iso, today_iso, yesterday_iso)}")
        st.caption(f"{len(day_events)} eveniment(e)")
        for event in day_events:
            _render_event_card(event)
        st.write("")  # spacer between days


def _render_event_card(event: Dict[str, Any]) -> None:
    icon = event.get("icon", "•")
    event_type = event.get("type", "")
    title = event.get("title", "")
    preview = event.get("preview", "")
    topic = event.get("topic_collection", "global")
    ts = event.get("timestamp", "")
    time_part = ts[11:16] if len(ts) >= 16 else ""

    type_label = {
        "note_created": "Nota noua",
        "note_updated": "Nota editata",
        "decision": "Decizie",
        "task_created": "Task nou",
        "task_done": "Task inchis",
        "document_added": "Document adaugat",
        "chat_started": "Chat",
    }.get(event_type, event_type)

    meta = event.get("metadata") or {}

    with st.container(border=True):
        header_parts = [f"**{icon} {time_part}**", f"_{type_label}_", f"**{title}**"]
        st.markdown(" · ".join(p for p in header_parts if p))

        badge_parts = [f"📌 {topic}"]
        if event_type == "decision":
            conf = float(meta.get("confidence", 0.0) or 0.0)
            badge_parts.append(f"conf={conf:.2f}")
        elif event_type in ("task_created", "task_done"):
            prio = meta.get("priority", "normal")
            badge_parts.append(f"prio: {prio}")
            if event_type == "task_created":
                badge_parts.append(f"status: {meta.get('status', 'open')}")
        elif event_type == "note_created":
            tags = meta.get("tags") or []
            if tags:
                badge_parts.append(" ".join(f"`{t}`" for t in tags))
        elif event_type == "document_added":
            badge_parts.append("indexat" if meta.get("indexed") else "neindexat")
        elif event_type == "chat_started":
            msg_count = int(meta.get("message_count") or 0)
            badge_parts.append(f"{msg_count} mesaje")
        st.caption(" · ".join(badge_parts))

        if preview:
            display_preview = preview[:300] + ("..." if len(preview) > 300 else "")
            st.write(display_preview)
