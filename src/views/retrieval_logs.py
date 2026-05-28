"""Vederea Retrieval Logs: observability pentru pipeline-ul de retrieval.

- Stats agregate (avg, p50, p95) pe fereastra configurabila
- Distribuie pe route (vector / hybrid / external)
- Tabel cu ultimele query-uri si latenta lor
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from views._shared import system_ready


ROUTE_OPTIONS = ["Toate", "vector", "memory", "hybrid", "external", "synthesis"]


def render_retrieval_logs() -> None:
    """Punct de intrare pentru tab-ul Logs."""
    st.title("Retrieval Logs")
    st.caption("Observability pentru pipeline-ul de retrieval — calitate si latenta.")

    if not system_ready():
        st.info("Sistemul CerebrumAI nu este initializat.")
        return

    apci_system = st.session_state.get("apci_system")
    repo = getattr(apci_system, "repository", None)
    if not repo or not getattr(repo, "enabled", False):
        st.warning("Retrieval logs necesita Postgres conectat.")
        return

    _render_stats_section(apci_system)
    st.divider()
    _render_logs_table(apci_system)


def _render_stats_section(apci_system) -> None:
    col_label, col_window = st.columns([3, 1])
    with col_label:
        st.subheader("Statistici agregate")
    with col_window:
        window = st.selectbox(
            "Zile",
            options=[1, 3, 7, 14, 30],
            index=2,
            key="logs_stats_window",
            label_visibility="collapsed",
        )

    try:
        stats = apci_system.get_retrieval_stats(days=int(window)) or {}
    except Exception:
        stats = {}

    if not stats or stats.get("total", 0) == 0:
        st.info(f"Niciun query in ultimele {window} zile.")
        return

    cols = st.columns(4)
    cols[0].metric("Total queries", stats.get("total", 0))
    cols[1].metric("Latenta avg (ms)", f"{stats.get('latency_ms_avg', 0):.0f}")
    cols[2].metric("Latenta p50 (ms)", f"{stats.get('latency_ms_p50', 0):.0f}")
    cols[3].metric("Latenta p95 (ms)", f"{stats.get('latency_ms_p95', 0):.0f}")

    by_route: Dict[str, Dict[str, Any]] = stats.get("by_route", {}) or {}
    if by_route:
        st.markdown("**Distributie pe route**")
        cols_routes = st.columns(len(by_route))
        for col, (route, info) in zip(cols_routes, by_route.items()):
            with col:
                st.metric(
                    route,
                    info.get("count", 0),
                    delta=f"avg {info.get('latency_ms_avg', 0):.0f} ms",
                    delta_color="off",
                )


def _render_logs_table(apci_system) -> None:
    col_label, col_filter, col_limit = st.columns([2, 1, 1])
    with col_label:
        st.subheader("Ultimele query-uri")
    with col_filter:
        route = st.selectbox(
            "Route",
            options=ROUTE_OPTIONS,
            index=0,
            key="logs_route_filter",
            label_visibility="collapsed",
        )
    with col_limit:
        limit = st.selectbox(
            "Limit",
            options=[25, 50, 100, 200],
            index=2,
            key="logs_limit",
            label_visibility="collapsed",
        )

    route_filter = None if route == "Toate" else route

    try:
        logs: List[Dict[str, Any]] = apci_system.list_retrieval_logs(
            limit=int(limit),
            route_filter=route_filter,
        ) or []
    except Exception:
        logs = []

    if not logs:
        st.info("Niciun log de afisat.")
        return

    rows = []
    for log in logs:
        metrics = log.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "id": log.get("id"),
                "created_at": (log.get("created_at") or "")[:19].replace("T", " "),
                "route": log.get("route_used", ""),
                "latency_ms": round(float(log.get("latency_ms") or 0.0), 1),
                "question": (log.get("question") or "")[:120],
                "vector_hits": metrics.get("vector_hits", ""),
                "memory_hits": metrics.get("memory_hits", ""),
                "note_hits": metrics.get("note_hits", ""),
                "graph_hits": metrics.get("graph_hits", ""),
                "web": metrics.get("used_web", ""),
                "hyde": metrics.get("used_hyde", ""),
                "rerank": metrics.get("used_reranker", ""),
                "bad_citations": ", ".join(metrics.get("citation_invalid", []) or []),
                "answer_origin": metrics.get("answer_origin", ""),
                "error": metrics.get("error", ""),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
