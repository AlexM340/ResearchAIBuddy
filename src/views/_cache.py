"""Cached read wrappers for the heavy Second Brain / Notebooks panels.

Streamlit re-runs the entire page script on every interaction *and* tab
switch, and `st.tabs` / `st.expander` execute their body regardless of
whether they are visible or expanded. That means every landing on a page
used to re-issue ~12 DB/Neo4j reads from scratch. These wrappers memoize the
expensive reads with a short TTL so repeated reruns (notably tab switches)
return instantly instead of hitting the database again.

Design notes:
- `_apci_system` is passed with a leading underscore so Streamlit skips
  hashing it (it is unhashable and lives per-session). The underlying data
  is global (single DB), so sharing the cache across sessions is correct.
- Wrappers are thin pass-throughs: call sites keep their existing
  `or []` / `or {}` defaults and try/except, so behaviour is unchanged on
  success and identical on failure (a raised call is not cached).
- Call `invalidate_reads()` after any write so edits reflect immediately.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Short TTL: long enough to collapse a burst of reruns (tab switches, widget
# clicks) into a single query, short enough that data feels fresh even if a
# write path forgets to invalidate.
READ_TTL = 30  # seconds


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def second_brain_status(_apci_system) -> Dict[str, Any]:
    return _apci_system.get_second_brain_status()


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def storage_index_status(_apci_system) -> Dict[str, Any]:
    # Fans out to 3 DB queries (index + migrations + artifact counts); the
    # sidebar renders it on every rerun, so memoizing it matters a lot.
    return _apci_system.get_storage_index_status()


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def chat_sessions(_manager, manager_kind: str) -> List[Dict[str, Any]]:
    # All chat sessions for the active manager. `manager_kind` discriminates
    # the DB repository from the local file manager so their caches don't
    # collide. Invalidated by every chat mutation via invalidate_reads().
    return _manager.list_sessions()


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def due_soon_tasks(_apci_system, within_days: int, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_due_soon_tasks(within_days=within_days, limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def memory_proposals(_apci_system, status: str, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_memory_proposals(status=status, limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def recent_decisions(_apci_system, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_recent_decisions(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def open_tasks(_apci_system, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_open_tasks(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def top_preferences(_apci_system, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_top_preferences(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def aged_decisions(_apci_system, min_age_days: int, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_aged_decisions(min_age_days=min_age_days, limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def contradictions(_apci_system, limit: int, include_dismissed: bool) -> List[Dict[str, Any]]:
    try:
        return _apci_system.get_contradictions(limit=limit, include_dismissed=include_dismissed)
    except TypeError:
        # Backward-compat: older wrapper without include_dismissed.
        return _apci_system.get_contradictions(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def weekly_summary(_apci_system, days: int) -> Dict[str, Any]:
    return _apci_system.get_weekly_summary(days=days)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def all_decisions(_apci_system, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_all_decisions(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def all_preferences(_apci_system, limit: int) -> List[Dict[str, Any]]:
    return _apci_system.list_all_preferences(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def graph_viz(
    _apci_system,
    limit_nodes: int,
    relation_types: Tuple[str, ...],
    collection_filter: Optional[str],
) -> Dict[str, Any]:
    return _apci_system.get_graph_viz_data(
        limit_nodes=limit_nodes,
        relation_types=list(relation_types),
        collection_filter=collection_filter,
    )


# All cached read functions, for bulk invalidation.
_CACHED_READS = (
    second_brain_status,
    storage_index_status,
    chat_sessions,
    due_soon_tasks,
    memory_proposals,
    recent_decisions,
    open_tasks,
    top_preferences,
    aged_decisions,
    contradictions,
    weekly_summary,
    all_decisions,
    all_preferences,
    graph_viz,
)


def invalidate_reads() -> None:
    """Clear all cached reads. Call after any write so edits reflect at once."""
    for fn in _CACHED_READS:
        try:
            fn.clear()
        except Exception:
            # Never let cache housekeeping break a write path.
            pass
