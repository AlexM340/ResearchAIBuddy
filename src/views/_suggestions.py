"""Mecanismul de sugestii de surse bazat pe directia conversatiei."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from views._documents import add_web_url_as_document


def _scope_key(topic_collection: Optional[str]) -> str:
    return topic_collection.strip() if topic_collection and topic_collection.strip() else "global"


def render_source_suggestions(topic_collection: Optional[str] = None) -> None:
    """Expander cu sugestii de surse web relevante pentru conversatia curenta.

    Se poate apela din Second Brain (topic_collection=None) sau din orice notebook
    (topic_collection='NumeNotebook'). Sugestiile sunt generate via Gemini + Tavily si
    pot fi adaugate direct in biblioteca si indexate.
    """
    apci_system = st.session_state.get("apci_system")
    if not apci_system:
        return

    web_available = bool(getattr(apci_system, "web_search_available", False))
    scope = _scope_key(topic_collection)
    sugg_key = f"sugg_data_{scope}"
    added_key = f"sugg_added_{scope}"

    with st.expander("Surse sugerate", expanded=False):
        if not web_available:
            st.info(
                "Configureaza cheia API Tavily in bara laterala pentru a activa sugestiile de surse."
            )
            return

        scope_label = f"'{topic_collection}'" if topic_collection else "Second Brain (global)"
        st.caption(
            f"Sugestii personalizate pentru {scope_label} — bazate pe directia "
            "conversatiei si interesele tale recente."
        )

        col_btn, col_count = st.columns([3, 1])
        with col_count:
            max_suggestions = st.selectbox(
                "Nr. sugestii",
                options=[3, 5, 8],
                index=1,
                key=f"sugg_count_{scope}",
                label_visibility="collapsed",
            )
        with col_btn:
            generate_clicked = st.button(
                "Genereaza sugestii",
                key=f"sugg_btn_{scope}",
                type="primary",
                use_container_width=True,
            )

        if generate_clicked:
            chat_history = st.session_state.get("chat_history", [])
            with st.spinner("Analizez conversatia si caut resurse..."):
                data = apci_system.suggest_sources(
                    topic_collection=topic_collection or None,
                    recent_chat_history=chat_history,
                    max_suggestions=int(max_suggestions),
                )
            st.session_state[sugg_key] = data
            st.session_state[added_key] = set()

        data: Optional[Dict[str, Any]] = st.session_state.get(sugg_key)
        if data is None:
            return

        if data.get("error"):
            st.warning(data["error"])
            return

        suggestions: List[Dict[str, Any]] = data.get("suggestions", [])
        if not suggestions:
            st.info("Nu s-au gasit sugestii pentru aceasta conversatie.")
            return

        added_urls: set = st.session_state.get(added_key, set())

        queries_used = data.get("queries_used", [])
        if queries_used:
            with st.expander("Interogari folosite", expanded=False):
                for q in queries_used:
                    st.caption(f"• {q}")

        for idx, s in enumerate(suggestions):
            url = s.get("url", "")
            title = s.get("title", url)
            snippet = s.get("snippet", "")
            already_added = url in added_urls

            with st.container(border=True):
                col_info, col_action = st.columns([4, 1])
                with col_info:
                    st.markdown(f"**[{title}]({url})**")
                    if snippet:
                        st.caption(snippet[:200] + ("..." if len(snippet) > 200 else ""))
                with col_action:
                    if already_added:
                        st.success("Adaugat")
                    else:
                        if st.button(
                            "Adauga",
                            key=f"sugg_add_{scope}_{idx}",
                            use_container_width=True,
                            type="secondary",
                        ):
                            target = topic_collection or "general"
                            with st.spinner(f"Fetch si indexare: {title[:40]}..."):
                                res = add_web_url_as_document(url, title, target)
                            if res.get("success"):
                                st.success(res["message"])
                                added_urls.add(url)
                                st.session_state[added_key] = added_urls
                                st.rerun()
                            else:
                                st.error(res["message"])
