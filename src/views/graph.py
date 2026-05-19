"""Vederea Knowledge Graph - vizualizare interactiva a conceptelor si relatiilor.

Functionalitati E3:
  - vizualizare Plotly a grafului Neo4j (noduri = concepte, muchii = relatii)
  - filtre per tip relatie si notebook/colectie
  - mod 'Doar contradictii'
  - fallback elegant cand Neo4j nu este disponibil
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import streamlit as st

RELATION_COLORS = {
    "RELATED_TO": "#6c8ebf",
    "DEPENDS_ON": "#82b366",
    "CONTRADICTS": "#e06c75",
    "DERIVED_FROM": "#d79b00",
}
RELATION_LABELS = {
    "RELATED_TO": "Legat de",
    "DEPENDS_ON": "Depinde de",
    "CONTRADICTS": "Contrazice",
    "DERIVED_FROM": "Derivat din",
}
ALL_RELATION_TYPES = list(RELATION_COLORS.keys())


def render_graph() -> None:
    """Punct de intrare pentru tab-ul Knowledge Graph."""
    st.title("Knowledge Graph")
    st.caption(
        "Harta vie a tot ce stii: concepte, documente, relatii si contradictii."
    )

    apci_system = st.session_state.get("apci_system")
    graph_stats = _try_get_graph_stats(apci_system)

    cols = st.columns(4)
    cols[0].metric("Concepte", graph_stats.get("concepts", "—"))
    cols[1].metric("Relatii", graph_stats.get("relations", "—"))
    cols[2].metric("Contradictii", graph_stats.get("contradictions", "—"))
    cols[3].metric("Documente in graf", graph_stats.get("documents", "—"))

    st.divider()

    neo4j_ok = _neo4j_enabled(apci_system)
    if not neo4j_ok:
        _render_no_graph_placeholder()
        return

    nodes_count = int(graph_stats.get("concepts") or 0)
    if nodes_count == 0:
        st.info(
            "Graful este gol. Incarca si proceseaza documente in tab-ul Notebooks "
            "pentru a popula graful de cunostinte."
        )
        return

    _render_graph_controls(apci_system)


# ---------------------------------------------------------------------------
# Controale si vizualizare
# ---------------------------------------------------------------------------


def _render_graph_controls(apci_system) -> None:
    ctrl_left, ctrl_right = st.columns([3, 1])

    with ctrl_left:
        selected_relations = st.multiselect(
            "Tipuri de relatii afisate",
            options=ALL_RELATION_TYPES,
            default=ALL_RELATION_TYPES,
            format_func=lambda r: RELATION_LABELS.get(r, r),
            key="graph_relation_filter",
        )

    with ctrl_right:
        only_contradictions = st.checkbox(
            "Doar contradictii",
            value=False,
            key="graph_only_contradictions",
            help="Afiseaza doar relatiile CONTRADICTS pentru a identifica tensiuni.",
        )

    document_manager = st.session_state.get("document_manager")
    collections = document_manager.get_collections() if document_manager else {}
    notebook_names = sorted(
        name for name in collections.keys()
        if name.lower() not in {"general", "default"}
    )
    collection_options = ["Toate notebook-urile"] + notebook_names
    selected_collection = st.selectbox(
        "Filtreaza dupa notebook",
        options=collection_options,
        index=0,
        key="graph_collection_filter",
    )

    limit = st.slider(
        "Numar maxim de concepte",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
        key="graph_node_limit",
    )

    if only_contradictions:
        active_relations = ["CONTRADICTS"]
    elif selected_relations:
        active_relations = selected_relations
    else:
        active_relations = ALL_RELATION_TYPES

    collection_filter: Optional[str] = (
        None if selected_collection == "Toate notebook-urile" else selected_collection
    )

    with st.spinner("Se incarca graful..."):
        viz_data = apci_system.get_graph_viz_data(
            limit_nodes=limit,
            relation_types=active_relations,
            collection_filter=collection_filter,
        )

    nodes = viz_data.get("nodes", [])
    edges = viz_data.get("edges", [])

    if not nodes:
        st.info("Nu exista concepte care sa corespunda filtrelor selectate.")
        return

    st.caption(f"{len(nodes)} concepte · {len(edges)} relatii vizualizate")
    _render_plotly_graph(nodes, edges)
    _render_legend()


def render_mini_graph(collection_filter: str) -> None:
    """Mini Knowledge Graph scoped pe un notebook specific."""
    apci_system = st.session_state.get("apci_system")
    if not _neo4j_enabled(apci_system):
        st.caption("Neo4j nu este disponibil pentru vizualizare graf.")
        return

    col_limit, col_rel = st.columns([1, 2])
    with col_limit:
        limit = st.slider(
            "Concepte max",
            min_value=10,
            max_value=100,
            value=40,
            step=10,
            key=f"mini_graph_limit_{collection_filter}",
            label_visibility="collapsed",
        )
    with col_rel:
        selected_relations = st.multiselect(
            "Relatii afisate",
            options=ALL_RELATION_TYPES,
            default=ALL_RELATION_TYPES,
            format_func=lambda r: RELATION_LABELS.get(r, r),
            key=f"mini_graph_rels_{collection_filter}",
            label_visibility="collapsed",
        )

    active_relations = selected_relations or ALL_RELATION_TYPES

    with st.spinner("Se incarca graful..."):
        viz_data = apci_system.get_graph_viz_data(
            limit_nodes=limit,
            relation_types=active_relations,
            collection_filter=collection_filter,
        )

    nodes = viz_data.get("nodes", [])
    edges = viz_data.get("edges", [])

    if not nodes:
        st.info(
            "Niciun concept in graful acestui notebook. "
            "Asigura-te ca documentele sunt indexate si Neo4j este activ."
        )
        return

    st.caption(f"{len(nodes)} concepte · {len(edges)} relatii")
    _render_plotly_graph(nodes, edges, height=380)


def _render_plotly_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    height: int = 600,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly nu este instalat. Ruleaza: pip install plotly")
        return

    pos = _spring_layout(nodes, edges)

    fig = go.Figure()

    rel_edge_map: Dict[str, List] = {r: [] for r in ALL_RELATION_TYPES}
    for edge in edges:
        rel_type = edge.get("type", "RELATED_TO")
        if rel_type not in rel_edge_map:
            rel_edge_map[rel_type] = []
        rel_edge_map[rel_type].append(edge)

    for rel_type, rel_edges in rel_edge_map.items():
        if not rel_edges:
            continue
        color = RELATION_COLORS.get(rel_type, "#aaa")
        ex: List[Any] = []
        ey: List[Any] = []
        for edge in rel_edges:
            src = edge["source"]
            tgt = edge["target"]
            if src not in pos or tgt not in pos:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        if ex:
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(width=1, color=color),
                    hoverinfo="none",
                    showlegend=True,
                    name=RELATION_LABELS.get(rel_type, rel_type),
                )
            )

    node_x = [pos[n["id"]][0] for n in nodes if n["id"] in pos]
    node_y = [pos[n["id"]][1] for n in nodes if n["id"] in pos]
    node_text = [n["label"] for n in nodes if n["id"] in pos]
    node_size = [n.get("size", 10) for n in nodes if n["id"] in pos]
    node_collection = [n.get("collection", "general") for n in nodes if n["id"] in pos]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_size,
                color="#4a90d9",
                line=dict(width=1, color="#2c5f8a"),
                opacity=0.85,
            ),
            text=node_text,
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="<b>%{text}</b><br>Notebook: %{customdata}<extra></extra>",
            customdata=node_collection,
            name="Concepte",
        )
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_legend() -> None:
    with st.expander("Legenda relatii", expanded=False):
        cols = st.columns(len(RELATION_COLORS))
        for col, (rel_type, color) in zip(cols, RELATION_COLORS.items()):
            col.markdown(
                f"<span style='color:{color}'>■</span> **{RELATION_LABELS[rel_type]}**",
                unsafe_allow_html=True,
            )


def _render_no_graph_placeholder() -> None:
    with st.container(border=True):
        st.subheader("Neo4j nu este disponibil")
        st.info(
            "Vizualizarea interactiva necesita o conexiune Neo4j activa.\n\n"
            "Configureaza variabilele de mediu in fisierul `.env`:\n"
            "- `NEO4J_URI` — adresa serverului (ex: bolt://localhost:7687)\n"
            "- `NEO4J_USER` — utilizator (ex: neo4j)\n"
            "- `NEO4J_PASSWORD` — parola\n\n"
            "Dupa conectare, documentele indexate vor popula automat graful de cunostinte."
        )


# ---------------------------------------------------------------------------
# Layout algoritm Fruchterman-Reingold (fara dependinte externe)
# ---------------------------------------------------------------------------


def _spring_layout(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    iterations: int = 50,
    k: float = 1.0,
) -> Dict[str, tuple]:
    """Force-directed layout — Fruchterman-Reingold simplificat."""
    if not nodes:
        return {}

    rng = random.Random(42)
    ids = [n["id"] for n in nodes]
    pos = {nid: (rng.uniform(-1, 1), rng.uniform(-1, 1)) for nid in ids}

    adj: Dict[str, set] = {nid: set() for nid in ids}
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if src in adj and tgt in adj:
            adj[src].add(tgt)
            adj[tgt].add(src)

    k_val = math.sqrt(len(ids) * k * k / max(len(ids), 1))

    temp = 1.0
    cooling = temp / (iterations + 1)

    for _ in range(iterations):
        disp: Dict[str, List[float]] = {nid: [0.0, 0.0] for nid in ids}

        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                d = math.sqrt(dx * dx + dy * dy) or 0.01
                rep = (k_val * k_val) / d / d
                disp[u][0] += dx * rep
                disp[u][1] += dy * rep
                disp[v][0] -= dx * rep
                disp[v][1] -= dy * rep

        for u in ids:
            for v in adj[u]:
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                d = math.sqrt(dx * dx + dy * dy) or 0.01
                att = (d * d) / k_val / d
                disp[u][0] -= dx * att
                disp[u][1] -= dy * att

        for nid in ids:
            dx, dy = disp[nid]
            d = math.sqrt(dx * dx + dy * dy) or 0.01
            scale = min(d, temp) / d
            pos[nid] = (pos[nid][0] + dx * scale, pos[nid][1] + dy * scale)

        temp -= cooling

    return {nid: (x, y) for nid, (x, y) in pos.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_get_graph_stats(apci_system) -> dict:
    if not apci_system:
        return {}
    if hasattr(apci_system, "get_graph_stats"):
        try:
            stats = apci_system.get_graph_stats() or {}
            return {
                "concepts": stats.get("concepts", "—"),
                "relations": stats.get("relations", "—"),
                "contradictions": stats.get("contradictions", "—"),
                "documents": stats.get("documents", "—"),
            }
        except Exception:
            return {}
    return {}


def _neo4j_enabled(apci_system) -> bool:
    if not apci_system:
        return False
    neo4j_client = getattr(apci_system, "neo4j_client", None)
    return bool(neo4j_client and getattr(neo4j_client, "enabled", False))
