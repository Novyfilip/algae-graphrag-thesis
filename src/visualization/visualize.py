"""
visualize.py

Builds a four-layer interactive diagram of the hybrid retrieval pipeline:

    Query  ->  Chunks  ->  Anchor entities  ->  Expanded neighbors

The diagram is a provenance view: every edge represents a concrete step
that the pipeline took, and hovering on any node or edge surfaces the
details. Layout is manual left-to-right layered, not force-directed,
because the reading direction is semantic.

Public entry point: create_graph_visualization(...)
"""

import plotly.graph_objects as go


# --------------------------------------------------------------------------
# Layer colors. One color per pipeline stage, chosen for contrast on white.
# --------------------------------------------------------------------------
COLOR_QUERY = "#2c3e50"      # dark slate   - the question
COLOR_CHUNK = "#27ae60"      # seaweed green - matches AlgaeBot main color
COLOR_ENTITY = "#f39c12"     # amber        - anchor entities
COLOR_NEIGHBOR = "#c0392b"   # brick red    - graph-expanded neighbors

EDGE_ACTIVE = "rgba(44, 62, 80, 0.8)"     # triplet reached the LLM context
EDGE_DIMMED = "rgba(180, 180, 180, 0.3)"  # triplet was retrieved but filtered


# --------------------------------------------------------------------------
# Layout configuration. X-coordinate per layer, plus vertical spread range.
# --------------------------------------------------------------------------
LAYER_X = {
    "query": 0.0,
    "chunk": 1.0,
    "entity": 2.0,
    "neighbor": 3.0,
}

Y_MIN = 0.0
Y_MAX = 1.0


def truncate(text, n):
    """Shorten a string for display without a jarring hard cut."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= n:
        return text
    return text[:n - 1].rstrip() + "..."


def spread_y(count):
    """
    Distribute `count` items evenly across [Y_MIN, Y_MAX].
    Returns a list of y-coordinates.
    A single item is centered; multiple items are spaced with equal gaps.
    """
    if count == 0:
        return []
    if count == 1:
        return [(Y_MIN + Y_MAX) / 2]
    step = (Y_MAX - Y_MIN) / (count - 1)
    return [Y_MIN + i * step for i in range(count)]


# --------------------------------------------------------------------------
# Step 1: turn the raw inputs into structured layer dictionaries.
#
# Each layer is a list of dicts with a stable key, a display label,
# a hover-text string, and any layer-specific attributes (size, etc.).
# Positions are assigned in a separate step so that layer-building is
# independent of layout.
# --------------------------------------------------------------------------

def build_query_layer(query):
    """The query layer is always a single node."""
    return [{
        "key": "__query__",
        "label": truncate(query, 40),
        "hover": f"<b>Query</b><br>{query}",
    }]


def build_chunk_layer(top_chunks):
    """
    One node per reranked chunk.

    top_chunks is a list of (score, Document) tuples as produced by rerank().
    We extract chunk_id, paper title (if present in metadata), and the
    rerank score for node sizing.
    """
    nodes = []
    for score, doc in top_chunks:
        chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")
        title = doc.metadata.get("title") or doc.metadata.get("paper_title") or ""
        snippet = truncate(doc.page_content, 200)

        if title:
            label = truncate(title, 30)
        else:
            label = truncate(chunk_id, 30)

        hover = (
            f"<b>Chunk</b><br>"
            f"Paper: {title or '(title unavailable)'}<br>"
            f"ID: {chunk_id}<br>"
            f"Rerank score: {score:.3f}<br>"
            f"<br>{snippet}"
        )

        nodes.append({
            "key": chunk_id,
            "label": label,
            "hover": hover,
            "score": float(score),
        })
    return nodes


def build_entity_layer(triplets):
    """
    Anchor entities are the unique subjects across all triplets.

    triplets are 5-tuples: (chunk_id, subject, predicate, object, confidence)

    For each entity we record which chunks mentioned it, so the hover
    tooltip can show provenance.
    """
    entity_to_chunks = {}
    for chunk_id, subject, _predicate, _obj, _conf in triplets:
        entity_to_chunks.setdefault(subject, set()).add(chunk_id)

    nodes = []
    for entity, chunk_ids in entity_to_chunks.items():
        chunk_list = sorted(chunk_ids)
        hover = (
            f"<b>Anchor entity</b><br>"
            f"{entity}<br>"
            f"Mentioned in {len(chunk_list)} chunk(s):<br>"
            + "<br>".join(f"  - {cid}" for cid in chunk_list)
        )
        nodes.append({
            "key": entity,
            "label": truncate(entity, 25),
            "hover": hover,
            "source_chunks": chunk_list,
        })
    return nodes


def build_neighbor_layer(triplets):
    """
    Neighbor nodes are the unique (object) values across all triplets.

    When the same neighbor is reached from multiple anchors, we merge
    into a single node and list all anchors in the hover.
    """
    neighbor_to_anchors = {}
    for _chunk_id, subject, predicate, obj, conf in triplets:
        neighbor_to_anchors.setdefault(obj, []).append((subject, predicate, conf))

    nodes = []
    for neighbor, anchors in neighbor_to_anchors.items():
        anchor_lines = [
            f"  - {subj} -[{pred} {conf:.2f}]-> {neighbor}"
            for subj, pred, conf in anchors
        ]
        hover = (
            f"<b>Expanded neighbor</b><br>"
            f"{neighbor}<br>"
            f"Reached via:<br>"
            + "<br>".join(anchor_lines)
        )
        nodes.append({
            "key": neighbor,
            "label": truncate(neighbor, 25),
            "hover": hover,
        })
    return nodes


# --------------------------------------------------------------------------
# Step 2: assign (x, y) positions to every node.
# --------------------------------------------------------------------------

def assign_positions(query_nodes, chunk_nodes, entity_nodes, neighbor_nodes):
    """
    Returns a dict: node_key -> (x, y).

    Chunk nodes are sorted by rerank score so the most relevant chunks
    sit near the top. Entity and neighbor nodes are left in insertion
    order, which follows the Cypher ORDER BY confidence DESC.
    """
    positions = {}

    for node, y in zip(query_nodes, spread_y(len(query_nodes))):
        positions[node["key"]] = (LAYER_X["query"], y)

    sorted_chunks = sorted(chunk_nodes, key=lambda n: -n["score"])
    for node, y in zip(sorted_chunks, spread_y(len(sorted_chunks))):
        positions[node["key"]] = (LAYER_X["chunk"], y)

    for node, y in zip(entity_nodes, spread_y(len(entity_nodes))):
        positions[node["key"]] = (LAYER_X["entity"], y)

    for node, y in zip(neighbor_nodes, spread_y(len(neighbor_nodes))):
        positions[node["key"]] = (LAYER_X["neighbor"], y)

    return positions


# --------------------------------------------------------------------------
# Step 3: build the edge list. Each edge carries the data needed for
# styling (width, color) and hover text.
# --------------------------------------------------------------------------

def build_edges(query_key, chunk_nodes, triplets, contributing_triplets):
    """
    Returns a list of edge dicts, each with:
        src, dst        - node keys
        width           - float
        color           - rgba string
        hover           - hover text
    """
    edges = []

    # Query -> chunk edges. Width scales with rerank score.
    for chunk in chunk_nodes:
        edges.append({
            "src": query_key,
            "dst": chunk["key"],
            "width": 1 + 3 * chunk["score"],
            "color": EDGE_ACTIVE,
            "hover": f"Rerank score: {chunk['score']:.3f}",
        })

    # Chunk -> entity edges. One per (chunk_id, subject) pair in triplets.
    chunk_entity_pairs = set()
    for chunk_id, subject, _pred, _obj, _conf in triplets:
        chunk_entity_pairs.add((chunk_id, subject))

    for chunk_id, subject in chunk_entity_pairs:
        edges.append({
            "src": chunk_id,
            "dst": subject,
            "width": 1.5,
            "color": EDGE_ACTIVE,
            "hover": f"{subject} mentioned in {chunk_id}",
        })

    # Entity -> neighbor edges. Width scales with confidence.
    # Color depends on whether this triplet reached the LLM context.
    for triplet in triplets:
        chunk_id, subject, predicate, obj, conf = triplet
        # The "contribution" check uses the 4-tuple (without chunk_id)
        # because the same fact can be surfaced by multiple chunks but
        # only contributes once to the LLM context.
        fact_key = (subject, predicate, obj, conf)
        is_contributing = (
            contributing_triplets is None
            or fact_key in contributing_triplets
        )
        edges.append({
            "src": subject,
            "dst": obj,
            "width": 1 + 3 * conf,
            "color": EDGE_ACTIVE if is_contributing else EDGE_DIMMED,
            "hover": f"{subject} -[{predicate} {conf:.2f}]-> {obj}",
        })

    return edges


# --------------------------------------------------------------------------
# Step 4: turn nodes and edges into plotly traces.
# --------------------------------------------------------------------------

def edge_trace(edges, positions):
    """
    Plotly renders edges efficiently as a single Scatter trace with
    None separators between segments. Unfortunately this means all
    edges in one trace must share a width and color. For variable
    styling we produce one Scatter per edge. It is less efficient but
    dramatically simpler than grouping edges by style.
    """
    traces = []
    for edge in edges:
        if edge["src"] not in positions or edge["dst"] not in positions:
            continue
        x0, y0 = positions[edge["src"]]
        x1, y1 = positions[edge["dst"]]
        traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=edge["width"], color=edge["color"]),
            hoverinfo="text",
            hovertext=edge["hover"],
            showlegend=False,
        ))
    return traces


def node_trace(nodes, positions, color, default_size=22, size_key=None, size_scale=30):
    """
    Build a single Scatter trace for one layer of nodes.

    If size_key is provided, node sizes scale by that attribute; otherwise
    all nodes in the layer share default_size.
    """
    xs, ys, labels, hovers, sizes = [], [], [], [], []
    for node in nodes:
        if node["key"] not in positions:
            continue
        x, y = positions[node["key"]]
        xs.append(x)
        ys.append(y)
        labels.append(node["label"])
        hovers.append(node["hover"])
        if size_key is not None:
            sizes.append(default_size + size_scale * node.get(size_key, 0))
        else:
            sizes.append(default_size)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=color,
            line=dict(width=2, color="white"),
        ),
        text=labels,
        textposition="middle right",
        textfont=dict(size=10, color="black"),
        hoverinfo="text",
        hovertext=hovers,
        showlegend=False,
    )


# --------------------------------------------------------------------------
# Step 5: assemble the full figure, including layer labels and the
# counterfactual badge.
# --------------------------------------------------------------------------

def build_layout(query_nodes, chunk_nodes, entity_nodes, neighbor_nodes):
    """Create the Plotly layout dict, including annotations."""
    n_chunks = len(chunk_nodes)
    n_entities = len(entity_nodes)
    n_neighbors = len(neighbor_nodes)

    badge_text = (
        f"{n_chunks} chunks retrieved  "
        f"→  {n_entities} anchor entities  "
        f"→  {n_neighbors} expanded facts"
    )

    layer_labels = [
        dict(text="<b>Query</b>", x=LAYER_X["query"], y=1.08,
             xref="x", yref="y", showarrow=False,
             font=dict(size=12, color=COLOR_QUERY)),
        dict(text="<b>Retrieved chunks</b>", x=LAYER_X["chunk"], y=1.08,
             xref="x", yref="y", showarrow=False,
             font=dict(size=12, color=COLOR_CHUNK)),
        dict(text="<b>Anchor entities</b>", x=LAYER_X["entity"], y=1.08,
             xref="x", yref="y", showarrow=False,
             font=dict(size=12, color=COLOR_ENTITY)),
        dict(text="<b>Graph-expanded neighbors</b>", x=LAYER_X["neighbor"], y=1.08,
             xref="x", yref="y", showarrow=False,
             font=dict(size=12, color=COLOR_NEIGHBOR)),
    ]

    badge = dict(
        text=badge_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=13, color="#555"),
        xanchor="center",
    )

    return go.Layout(
        title=dict(
            text="Hybrid Retrieval: Vector Anchors → Graph Expansion",
            x=0.5,
            font=dict(size=16),
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=60, l=20, r=20, t=80),
        annotations=layer_labels + [badge],
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.3, 3.6],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.1, 1.15],
        ),
        plot_bgcolor="white",
        height=600,
    )


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------

def create_graph_visualization(query, top_chunks, triplets, contributing_triplets=None):
    """
    Build the four-layer pipeline provenance diagram.

    Args:
        query:
            The user's question string.
        top_chunks:
            List of (score, Document) tuples from rerank().
        triplets:
            List of 5-tuples (chunk_id, subject, predicate, object, confidence)
            from expand_from_chunks() with the updated Cypher query.
        contributing_triplets:
            Optional set of 4-tuples (subject, predicate, object, confidence)
            that were actually included in the LLM's context. If None, all
            triplets are treated as contributing (current pipeline behavior).

    Returns:
        A plotly Figure, or None if there is nothing to visualize.
    """
    if not triplets and not top_chunks:
        return None

    query_nodes = build_query_layer(query)
    chunk_nodes = build_chunk_layer(top_chunks)
    entity_nodes = build_entity_layer(triplets)
    neighbor_nodes = build_neighbor_layer(triplets)

    positions = assign_positions(
        query_nodes, chunk_nodes, entity_nodes, neighbor_nodes
    )

    edges = build_edges(
        query_key=query_nodes[0]["key"],
        chunk_nodes=chunk_nodes,
        triplets=triplets,
        contributing_triplets=contributing_triplets,
    )

    edge_traces = edge_trace(edges, positions)

    node_traces = [
        node_trace(query_nodes, positions, COLOR_QUERY, default_size=26),
        node_trace(chunk_nodes, positions, COLOR_CHUNK,
                   default_size=18, size_key="score", size_scale=20),
        node_trace(entity_nodes, positions, COLOR_ENTITY, default_size=20),
        node_trace(neighbor_nodes, positions, COLOR_NEIGHBOR, default_size=18),
    ]

    layout = build_layout(query_nodes, chunk_nodes, entity_nodes, neighbor_nodes)

    return go.Figure(data=edge_traces + node_traces, layout=layout)