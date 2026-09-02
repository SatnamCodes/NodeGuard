"""NodeGuard — Streamlit demo. Reuses the existing data/model pipeline
(no reimplemented graph/feature/model logic). Run with:
    streamlit run src/app/streamlit_app.py
"""

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import torch

from src.data_gen.generate_graph import build_graph
from src.data_gen.node_features import compute_node_features
from src.data_gen.to_pyg import graph_to_pyg_data
from src.eval.error_analysis import FEATURE_NAMES, pattern_type
from src.eval.metrics import confusion_counts
from src.model.gnn import MuleGCN
from src.model.train import DROPOUT, HIDDEN_CHANNELS, f1_score, make_split_masks

BEST_MODEL_PATH = "src/model/mule_gcn_best.pt"

# --- Palette -----------------------------------------------------------
BG = "#0B0F14"
PANEL = "#131A22"
BORDER = "#1E2A35"
TEXT = "#E6EDF3"
RED = "#FF4D4D"
TEAL = "#2DD4BF"
AMBER = "#F2B705"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, .stApp {{
            background-color: {BG};
            color: {TEXT};
            font-family: 'IBM Plex Sans', sans-serif;
        }}

        [data-testid="stSidebar"] {{
            background-color: {PANEL};
            border-right: 1px solid {BORDER};
        }}

        h1, h2, h3, h4 {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 600;
            color: {TEXT};
            letter-spacing: 0.02em;
        }}

        p, li, span, label, div {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}

        code, .stCode, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        [data-testid="stMetric"] {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 12px 16px;
        }}

        [data-testid="stMetricValue"] {{
            color: {TEXT};
        }}

        .ng-caveat {{
            background-color: {PANEL};
            border-left: 3px solid {AMBER};
            color: {AMBER};
            padding: 10px 14px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            border-radius: 2px;
            margin: 8px 0 16px 0;
        }}

        .ng-panel {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }}

        .ng-badge-mule {{
            color: {RED};
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
        }}

        .ng-badge-normal {{
            color: {TEAL};
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
        }}

        .stSelectbox label, .stRadio label {{
            font-family: 'IBM Plex Sans', sans-serif;
            color: {TEXT};
        }}

        [data-testid="stDataFrame"] {{
            font-family: 'IBM Plex Mono', monospace;
        }}

        hr {{
            border-color: {BORDER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_everything():
    graph, mule_ids = build_graph()
    features = compute_node_features(graph)
    data = graph_to_pyg_data(graph, features)

    train_mask, val_mask, test_mask = make_split_masks(data.y)

    model = MuleGCN(in_channels=data.x.shape[1], hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    return {
        "graph": graph,
        "mule_ids": mule_ids,
        "features": features,
        "data": data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "model": model,
        "logits": logits,
        "probs": probs,
        "preds": preds,
    }


def page_overview(state):
    st.title("NodeGuard")
    st.caption("Mule-ring detection over a transaction graph — Razorpay AI Buildathon, Track 2")

    data, preds, test_mask = state["data"], state["preds"], state["test_mask"]
    tp, fp, tn, fn = confusion_counts(preds, data.y, test_mask)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(precision, recall)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1", f"{f1:.3f}")

    st.markdown(
        f'<div class="ng-caveat">⚠ Test set contains only {tp + fn} mule accounts — '
        "treat these numbers as directional, not statistically precise.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ng-panel">
        NodeGuard flags coordinated mule-account rings by modeling accounts as
        nodes and transactions as edges, then running a graph neural network
        (GCN) over that structure instead of scoring each account in isolation.
        A single mule account often looks unremarkable on its own — normal-ish
        transaction count, normal-ish amounts — what gives it away is who it's
        connected to: a chain of accounts passing money along fast, or a
        cluster of ordinary accounts quietly feeding into one collection point.
        A flat per-account model can't see that, because "my neighbor also
        looks weird" isn't a column in a feature table — a GNN can, because it
        passes information along edges.
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_neighborhood_figure(graph, node_id, preds, data):
    in_ids = {u for u, _v in graph.in_edges(node_id)}
    out_ids = {v for _u, v in graph.out_edges(node_id)}
    neighbor_ids = in_ids | out_ids

    sub_nodes = [node_id] + sorted(neighbor_ids)
    sub = nx.DiGraph()
    sub.add_nodes_from(sub_nodes)
    for u in in_ids:
        sub.add_edge(u, node_id)
    for v in out_ids:
        sub.add_edge(node_id, v)

    pos = nx.spring_layout(sub, seed=42)

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.2, color=BORDER),
        hoverinfo="none",
    )

    node_x, node_y, node_color, node_text, node_size, node_line = [], [], [], [], [], []
    for n in sub_nodes:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        pred_label = "MULE" if int(preds[n].item()) == 1 else "normal"
        true_label = "MULE" if int(data.y[n].item()) == 1 else "normal"
        node_color.append(RED if int(preds[n].item()) == 1 else TEAL)
        node_text.append(f"node {n}<br>predicted: {pred_label}<br>true: {true_label}")
        if n == node_id:
            node_size.append(28)
            node_line.append(dict(width=3, color=TEXT))
        else:
            node_size.append(18)
            node_line.append(dict(width=1, color=BORDER))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[str(n) for n in sub_nodes],
        textposition="top center",
        textfont=dict(family="IBM Plex Mono", color=TEXT, size=11),
        hovertext=node_text, hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(
                width=[l["width"] for l in node_line],
                color=[l["color"] for l in node_line],
            ),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=420,
    )
    return fig


def page_account_lookup(state):
    st.title("Account Lookup")

    graph = state["graph"]
    data = state["data"]
    preds = state["preds"]
    probs = state["probs"]
    test_mask = state["test_mask"]

    test_ids = sorted(test_mask.nonzero(as_tuple=True)[0].tolist())
    node_id = st.selectbox("Select a test-set account (node ID)", test_ids)

    pred_label = int(preds[node_id].item())
    true_label = int(data.y[node_id].item())
    confidence = float(probs[node_id, pred_label].item())

    if true_label == 1 and pred_label == 1:
        outcome = "True positive — mule, correctly caught"
    elif true_label == 1 and pred_label == 0:
        outcome = "False negative — mule, missed"
    elif true_label == 0 and pred_label == 1:
        outcome = "False positive — normal account, wrongly flagged"
    else:
        outcome = "True negative — normal account, correctly cleared"

    col1, col2, col3 = st.columns(3)
    with col1:
        badge_class = "ng-badge-mule" if pred_label == 1 else "ng-badge-normal"
        st.markdown(f"**Prediction**<br><span class='{badge_class}'>"
                    f"{'MULE' if pred_label == 1 else 'NORMAL'}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Confidence", f"{confidence:.3f}")
    with col3:
        badge_class = "ng-badge-mule" if true_label == 1 else "ng-badge-normal"
        st.markdown(f"**True label**<br><span class='{badge_class}'>"
                    f"{'MULE' if true_label == 1 else 'NORMAL'}</span>", unsafe_allow_html=True)

    st.markdown(f'<div class="ng-panel">{outcome}</div>', unsafe_allow_html=True)

    st.subheader("Immediate transaction neighborhood")
    st.caption("Red = predicted mule · Teal = predicted normal · Selected node is outlined")
    fig = build_neighborhood_figure(graph, node_id, preds, data)
    st.plotly_chart(fig, use_container_width=True)


def page_error_analysis(state):
    st.title("Error Analysis Summary")

    data, preds, test_mask, graph, features, mule_ids = (
        state["data"], state["preds"], state["test_mask"], state["graph"],
        state["features"], state["mule_ids"],
    )
    test_ids = test_mask.nonzero(as_tuple=True)[0].tolist()

    caught = {"layering": 0, "funnel": 0}
    total = {"layering": 0, "funnel": 0}
    false_positives = []

    for node_id in test_ids:
        true_label = int(data.y[node_id].item())
        pred_label = int(preds[node_id].item())
        pattern = pattern_type(node_id)

        if true_label == 1:
            total[pattern] = total.get(pattern, 0) + 1
            if pred_label == 1:
                caught[pattern] = caught.get(pattern, 0) + 1
        elif true_label == 0 and pred_label == 1:
            false_positives.append(node_id)

    st.subheader("Recall by fraud pattern")
    col1, col2 = st.columns(2)
    layering_recall = caught["layering"] / total["layering"] if total["layering"] else 0.0
    funnel_recall = caught["funnel"] / total["funnel"] if total["funnel"] else 0.0
    col1.metric("Layering-chain recall", f"{layering_recall:.0%}", f"{caught['layering']}/{total['layering']}")
    col2.metric("Funnel recall", f"{funnel_recall:.0%}", f"{caught['funnel']}/{total['funnel']}")
    st.markdown(
        '<div class="ng-panel">Funnel mules are harder to catch: they only look '
        "unusual once the model sees that their <em>neighbor</em> (the collector) "
        "has an abnormal in-degree — that needs 2-hop reasoning, learned from just "
        "6 funnel-collector examples in training.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("False positives")
    if not false_positives:
        st.markdown('<div class="ng-panel">No false positives in this test split.</div>', unsafe_allow_html=True)
    else:
        normal_ids = [n for n in graph.nodes() if not graph.nodes[n]["is_mule"]]
        mule_id_set = set(mule_ids)

        def avg_features(node_ids):
            rows = [features[n] for n in node_ids]
            return [sum(col) / len(col) for col in zip(*rows)]

        normal_avg = avg_features(normal_ids)
        mule_avg = avg_features(mule_id_set)

        for node_id in false_positives:
            vals = features[node_id]
            closer_to_mule = sum(
                1 for i in range(len(FEATURE_NAMES))
                if abs(vals[i] - mule_avg[i]) < abs(vals[i] - normal_avg[i])
            )
            explainable = closer_to_mule >= 3

            in_ids = {u for u, _v in graph.in_edges(node_id)}
            out_ids = {v for _u, v in graph.out_edges(node_id)}
            one_hop_mule = any(graph.nodes[n]["is_mule"] for n in in_ids | out_ids)

            two_hop_mule = False
            if not one_hop_mule:
                for mid in in_ids | out_ids:
                    mid_neighbors = {u for u, _v in graph.in_edges(mid)} | {v for _u, v in graph.out_edges(mid)}
                    if any(graph.nodes[n]["is_mule"] for n in mid_neighbors if n != node_id):
                        two_hop_mule = True
                        break

            if explainable:
                verdict = f"Feature-outlier explainable ({closer_to_mule}/{len(FEATURE_NAMES)} features closer to mule average)"
            elif one_hop_mule or two_hop_mule:
                hop = "1-hop" if one_hop_mule else "2-hop"
                verdict = f"Structurally explainable — {hop} connection to a real mule"
            else:
                verdict = "Unexplained at 1–2 hops — most plausibly the model overgeneralizing from only 14 total mule examples"

            st.markdown(
                f'<div class="ng-panel"><span class="ng-badge-mule">node {node_id}</span> — {verdict}</div>',
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(page_title="NodeGuard", layout="wide", page_icon="🛡")
    inject_css()

    state = load_everything()

    page = st.sidebar.radio("NAVIGATION", ["Overview", "Account Lookup", "Error Analysis"])

    if page == "Overview":
        page_overview(state)
    elif page == "Account Lookup":
        page_account_lookup(state)
    else:
        page_error_analysis(state)


if __name__ == "__main__":
    main()
