"""Interactive Plotly visualizations for LDA topic modeling."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


COLOR_PALETTE = [
    "#4361EE",  # Vibrant Blue
    "#3A0CA3",  # Deep Indigo
    "#7209B7",  # Purple
    "#F72585",  # Pink
    "#4CC9F0",  # Cyan
    "#06D6A0",  # Emerald
    "#FFD166",  # Amber
    "#EF476F",  # Coral
    "#118AB2",  # Ocean Blue
    "#073B4C",  # Navy
]


def plot_topic_top_words(topics_dict: Dict[int, List[Tuple[str, float]]], topic_names: Dict[int, str], n_words: int = 10):
    """Create faceted horizontal bar charts showing top words per topic."""
    n_topics = len(topics_dict)
    cols = 2 if n_topics > 1 else 1
    rows = (n_topics + 1) // 2 if cols == 2 else n_topics

    subplot_titles = [topic_names.get(i, f"Topic {i + 1}") for i in range(n_topics)]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.12 if rows > 2 else 0.18,
    )

    for i in range(n_topics):
        r = (i // cols) + 1
        c = (i % cols) + 1
        words_weights = topics_dict[i][:n_words]
        words = [w for w, _ in reversed(words_weights)]
        weights = [wt for _, wt in reversed(words_weights)]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        fig.add_trace(
            go.Bar(
                x=weights,
                y=words,
                orientation="h",
                marker=dict(color=color),
                name=topic_names.get(i, f"Topic {i + 1}"),
                hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        height=max(400, rows * 260),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
    )
    fig.update_xaxes(title_text="Normalized Term Weight")
    return fig


def plot_topic_prevalence(df_topics: pd.DataFrame):
    """Create an interactive donut chart showing the document distribution across topics."""
    counts = df_topics["Dominant Topic Label"].value_counts().reset_index()
    counts.columns = ["Topic", "Count"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Topic"],
                values=counts["Count"],
                hole=0.45,
                marker=dict(colors=COLOR_PALETTE[: len(counts)]),
                textinfo="percent+label",
                insidetextorientation="radial",
                hovertemplate="<b>%{label}</b><br>Documents: %{value} (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="<b>Dominant Topic Distribution Across Documents</b>",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
    )
    return fig


def plot_intertopic_distance(components: np.ndarray, topic_names: Dict[int, str], df_topics: pd.DataFrame):
    """Project topic-word distributions into 2D space using PCA (Intertopic Distance Map)."""
    n_topics = components.shape[0]

    # Calculate topic prevalences (document counts)
    counts = df_topics["Dominant Topic Index"].value_counts().to_dict()
    sizes = [max(15, counts.get(i, 1) * 6) for i in range(n_topics)]
    doc_counts = [counts.get(i, 0) for i in range(n_topics)]

    if n_topics >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(components)
        pc1, pc2 = coords[:, 0], coords[:, 1]
        var1, var2 = pca.explained_variance_ratio_[:2]
    else:
        pc1, pc2 = [0.0], [0.0]
        var1, var2 = 0.0, 0.0

    labels = [topic_names.get(i, f"Topic {i + 1}") for i in range(n_topics)]
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(n_topics)]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pc1,
            y=pc2,
            mode="markers+text",
            text=[f"T{i + 1}" for i in range(n_topics)],
            textposition="middle center",
            textfont=dict(color="#ffffff", size=13, family="sans-serif"),
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.85,
                line=dict(width=2, color="#ffffff"),
            ),
            hoverinfo="text",
            hovertext=[
                f"<b>{labels[i]}</b><br>Dominant in: {doc_counts[i]} docs<br>PC1: {pc1[i]:.3f}<br>PC2: {pc2[i]:.3f}"
                for i in range(n_topics)
            ],
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"<b>Intertopic Semantic Distance Map (2D PCA)</b><br><span style='font-size:11px;color:gray'>Bubble size represents document prevalence. Distance indicates semantic divergence.</span>",
        xaxis_title=f"PC 1 ({var1 * 100:.1f}% Variance)" if n_topics >= 2 else "Dimension 1",
        yaxis_title=f"PC 2 ({var2 * 100:.1f}% Variance)" if n_topics >= 2 else "Dimension 2",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
    )
    return fig


def plot_document_topic_heatmap(doc_topic_dist: np.ndarray, topic_names: Dict[int, str], max_docs: int = 25):
    """Heatmap of document-topic probability mixtures for a sample of documents."""
    sample_size = min(len(doc_topic_dist), max_docs)
    sample_matrix = doc_topic_dist[:sample_size]

    topic_labels = [f"T{i + 1}" for i in range(doc_topic_dist.shape[1])]
    doc_labels = [f"Doc {i + 1}" for i in range(sample_size)]

    fig = go.Figure(
        data=go.Heatmap(
            z=sample_matrix,
            x=topic_labels,
            y=doc_labels,
            colorscale="Viridis",
            colorbar=dict(title="Probability"),
            hovertemplate="Document: %{y}<br>Topic: %{x}<br>Probability: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Document-Topic Probability Heatmap (First {sample_size} Docs)</b>",
        xaxis_title="Topics",
        yaxis_title="Documents",
        template="plotly_white",
        height=max(400, sample_size * 20),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
    )
    return fig


def plot_prediction_probabilities(breakdown: List[Dict]):
    """Bar chart showing predicted topic probability distribution for new unseen text."""
    labels = [b["topic_label"] for b in breakdown]
    probs = [b["probability"] * 100 for b in breakdown]
    colors = [COLOR_PALETTE[b["topic_index"] % len(COLOR_PALETTE)] for b in breakdown]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Predicted Topic Probabilities</b>",
        xaxis_title="Confidence Probability (%)",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
    )
    return fig
