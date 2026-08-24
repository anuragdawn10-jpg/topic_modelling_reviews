"""
Matplotlib-only visualizations (no wordcloud dependency needed):
- horizontal bar chart of top words per topic
- topic-count selection curve (log-likelihood / perplexity vs k)
- topic distribution across the corpus
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_top_words_per_topic(model, vectorizer, n_words=10, title="Topics", save_path=None):
    feature_names = np.array(vectorizer.get_feature_names_out())
    n_topics = model.components_.shape[0]
    n_cols = min(3, n_topics)
    n_rows = int(np.ceil(n_topics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for topic_idx, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:n_words]
        top_words = feature_names[top_indices]
        weights = component[top_indices]

        ax = axes[topic_idx]
        ax.barh(range(len(top_words)), weights[::-1], color="steelblue")
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words[::-1], fontsize=10)
        ax.set_title(f"Topic {topic_idx}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[n_topics:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_topic_selection(results, save_path=None):
    ks = [r["n_topics"] for r in results]
    ll = [r["log_likelihood"] for r in results]
    perp = [r["perplexity"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ks, ll, marker="o", color="steelblue")
    axes[0].set_xlabel("Number of topics (k)")
    axes[0].set_ylabel("Log-likelihood (higher = better fit)")
    axes[0].set_title("Log-likelihood vs k")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, perp, marker="o", color="indianred")
    axes[1].set_xlabel("Number of topics (k)")
    axes[1].set_ylabel("Perplexity (lower = better fit)")
    axes[1].set_title("Perplexity vs k")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_topic_distribution(dominant_topics, n_topics, save_path=None):
    counts = np.bincount(dominant_topics, minlength=n_topics)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(n_topics), counts, color="darkseagreen")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of documents")
    ax.set_title("Document counts per dominant topic")
    ax.set_xticks(range(n_topics))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
