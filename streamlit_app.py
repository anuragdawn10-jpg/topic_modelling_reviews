"""
Streamlit app for the Topic Modeling on Reviews project.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
import io
import os
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_corpus
from topic_model import vectorize, fit_lda, fit_nmf, select_n_topics_lda, get_top_words, assign_dominant_topic

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_default_csv():
    candidates = [
        os.path.join(HERE, "data", "sample_reviews.csv"),
        os.path.join(HERE, "sample_reviews.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No bundled sample CSV found. Expected one of: " + ", ".join(candidates)
    )


DEFAULT_CSV = resolve_default_csv()

st.set_page_config(page_title="Topic Modeling on Reviews", layout="wide")


@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_CSV)


@st.cache_data(show_spinner=False)
def run_topic_selection(cleaned_texts, min_k, max_k):
    X, _ = vectorize(cleaned_texts, method="count")
    results = select_n_topics_lda(X, candidate_range=range(min_k, max_k + 1))
    return results


def bar_fig_for_topic(words, weights, title):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    order = np.argsort(weights)
    ax.barh(np.array(words)[order], np.array(weights)[order], color="steelblue")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def main():
    st.title("📝 Topic Modeling on Reviews")
    st.caption("LDA / NMF topic discovery over review text — built with scikit-learn.")

    # ---------------- Sidebar controls ----------------
    st.sidebar.header("1. Data")
    source = st.sidebar.radio("Review source", ["Use bundled sample data", "Upload my own CSV"])

    if source == "Upload my own CSV":
        uploaded = st.sidebar.file_uploader("Upload a CSV of reviews", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV in the sidebar, or switch to the bundled sample data, to get started.")
            st.stop()
        df = pd.read_csv(uploaded)
    else:
        df = load_default_data()

    text_col = st.sidebar.selectbox(
        "Text column", options=list(df.columns),
        index=list(df.columns).index("review") if "review" in df.columns else 0,
    )

    st.sidebar.header("2. Model")
    method = st.sidebar.radio("Algorithm", ["NMF", "LDA"], help="NMF often gives crisper topics on shorter review text; LDA is probabilistic and supports the auto-k search below.")
    n_words = st.sidebar.slider("Top words per topic", 5, 15, 10)

    auto_k = False
    if method == "LDA":
        auto_k = st.sidebar.checkbox("Auto-select number of topics (LDA)", value=False)

    if auto_k:
        min_k, max_k = st.sidebar.slider("Search range for k", 2, 15, (2, 10))
        n_topics = None
    else:
        n_topics = st.sidebar.slider("Number of topics", 2, 15, 5)

    run = st.sidebar.button("Run topic modeling", type="primary")

    st.subheader("Preview of the data")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df)} reviews loaded.")

    if not run:
        st.stop()

    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    raw_texts = df[text_col].astype(str).tolist()

    with st.spinner("Cleaning text..."):
        cleaned = clean_corpus(raw_texts)

    if auto_k:
        with st.spinner(f"Searching k in [{min_k}, {max_k}]..."):
            results = run_topic_selection(tuple(cleaned), min_k, max_k)
        results_df = pd.DataFrame(results)
        st.subheader("Topic-count selection (LDA)")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(results_df.set_index("n_topics")["log_likelihood"])
            st.caption("Log-likelihood vs. k (higher = better fit)")
        with col2:
            st.line_chart(results_df.set_index("n_topics")["perplexity"])
            st.caption("Perplexity vs. k (lower = better fit)")
        n_topics = int(results_df.loc[results_df["perplexity"].idxmin(), "n_topics"])
        st.success(f"Selected k = {n_topics} (lowest perplexity). You can turn off auto-select to override.")

    vec_method = "count" if method == "LDA" else "tfidf"
    with st.spinner(f"Vectorizing ({vec_method})..."):
        X, vectorizer = vectorize(cleaned, method=vec_method)

    with st.spinner(f"Fitting {method} with {n_topics} topics..."):
        if method == "LDA":
            model, doc_topic = fit_lda(X, n_topics)
        else:
            model, doc_topic = fit_nmf(X, n_topics)

    topics = get_top_words(model, vectorizer, n_words=n_words)
    dominant = assign_dominant_topic(doc_topic)

    st.subheader(f"Discovered topics ({method}, k={n_topics})")
    n_cols = 3
    rows_needed = int(np.ceil(n_topics / n_cols))
    topic_idx = 0
    for _ in range(rows_needed):
        cols = st.columns(n_cols)
        for c in cols:
            if topic_idx >= n_topics:
                break
            words = topics[topic_idx][1]
            weights = model.components_[topic_idx][
                model.components_[topic_idx].argsort()[::-1][:n_words]
            ]
            n_docs = int((dominant == topic_idx).sum())
            with c:
                st.pyplot(bar_fig_for_topic(words, weights, f"Topic {topic_idx} ({n_docs} docs)"), use_container_width=True)
            topic_idx += 1

    st.subheader("Topic distribution across reviews")
    dist = pd.Series(dominant).value_counts().sort_index()
    dist.index = [f"Topic {i}" for i in dist.index]
    st.bar_chart(dist)

    df_out = df.copy()
    df_out["dominant_topic"] = dominant
    for idx, words in topics:
        df_out.loc[df_out["dominant_topic"] == idx, "topic_top_words"] = ", ".join(words[:5])

    st.subheader("Reviews tagged with their dominant topic")
    st.dataframe(df_out, use_container_width=True)

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv_bytes, file_name="reviews_with_topics.csv", mime="text/csv")


if __name__ == "__main__":
    main()
