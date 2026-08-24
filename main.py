"""
Topic Modeling on Reviews — end-to-end pipeline.

Usage:
    python main.py                                   # runs on the bundled sample_reviews.csv
    python main.py --csv path/to/your_reviews.csv --text-col review --n-topics 5 --method lda
    python main.py --method nmf --n-topics 6

Outputs (written to outputs/):
    topics_summary.txt          top words for each discovered topic
    topic_selection.png         log-likelihood / perplexity curve across candidate k (LDA only)
    topics_barplots.png         top words per topic, as bar charts
    topic_distribution.png      how many reviews fall into each topic
    reviews_with_topics.csv     original reviews + assigned dominant topic
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_corpus
from topic_model import vectorize, fit_lda, fit_nmf, select_n_topics_lda, get_top_words, assign_dominant_topic
from visualize import plot_top_words_per_topic, plot_topic_selection, plot_topic_distribution

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
OUT_DIR = os.path.join(HERE, "outputs")


def main():
    parser = argparse.ArgumentParser(description="Topic modeling on review text")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to a CSV of reviews")
    parser.add_argument("--text-col", default="review", help="Name of the column containing review text")
    parser.add_argument("--method", choices=["lda", "nmf"], default="lda", help="Topic modeling algorithm")
    parser.add_argument("--n-topics", type=int, default=None, help="Number of topics (skips auto-selection)")
    parser.add_argument("--n-words", type=int, default=10, help="Top words to show per topic")
    parser.add_argument("--min-k", type=int, default=2, help="Min topics to try during auto-selection")
    parser.add_argument("--max-k", type=int, default=10, help="Max topics to try during auto-selection")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading reviews from {args.csv} ...")
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[args.text_col]).reset_index(drop=True)
    raw_texts = df[args.text_col].astype(str).tolist()
    print(f"Loaded {len(raw_texts)} reviews.")

    print("Cleaning text (lowercasing, removing stopwords/punctuation)...")
    cleaned = clean_corpus(raw_texts)

    vec_method = "count" if args.method == "lda" else "tfidf"
    print(f"Vectorizing with {vec_method} vectorizer...")
    X, vectorizer = vectorize(cleaned, method=vec_method)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    n_topics = args.n_topics
    if n_topics is None and args.method == "lda":
        print(f"No --n-topics given: searching k in [{args.min_k}, {args.max_k}] using LDA log-likelihood/perplexity...")
        results = select_n_topics_lda(X, candidate_range=range(args.min_k, args.max_k + 1))
        for r in results:
            print(f"  k={r['n_topics']:>2}  log-likelihood={r['log_likelihood']:.1f}  perplexity={r['perplexity']:.1f}")
        best = min(results, key=lambda r: r["perplexity"])
        n_topics = best["n_topics"]
        print(f"Selected k={n_topics} (lowest perplexity). Override anytime with --n-topics.")
        plot_topic_selection(results, save_path=os.path.join(OUT_DIR, "topic_selection.png"))
    elif n_topics is None:
        n_topics = 5
        print(f"No --n-topics given for NMF: defaulting to {n_topics}.")

    print(f"Fitting {args.method.upper()} with {n_topics} topics...")
    if args.method == "lda":
        model, doc_topic = fit_lda(X, n_topics)
    else:
        model, doc_topic = fit_nmf(X, n_topics)

    topics = get_top_words(model, vectorizer, n_words=args.n_words)
    dominant = assign_dominant_topic(doc_topic)

    summary_lines = [f"Topic Modeling Results ({args.method.upper()}, k={n_topics})", "=" * 50]
    for topic_idx, words in topics:
        n_docs = (dominant == topic_idx).sum()
        summary_lines.append(f"\nTopic {topic_idx}  ({n_docs} documents)")
        summary_lines.append("  " + ", ".join(words))
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(os.path.join(OUT_DIR, "topics_summary.txt"), "w") as f:
        f.write(summary_text)

    plot_top_words_per_topic(
        model, vectorizer, n_words=args.n_words,
        title=f"Top words per topic ({args.method.upper()}, k={n_topics})",
        save_path=os.path.join(OUT_DIR, "topics_barplots.png"),
    )
    plot_topic_distribution(dominant, n_topics, save_path=os.path.join(OUT_DIR, "topic_distribution.png"))

    df_out = df.copy()
    df_out["dominant_topic"] = dominant
    for topic_idx, words in topics:
        df_out.loc[df_out["dominant_topic"] == topic_idx, "topic_top_words"] = ", ".join(words[:5])
    df_out.to_csv(os.path.join(OUT_DIR, "reviews_with_topics.csv"), index=False)

    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
