"""
Topic modeling core: vectorization, LDA / NMF fitting, topic-count selection,
and helpers to extract human-readable topics and per-document assignments.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


def vectorize(corpus, method="count", max_df=0.9, min_df=2, max_features=2000, ngram_range=(1, 1)):
    """
    method='count' -> CountVectorizer (use for LDA, which models raw word counts)
    method='tfidf' -> TfidfVectorizer (use for NMF, which factorizes weighted term importance)

    ngram_range defaults to unigrams only: on small corpora, bigrams that happen to
    pass min_df can dominate and make topics look noisier/less interpretable. Pass
    ngram_range=(1, 2) explicitly if you want bigrams included (usually helps more
    on larger, real-world datasets).
    """
    if method == "count":
        vectorizer = CountVectorizer(
            max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range
        )
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(
            max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range
        )
    else:
        raise ValueError("method must be 'count' or 'tfidf'")
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def fit_lda(X, n_topics, random_state=42, max_iter=25):
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="online",
        max_iter=max_iter,
    )
    doc_topic = model.fit_transform(X)
    return model, doc_topic


def fit_nmf(X, n_topics, random_state=42, max_iter=400):
    model = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=max_iter)
    doc_topic = model.fit_transform(X)
    return model, doc_topic


def select_n_topics_lda(X, candidate_range=range(2, 11), random_state=42):
    """
    Fit LDA for each candidate topic count and return their log-likelihood scores
    (higher is better fit) and perplexity (lower is better) so you can eyeball an
    elbow point. This is a held-in-sample diagnostic, not a substitute for held-out
    evaluation on a real dataset, but is a reasonable quick default.
    """
    results = []
    for k in candidate_range:
        model = LatentDirichletAllocation(
            n_components=k, random_state=random_state, learning_method="online", max_iter=25
        )
        model.fit(X)
        results.append(
            {
                "n_topics": k,
                "log_likelihood": model.score(X),
                "perplexity": model.perplexity(X),
            }
        )
    return results


def get_top_words(model, vectorizer, n_words=10):
    """Returns a list of (topic_index, [top words]) for either an LDA or NMF model."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic_idx, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:n_words]
        topics.append((topic_idx, list(feature_names[top_indices])))
    return topics


def assign_dominant_topic(doc_topic_matrix):
    """Returns the index of the highest-weighted topic per document."""
    return doc_topic_matrix.argmax(axis=1)
