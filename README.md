# Topic Modeling on Reviews

An end-to-end NLP project that discovers latent topics in a set of product/service
reviews using **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix
Factorization (NMF)**, built entirely on `scikit-learn` (no `gensim`/`nltk` required).

## What it does

1. **Loads** a CSV of reviews (a bundled synthetic sample is included, or point it
   at your own data).
2. **Cleans** the text — lowercasing, punctuation/number stripping, stopword removal.
3. **Vectorizes** — `CountVectorizer` for LDA (models raw word co-occurrence),
   `TfidfVectorizer` for NMF (models weighted term importance).
4. **Selects the number of topics** — for LDA, sweeps a range of `k` and reports
   log-likelihood / perplexity so you can pick a good `k` (or supply your own).
5. **Fits the topic model** and extracts the top words per topic.
6. **Visualizes** results — bar charts of top words per topic, a topic-count
   selection curve, and a histogram of how many reviews fall into each topic.
7. **Exports** the original reviews tagged with their dominant topic.

## Project structure

```
topic_modeling_reviews/
├── main.py                     # CLI entry point — run this
├── requirements.txt
├── data/
│   ├── sample_reviews.csv      # bundled synthetic reviews (5 domains, 170 rows)
│   └── generate_sample_data.py # regenerate/customize the sample data
├── src/
│   ├── preprocess.py           # text cleaning
│   ├── topic_model.py          # vectorization, LDA/NMF fitting, topic extraction
│   └── visualize.py            # matplotlib charts
└── outputs/                    # generated on each run
    ├── topics_summary.txt
    ├── topics_barplots.png
    ├── topic_selection.png     # LDA only
    ├── topic_distribution.png
    └── reviews_with_topics.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run on the bundled sample data with automatic topic-count selection:
```bash
python main.py
```

Use your own reviews CSV (needs at least a text column):
```bash
python main.py --csv path/to/your_reviews.csv --text-col review_text
```

Fix the number of topics instead of auto-selecting:
```bash
python main.py --n-topics 5
```

Use NMF instead of LDA (often sharper/more distinct topics on smaller datasets):
```bash
python main.py --method nmf --n-topics 5
```

Full options:
```bash
python main.py --help
```

## LDA vs. NMF — which to use

- **LDA** is probabilistic — it models each document as a mixture of topics and
  each topic as a distribution over words. It tends to do better with larger
  corpora and gives you principled model-selection metrics (log-likelihood,
  perplexity), which is why `main.py` uses those to auto-pick `k` for LDA.
- **NMF** factorizes the TF-IDF matrix directly. It has no probabilistic
  interpretation but often produces crisper, more human-readable topics,
  especially on smaller or shorter-text datasets like reviews — the bundled
  sample data demonstrates this: NMF separates the 5 review domains (electronics,
  restaurant, hotel, streaming, clothing) more cleanly than LDA does at the
  same `k`.

Try both and compare `outputs/topics_summary.txt` between runs.

## Using your own dataset

Swap in any CSV with a text column of reviews. Real-world review datasets (Amazon,
Yelp, IMDB, app store reviews, etc.) will generally produce cleaner topics than the
bundled sample, since real reviews carry more consistent vocabulary per domain and
far higher volume. Popular sources: Kaggle's Amazon/Yelp review dumps, the
Stanford SNAP review datasets, or your own product/support review export.

## Streamlit app

An interactive frontend is included in `streamlit_app.py` — upload your own reviews
CSV (or use the bundled sample), pick LDA or NMF, choose or auto-select the number
of topics, and browse the resulting topics, distribution, and per-review tags in
the browser, with a CSV download button.

Run it locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick this repo/branch, and set the main file path to
   `streamlit_app.py`.
4. Deploy — Streamlit Cloud installs from `requirements.txt` automatically.

## Extending this project

- Swap in `gensim`'s `LdaModel` + `CoherenceModel` for proper topic coherence
  scores (c_v, u_mass) if you have `gensim` available — a more standard metric
  than perplexity for choosing `k`.
- Add sentiment analysis (e.g. `TextBlob`/`VADER`) per topic to see which topics
  skew positive vs. negative.
- Try `BERTopic` (transformer embeddings + clustering) for state-of-the-art
  topic quality on larger datasets — significantly heavier dependency-wise.
- Add bigrams back in (`ngram_range=(1, 2)` in `src/topic_model.py`) once your
  dataset is large enough that meaningful bigrams pass `min_df` without noise.
