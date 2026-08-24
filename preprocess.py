"""
Lightweight text preprocessing for review text — no external NLP corpora required
(pure regex + sklearn's built-in stopword list), so it runs anywhere sklearn does.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# A few extra stopwords common in review text that aren't in sklearn's default list
# but carry little topical meaning on their own.
EXTRA_STOPWORDS = {
    "just", "really", "very", "quite", "also", "would", "could", "got", "get",
    "one", "even", "still", "though", "went", "im", "ive", "didnt", "dont",
    "wasnt", "isnt", "us", "will", "make", "made", "like",
}

STOPWORDS = ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS)

_word_re = re.compile(r"[a-zA-Z]+")


def clean_text(text: str, min_len: int = 3) -> str:
    """Lowercase, strip non-letters, drop stopwords and very short tokens."""
    text = text.lower()
    tokens = _word_re.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= min_len]
    return " ".join(tokens)


def clean_corpus(texts):
    """Apply clean_text to an iterable of raw review strings."""
    return [clean_text(t) for t in texts]
