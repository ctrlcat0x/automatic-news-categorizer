"""
features.py - Feature extraction utilities for headline sentiment classification.

Provides functions for POS tag analysis, TF-IDF vectorization,
and linguistic feature extraction for sentiment-labeled headlines.
"""

from collections import Counter
from typing import Optional

import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# POS tag analysis
# ---------------------------------------------------------------------------

def get_pos_distribution(
    texts: list[str],
    nlp: spacy.language.Language,
    sentiments: list[str],
) -> pd.DataFrame:
    """Compute POS tag distribution per sentiment class.

    Parameters
    ----------
    texts : list[str]
        Raw headline strings.
    nlp : spacy.language.Language
        Loaded SpaCy model.
    sentiments : list[str]
        Sentiment label for each text (parallel to texts).

    Returns
    -------
    pd.DataFrame
        Columns: sentiment, pos_tag, count.
    """
    records: list[dict] = []
    for doc, sent in zip(nlp.pipe(texts, batch_size=500), sentiments):
        pos_counts: dict[str, int] = Counter()
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counts[token.pos_] += 1
        for pos, count in pos_counts.items():
            records.append({"sentiment": sent, "pos_tag": pos, "count": count})

    return pd.DataFrame(records)


def get_top_words_per_sentiment(
    texts: list[str],
    sentiments: list[str],
    nlp: spacy.language.Language,
    pos_tags: Optional[list[str]] = None,
    top_n: int = 15,
) -> dict[str, list[tuple[str, int]]]:
    """Extract top N words (by POS) per sentiment class.

    Parameters
    ----------
    texts : list[str]
        Raw headline strings.
    sentiments : list[str]
        Parallel sentiment labels.
    nlp : spacy.language.Language
        Loaded SpaCy model.
    pos_tags : list[str] or None
        POS tags to consider. Defaults to ['NOUN', 'PROPN', 'ADJ', 'VERB'].
    top_n : int
        Number of top words to return per sentiment.

    Returns
    -------
    dict
        {sentiment: [(word, count), ...]}.
    """
    if pos_tags is None:
        pos_tags = ["NOUN", "PROPN", "ADJ", "VERB"]

    sent_counters: dict[str, Counter] = {}
    for doc, sent in zip(nlp.pipe(texts, batch_size=500), sentiments):
        if sent not in sent_counters:
            sent_counters[sent] = Counter()
        for token in doc:
            if token.pos_ in pos_tags and not token.is_stop and len(token.text) > 1:
                sent_counters[sent][token.lemma_.lower()] += 1

    return {s: counter.most_common(top_n) for s, counter in sorted(sent_counters.items())}


# ---------------------------------------------------------------------------
# TF-IDF vectorization
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    max_df: float = 0.8,
    min_df: int = 5,
    max_features: int = 10000,
    use_stop_words: bool = False,
) -> TfidfVectorizer:
    """Create a configured TfidfVectorizer."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        stop_words="english" if use_stop_words else None,
        sublinear_tf=True,
    )
