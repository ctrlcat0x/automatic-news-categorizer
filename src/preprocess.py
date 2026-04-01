"""
preprocess.py - Text preprocessing utilities for headline sentiment analysis.

Provides functions for loading the ABC News headlines dataset, cleaning text,
tokenization, lemmatization, stopword removal, sentiment labeling (VADER),
and named entity extraction using SpaCy.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Doc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_headline_dataset(
    csv_path: str,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the ABC News headlines CSV into a DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the abcnews-date-text.csv file.
    sample_size : int or None
        If set, randomly sample this many rows (for faster iteration).
    random_state : int
        Random state for sampling reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'publish_date' (datetime), 'headline_text'.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse yyyymmdd integer dates to datetime
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d")
    df["headline_text"] = df["headline_text"].astype(str).str.strip()

    # Drop empty headlines
    df = df[df["headline_text"].str.len() > 0].reset_index(drop=True)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    print(f"Loaded {len(df)} headlines "
          f"({df['publish_date'].min().date()} to {df['publish_date'].max().date()}).")
    return df


# ---------------------------------------------------------------------------
# Sentiment labeling with VADER
# ---------------------------------------------------------------------------

def label_sentiment_vader(
    texts: list[str],
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05,
) -> tuple[list[str], list[float]]:
    """Assign sentiment labels using NLTK VADER.

    Parameters
    ----------
    texts : list[str]
        Raw headline strings.
    pos_threshold : float
        VADER compound score above this → positive.
    neg_threshold : float
        VADER compound score below this → negative.

    Returns
    -------
    labels : list[str]
        'positive', 'negative', or 'neutral' per headline.
    scores : list[float]
        Raw VADER compound scores.
    """
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    labels: list[str] = []
    scores: list[float] = []
    for text in texts:
        compound = sia.polarity_scores(text)["compound"]
        scores.append(compound)
        if compound > pos_threshold:
            labels.append("positive")
        elif compound < neg_threshold:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels, scores


# ---------------------------------------------------------------------------
# SpaCy preprocessing
# ---------------------------------------------------------------------------

def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """Load a SpaCy model, downloading it if necessary."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading SpaCy model '{model_name}'...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def preprocess_text(
    doc: Doc,
    remove_stopwords: bool = True,
    remove_punct: bool = True,
    lemmatize: bool = True,
) -> str:
    """Clean a single SpaCy Doc: lemmatize, remove stopwords & punctuation."""
    tokens: list[str] = []
    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        if remove_punct and (token.is_punct or token.is_space):
            continue
        if token.like_num:
            continue
        word = token.lemma_.lower().strip() if lemmatize else token.text.lower().strip()
        if len(word) > 1:
            tokens.append(word)
    return " ".join(tokens)


def preprocess_batch(
    texts: list[str],
    nlp: spacy.language.Language,
    batch_size: int = 500,
    n_process: int = 1,
) -> list[str]:
    """Preprocess a batch of texts using SpaCy pipe for efficiency."""
    cleaned: list[str] = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        cleaned.append(preprocess_text(doc))
    return cleaned


# ---------------------------------------------------------------------------
# Named Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(doc: Doc) -> dict:
    """Extract named entity counts from a SpaCy Doc."""
    entity_counts: dict[str, int] = {}
    for ent in doc.ents:
        label = ent.label_
        entity_counts[label] = entity_counts.get(label, 0) + 1
    return entity_counts


def extract_entity_features(
    texts: list[str],
    nlp: spacy.language.Language,
    entity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a DataFrame of entity-count features for a list of texts."""
    if entity_labels is None:
        entity_labels = ["PERSON", "ORG", "GPE", "PRODUCT", "MONEY",
                         "DATE", "NORP", "EVENT", "WORK_OF_ART", "LAW"]

    rows: list[dict] = []
    for doc in nlp.pipe(texts, batch_size=500):
        counts = extract_entities(doc)
        row = {f"ent_{label}": counts.get(label, 0) for label in entity_labels}
        rows.append(row)

    return pd.DataFrame(rows)
