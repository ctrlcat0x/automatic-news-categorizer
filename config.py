# =============================================================================
# config.py — Project Configuration
# =============================================================================
# Edit the values in this file to control the pipeline behaviour without
# touching the notebook. All settings are imported at the top of the notebook.
# =============================================================================

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------

# Number of headlines to randomly sample from the full ~1.2M dataset.
# Lower values = faster iteration; higher values = more representative results.
# Recommended range: 10_000 (fast) → 200_000 (comprehensive)
SAMPLE_SIZE: int = 100_000

# Random seed used throughout for reproducible sampling, train/test splits,
# and model training. Change to get a different random partition.
RANDOM_STATE: int = 42

# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------

# How many of the most-common words to display and save in the Top Words chart.
TOP_N_COMMON_WORDS: int = 50

# Number of headlines used for Named Entity Recognition (NER) analysis.
# NER is computationally intensive — keep this well below SAMPLE_SIZE.
NER_SAMPLE: int = 10_000

# Number of headlines used for Part-of-Speech (POS) tag analysis.
POS_SAMPLE: int = 10_000

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

# TF-IDF vocabulary size (max number of n-gram features).
TFIDF_MAX_FEATURES: int = 10_000
