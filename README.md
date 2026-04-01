# Automatic News Headline Sentiment Categorizer

A multi-class NLP classifier that automatically categorizes news headlines by **sentiment** (Positive / Negative / Neutral) using linguistic features and logical reasoning analysis.

## Project Overview

This project demonstrates key NLP concepts including:

- **Natural Language Understanding (NLU)** — text preprocessing with SpaCy (tokenization, lemmatization, NER)
- **Sentiment Analysis** — automated VADER labeling + ML-based classification
- **Sequence Labeling & POS Analysis** — uncovering linguistic patterns across sentiment classes
- **Ambiguity Analysis** — logical reasoning about misclassified and ambiguous headlines

## Dataset

**ABC News Headlines** (~1.2 million headlines, 2003–2021)

- **Format:** `publish_date` (yyyymmdd) | `headline_text`
- Sampled to 50,000 headlines for tractable analysis (configurable)

## Project Structure

```
automatic-news-categorizer/
├── data/
│   └── abcnews-date-text.csv      # ABC News headlines dataset
├── notebooks/
│   └── news_categorizer.ipynb      # Main Jupyter notebook (end-to-end pipeline)
├── src/
│   ├── preprocess.py               # Dataset loading, VADER labeling, SpaCy preprocessing
│   ├── features.py                 # POS analysis & TF-IDF feature extraction
│   └── model.py                    # Model training, evaluation & ambiguity analysis
├── results/                        # Generated outputs (plots, CSVs)
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download SpaCy language model
python -m spacy download en_core_web_sm
```

## How to Run

1. Place `abcnews-date-text.csv` in the `data/` folder.
2. Open `notebooks/news_categorizer.ipynb` in Jupyter or VS Code.
3. Run all cells sequentially (the notebook is fully self-contained).
4. Results are saved to the `results/` folder.

## Pipeline Overview

1. **Load & Explore** — parse dates, analyze headline length/temporal distribution
2. **Sentiment Labeling** — VADER assigns positive/negative/neutral labels
3. **SpaCy NLP** — tokenize, lemmatize, extract named entities
4. **POS Analysis** — POS tag distributions and top words per sentiment
5. **TF-IDF Vectorization** — (1,2)-gram features, 10K max features
6. **Train & Evaluate** — 4 models: Linear SVM, Logistic Regression, Naive Bayes, Random Forest
7. **Ambiguity Analysis** — misclassified examples with confidence scores & logical reasoning
8. **Confusion Matrix** — per-model and comparative visualizations

## Analysis Highlights

### Linguistic Reasoning Patterns

| Sentiment    | POS Pattern                    | Entity Pattern         | Key Words                   |
| ------------ | ------------------------------ | ---------------------- | --------------------------- |
| **Negative** | Strong ADJ, action VERB        | GPE (crisis locations) | kill, attack, crash, death  |
| **Positive** | Achievement VERB, positive ADJ | PERSON (achievers)     | win, boost, new, celebrate  |
| **Neutral**  | Noun-heavy, entity-dense       | ORG, GPE (factual)     | say, plan, report, announce |

### Ambiguity Findings

Misclassifications occur due to genuine linguistic ambiguity:

- **Negation blindness**: BoW models can't distinguish "not guilty" from "guilty"
- **Mixed valence**: "Hero dies saving family" — tragic yet heroic
- **Domain connotation**: "Australia crushes England" (sports positive, literal negative)
- **VADER threshold edges**: Headlines scoring near ±0.05 are inherently ambiguous

## Output Files

| File                                        | Description                                    |
| ------------------------------------------- | ---------------------------------------------- |
| `sample_results/confusion_matrix.png`       | Confusion matrix for best model                |
| `sample_results/all_confusion_matrices.png` | Side-by-side comparison of all models          |
| `sample_results/model_comparison.png`       | Accuracy/F1 bar chart                          |
| `sample_results/sentiment_distribution.png` | Sentiment class distribution                   |
| `sample_results/sentiment_over_time.png`    | Sentiment trends across years                  |
| `sample_results/pos_distribution.png`       | POS tag proportions by sentiment               |
| `sample_results/misclassified_examples.csv` | All misclassified test samples with confidence |

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, spacy, nltk

## License

Dataset sourced from ABC News (Australia). Used for educational purposes only.
