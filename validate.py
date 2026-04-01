"""Quick validation script — runs the entire sentiment pipeline end-to-end."""
import sys, os
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from src.preprocess import load_headline_dataset, label_sentiment_vader, load_spacy_model, preprocess_batch
from src.features import get_pos_distribution, get_top_words_per_sentiment, build_tfidf_vectorizer
from src.model import build_pipeline, evaluate_model, get_misclassified, get_top_features_per_class

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Step 1: Load (small sample for validation)
df = load_headline_dataset('data/abcnews-date-text.csv', sample_size=5000, random_state=RANDOM_STATE)
print('Step 1 OK')

# Step 2: Sentiment labeling
labels, scores = label_sentiment_vader(df['headline_text'].tolist())
df['sentiment'] = labels
df['vader_score'] = scores
print(f'Step 2 OK — distribution: {df["sentiment"].value_counts().to_dict()}')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['headline_text'].tolist(), df['sentiment'].tolist(),
    test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment'])
print(f'Split: Train={len(X_train)}, Test={len(X_test)}')

# Step 3: NLP preprocessing
nlp = load_spacy_model('en_core_web_sm')
nlp.max_length = 2_000_000
X_train_clean = preprocess_batch(X_train, nlp, batch_size=500)
X_test_clean = preprocess_batch(X_test, nlp, batch_size=500)
print('Step 3 OK — preprocessing done')

# Step 4: POS analysis
pos_df = get_pos_distribution(X_train[:500], nlp, y_train[:500])
top_words = get_top_words_per_sentiment(X_train[:500], y_train[:500], nlp, top_n=5)
for s, words in top_words.items():
    print(f'  {s}: {[w for w,_ in words]}')
print('Step 4 OK')

# Step 5: TF-IDF
tfidf = build_tfidf_vectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5, max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train_clean)
print(f'Step 5 OK — TF-IDF shape {X_train_tfidf.shape}')

# Step 6: Train models
SENTIMENTS = sorted(df['sentiment'].unique())
model_names = ['linearsvc', 'logistic', 'naivebayes', 'randomforest']
results = {}
for name in model_names:
    vec = build_tfidf_vectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5, max_features=10000)
    pipeline = build_pipeline(vec, classifier_name=name, random_state=RANDOM_STATE)
    pipeline.fit(X_train_clean, y_train)
    ev = evaluate_model(pipeline, X_test_clean, y_test, label_names=SENTIMENTS)
    results[name] = {
        'pipeline': pipeline, 'accuracy': ev['accuracy'],
        'predictions': ev['predictions'], 'report_dict': ev['report_dict'],
    }
    acc = ev['accuracy']
    print(f'  {name}: {acc:.4f}')
print('Step 6 OK')

# Step 7: Misclassified
best_name = max(results, key=lambda n: results[n]['accuracy'])
best_pipeline = results[best_name]['pipeline']
y_pred = results[best_name]['predictions']
mis = get_misclassified(X_test_clean, y_test, y_pred.tolist(), best_pipeline)
print(f'Step 7 OK — {len(mis)} misclassified')

# Top features
top_feats = get_top_features_per_class(best_pipeline, SENTIMENTS, top_n=10)
for s, feats in top_feats.items():
    print(f'  {s}: {[w for w,_ in feats[:5]]}')

print('\nALL STEPS COMPLETE')
