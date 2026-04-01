"""
model.py - Model training, evaluation, and ambiguity analysis utilities
for headline sentiment classification.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_pipeline(
    vectorizer,
    classifier_name: str = "linearsvc",
    random_state: int = 42,
) -> Pipeline:
    """Build an sklearn Pipeline with a vectorizer and classifier.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Configured (unfitted) vectorizer.
    classifier_name : str
        One of 'linearsvc', 'logistic', 'naivebayes', 'randomforest'.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    Pipeline
    """
    classifiers = {
        "linearsvc": CalibratedClassifierCV(
            LinearSVC(random_state=random_state, max_iter=10000, C=1.0),
            cv=3,
        ),
        "logistic": LogisticRegression(
            random_state=random_state, max_iter=1000, C=1.0, solver="lbfgs",
        ),
        "naivebayes": MultinomialNB(alpha=1.0),
        "randomforest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1,
        ),
    }

    clf = classifiers.get(classifier_name.lower())
    if clf is None:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Choose from: {list(classifiers.keys())}"
        )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    X_test: list[str],
    y_test: list[str],
    label_names: Optional[list[str]] = None,
) -> dict:
    """Evaluate a trained pipeline on the test set.

    Returns dict with keys: accuracy, report, report_dict, confusion_matrix, predictions.
    """
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=False,
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_names)

    return {
        "accuracy": acc,
        "report": report,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


# ---------------------------------------------------------------------------
# Misclassification / ambiguity analysis
# ---------------------------------------------------------------------------

def get_misclassified(
    texts: list[str],
    y_true: list[str],
    y_pred,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Build a DataFrame of misclassified samples with confidence scores."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mask = y_true_arr != y_pred_arr
    indices = np.where(mask)[0]

    # Get probabilities if available
    try:
        proba = pipeline.predict_proba(texts)
        classes = pipeline.classes_
    except AttributeError:
        proba = None
        classes = None

    records: list[dict] = []
    for idx in indices:
        record = {
            "headline": texts[idx],
            "true_label": y_true_arr[idx],
            "predicted_label": y_pred_arr[idx],
        }
        if proba is not None and classes is not None:
            pred_class_idx = np.where(classes == y_pred_arr[idx])[0][0]
            true_class_idx = np.where(classes == y_true_arr[idx])[0][0]
            record["confidence"] = round(float(proba[idx, pred_class_idx]), 4)
            record["true_label_prob"] = round(float(proba[idx, true_class_idx]), 4)
        records.append(record)

    return pd.DataFrame(records)


def get_top_features_per_class(
    pipeline: Pipeline,
    class_names: list[str],
    top_n: int = 15,
) -> dict[str, list[tuple[str, float]]]:
    """Extract top weighted features per class from a linear model."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    feature_names = vectorizer.get_feature_names_out()

    # Handle CalibratedClassifierCV wrapping
    if hasattr(clf, "calibrated_classifiers_"):
        coefs = np.mean(
            [cc.estimator.coef_ for cc in clf.calibrated_classifiers_], axis=0
        )
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        coefs = clf.feature_log_prob_
    elif hasattr(clf, "feature_importances_"):
        # RandomForest: shared importances (not per-class), return same for all
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[-top_n:][::-1]
        shared = [(feature_names[j], round(float(importances[j]), 4)) for j in top_idx]
        return {name: shared for name in class_names}
    else:
        return {name: [] for name in class_names}

    top_features: dict[str, list[tuple[str, float]]] = {}
    for i, name in enumerate(class_names):
        top_idx = np.argsort(coefs[i])[-top_n:][::-1]
        top_features[name] = [
            (feature_names[j], round(float(coefs[i, j]), 4)) for j in top_idx
        ]

    return top_features
