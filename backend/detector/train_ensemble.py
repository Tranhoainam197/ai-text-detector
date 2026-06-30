"""
Trains the ensemble combiner: a small Logistic Regression that learns how to
weight perplexity, burstiness, and the stylometry classifier's output.

This replaces hand-picked constants (e.g. "weight perplexity 0.6, burstiness
0.4") with weights estimated from labeled data — the standard "stacking"
approach: base signals -> meta-features -> meta-classifier.

Usage:
    python detector/train_ensemble.py

Reads backend/data/dataset.csv (text,label) and writes
backend/models/ensemble.pkl. Requires backend/models/classifier.pkl to
already exist (run train_classifier.py first).
"""
import os
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import stylometry_features
from classifier import StyleClassifier
from perplexity import get_scorer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble.pkl")

# Meta-features fed into the combiner, in fixed order.
META_FEATURE_ORDER = ["perplexity", "burstiness", "classifier_ai_prob"]


def build_meta_features(text: str, clf: StyleClassifier, scorer) -> list[float]:
    feats = stylometry_features(text)
    ppl = scorer.score(text)
    if ppl == float("inf") or ppl != ppl:
        ppl = 1000.0  # cap degenerate values instead of letting inf poison training
    clf_out = clf.predict(text)
    return [ppl, feats["burstiness"], clf_out["ai_prob"]]


def main():
    print(f"Loading dataset from {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)

    clf = StyleClassifier()
    if clf.model is None:
        raise RuntimeError("classifier.pkl not found — run train_classifier.py first.")
    scorer = get_scorer()

    print("Building meta-features (perplexity + burstiness + classifier prob)...")
    print("This is slow: GPT-2 has to score every document.")
    X = []
    for i, text in enumerate(df["text"]):
        X.append(build_meta_features(str(text), clf, scorer))
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(df)}")
    X = np.array(X, dtype=np.float64)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Logistic Regression combiner...")
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_train, y_train)

    print("\nClassification report:")
    print(classification_report(y_test, meta_clf.predict(X_test), target_names=["human", "AI"]))
    auc = roc_auc_score(y_test, meta_clf.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC: {auc:.3f}")
    print("\nLearned weights (higher |coef| = more influence on the decision):")
    for name, coef in zip(META_FEATURE_ORDER, meta_clf.coef_[0]):
        print(f"  {name:20s} {coef:+.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump({"model": meta_clf, "feature_order": META_FEATURE_ORDER}, OUT_PATH)
    print(f"\nSaved combiner to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
