"""
Train the stylometry classifier on backend/data/dataset.csv.
Saves the model to backend/models/classifier.pkl.
"""
import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Allow running this script directly from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import stylometry_features, FEATURE_ORDER

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")


def main():
    print(f"Loading dataset from {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    print("\nExtracting features (this takes 1-2 minutes)...")
    X = []
    for i, t in enumerate(df["text"]):
        feats = stylometry_features(str(t))
        X.append([feats[k] for k in FEATURE_ORDER])
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(df)}")
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Gradient Boosting Classifier...")
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    print("\nClassification report:")
    print(classification_report(y_test, clf.predict(X_test), target_names=["human", "AI"]))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC: {auc:.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(clf, OUT_PATH)
    print(f"\nSaved model to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()