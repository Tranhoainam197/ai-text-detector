import os
import joblib

try:
    # Normal case: imported as part of the `detector` package (e.g. by main.py)
    from .features import stylometry_features, features_to_vector
except ImportError:
    # Fallback: run as a standalone script (e.g. `python detector/train_ensemble.py`),
    # where `detector/` itself is on sys.path instead of its parent.
    from features import stylometry_features, features_to_vector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")


class StyleClassifier:
    """
    Gradient Boosting classifier trained on stylometry features
    (see train_classifier.py). Its raw probability is used directly -
    no manual rescaling - because the downstream ensemble (ensemble.py)
    is a Logistic Regression trained to combine this signal with the
    others, so any miscalibration is learned away there instead of
    guessed here.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> dict:
        feats = stylometry_features(text)
        if self.model is None:
            return {"ai_prob": 0.5, "model_loaded": False}

        x = features_to_vector(feats)
        proba = self.model.predict_proba(x)[0]
        ai_prob = float(proba[1])
        return {"ai_prob": ai_prob, "model_loaded": True}


_clf = None


def get_classifier() -> StyleClassifier:
    global _clf
    if _clf is None:
        _clf = StyleClassifier()
    return _clf
