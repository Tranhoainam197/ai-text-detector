"""
Combines the three base signals - perplexity, burstiness, and the stylometry
classifier - into a single AI-probability score.

The combination weights are learned from data (Logistic Regression trained in
train_ensemble.py: see backend/models/ensemble.pkl) rather than hand-picked.
This is a "stacking" ensemble: base model outputs become features for a
meta-classifier. If the learned combiner hasn't been trained yet, we fall
back to a simple, clearly-labeled average so the API still works.
"""
import os
import math
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble.pkl")

_combiner = None
_combiner_loaded_from_disk = False


def _load_combiner():
    global _combiner, _combiner_loaded_from_disk
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        _combiner = bundle["model"]
        _combiner_loaded_from_disk = True
    return _combiner


_load_combiner()


def _safe_perplexity(ppl: float) -> float:
    """Cap degenerate perplexity values (inf / NaN) to a large-but-finite number."""
    if ppl == float("inf") or ppl != ppl:
        return 1000.0
    return ppl


def _fallback_combine(perplexity: float, burstiness: float, classifier_ai_prob: float) -> float:
    """
    Unweighted average used only if the learned combiner is missing
    (e.g. before train_ensemble.py has been run). Not the primary path.
    """
    midpoint, steepness = 45.0, 0.08
    ppl_ai = 1.0 / (1.0 + math.exp(steepness * (_safe_perplexity(perplexity) - midpoint)))
    burst_ai = max(0.0, min(1.0, 0.9 - 0.8 * max(0.0, min(burstiness, 1.2))))
    return float(np.mean([ppl_ai, burst_ai, classifier_ai_prob]))


def combine(perplexity: float, feats: dict, classifier_ai_prob: float = None,
            classifier_loaded: bool = False) -> dict:
    burstiness = feats["burstiness"]
    classifier_ai_prob = classifier_ai_prob if classifier_loaded and classifier_ai_prob is not None else 0.5

    components = {
        "perplexity": _safe_perplexity(perplexity),
        "burstiness": burstiness,
        "classifier_ai_prob": classifier_ai_prob,
    }

    if _combiner is not None:
        x = np.array([[components["perplexity"], burstiness, classifier_ai_prob]])
        ai_score = float(_combiner.predict_proba(x)[0, 1])
        method = "learned_logistic_regression"
    else:
        ai_score = _fallback_combine(perplexity, burstiness, classifier_ai_prob)
        method = "fallback_average"

    ai_score = max(0.0, min(1.0, ai_score))
    confidence = abs(ai_score - 0.5) * 2

    return {
        "ai_probability": ai_score,
        "human_probability": 1 - ai_score,
        "confidence": confidence,
        "components": components,
        "method": method,
    }
