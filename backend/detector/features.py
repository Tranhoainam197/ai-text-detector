import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


def split_sentences(text: str):
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def burstiness_score(text: str) -> dict:
    """
    Burstiness = variance in sentence length.
    Humans burst (short, short, LONG, short). AI tends to be uniform.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return {"burstiness": 0.0, "mean_len": 0.0, "std_len": 0.0, "sentence_count": len(sentences)}

    lengths = [len(word_tokenize(s)) for s in sentences]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    burstiness = std_len / mean_len if mean_len > 0 else 0.0
    return {
        "burstiness": burstiness,
        "mean_len": mean_len,
        "std_len": std_len,
        "sentence_count": len(sentences),
    }


def stylometry_features(text: str) -> dict:
    """
    Hand-crafted writing-style signals. These feed the ML classifier later.
    """
    words = word_tokenize(text)
    sentences = split_sentences(text)
    word_count = max(len(words), 1)

    unique_words = len(set(w.lower() for w in words if w.isalpha()))
    ttr = unique_words / word_count

    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    punct_count = sum(1 for c in text if c in ".,;:!?-—")
    punct_density = punct_count / max(len(text), 1)

    function_words = {"the", "of", "and", "a", "in", "to", "is", "was", "that",
                      "it", "for", "on", "with", "as", "but"}
    fw_count = sum(1 for w in words if w.lower() in function_words)
    fw_ratio = fw_count / word_count

    burst = burstiness_score(text)

    return {
        "word_count": word_count,
        "sentence_count": burst["sentence_count"],
        "type_token_ratio": ttr,
        "avg_word_len": avg_word_len,
        "punct_density": punct_density,
        "function_word_ratio": fw_ratio,
        "burstiness": burst["burstiness"],
        "mean_sentence_len": burst["mean_len"],
        "std_sentence_len": burst["std_len"],
    }


FEATURE_ORDER = [
    "type_token_ratio",
    "avg_word_len",
    "punct_density",
    "function_word_ratio",
    "burstiness",
    "mean_sentence_len",
    "std_sentence_len",
]


def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)