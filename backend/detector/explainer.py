def explain(feats: dict, perplexity: float, ai_prob: float) -> list[str]:
    """
    Generate human-readable explanations from the measured features.
    Each rule fires only when its threshold is crossed.
    """
    reasons = []

    # Perplexity signals
    if perplexity < 30:
        reasons.append(
            f"Very low perplexity ({perplexity:.1f}) — the text is highly predictable, "
            f"which is typical of language-model output."
        )
    elif perplexity < 50:
        reasons.append(
            f"Low perplexity ({perplexity:.1f}) — text is more predictable than average human writing."
        )
    elif perplexity > 100:
        reasons.append(
            f"High perplexity ({perplexity:.1f}) — text contains unusual or unpredictable word choices, "
            f"more characteristic of human writing."
        )

    # Burstiness signals
    burst = feats["burstiness"]
    if burst < 0.25:
        reasons.append(
            f"Very low burstiness ({burst:.2f}) — sentence lengths are unusually uniform. "
            f"Humans typically mix short and long sentences."
        )
    elif burst < 0.4:
        reasons.append(
            f"Low burstiness ({burst:.2f}) — limited variation in sentence length."
        )
    elif burst > 0.7:
        reasons.append(
            f"High burstiness ({burst:.2f}) — strong variation in sentence length, typical of human writing."
        )

    # Vocabulary diversity
    ttr = feats["type_token_ratio"]
    if ttr > 0.85 and feats["word_count"] > 50:
        reasons.append(
            f"Very high vocabulary diversity (TTR={ttr:.2f}) — AI text often avoids word repetition."
        )
    elif ttr < 0.4:
        reasons.append(
            f"Low vocabulary diversity (TTR={ttr:.2f}) — heavy word repetition, more common in casual human writing."
        )

    # Function-word ratio
    fwr = feats["function_word_ratio"]
    if fwr < 0.15:
        reasons.append(
            f"Low function-word ratio ({fwr:.2f}) — fewer connecting words ('the', 'of', 'and'), "
            f"suggesting more formal or AI-like phrasing."
        )

    # Sentence-length signals
    mean_len = feats["mean_sentence_len"]
    if mean_len > 25:
        reasons.append(
            f"Long average sentence length ({mean_len:.1f} words) — AI tends toward verbose, compound sentences."
        )

    # Overall verdict line based on ai_prob
    if ai_prob > 0.75:
        verdict = "Overall: text shows strong signals of AI generation."
    elif ai_prob > 0.55:
        verdict = "Overall: text leans toward AI-generated, but with some human characteristics."
    elif ai_prob > 0.45:
        verdict = "Overall: signals are mixed — cannot confidently classify."
    elif ai_prob > 0.25:
        verdict = "Overall: text leans toward human-written."
    else:
        verdict = "Overall: text shows strong signals of human writing."

    # If no rules fired (text in a normal range), give a neutral explanation
    if not reasons:
        reasons.append("No strong individual signals detected — features are in a normal range.")

    reasons.append(verdict)
    return reasons