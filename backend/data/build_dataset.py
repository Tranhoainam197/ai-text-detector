"""
Downloads HC3 (Human ChatGPT Comparison Corpus) raw JSONL files directly
from HuggingFace and builds a balanced CSV with columns: text, label.
"""
import os
import json
import random
import pandas as pd
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "dataset.csv")
SAMPLES_PER_CLASS = 1000

# Direct URLs to the JSONL files in the HC3 repo
HC3_FILES = [
    "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl",
]


def fetch(url: str) -> list[dict]:
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    print(f"  -> {len(rows)} rows")
    return rows


def main():
    all_rows = []
    for url in HC3_FILES:
        all_rows.extend(fetch(url))

    human_texts, ai_texts = [], []
    for row in all_rows:
        for h in row.get("human_answers", []) or []:
            h = (h or "").strip()
            if 100 < len(h) < 3000:
                human_texts.append(h)
        for a in row.get("chatgpt_answers", []) or []:
            a = (a or "").strip()
            if 100 < len(a) < 3000:
                ai_texts.append(a)

    print(f"Available: {len(human_texts)} human, {len(ai_texts)} AI")

    random.seed(42)
    random.shuffle(human_texts)
    random.shuffle(ai_texts)
    human_texts = human_texts[:SAMPLES_PER_CLASS]
    ai_texts = ai_texts[:SAMPLES_PER_CLASS]
    print(f"Using {len(human_texts)} human and {len(ai_texts)} AI samples.")

    df = pd.DataFrame(
        {"text": human_texts + ai_texts,
         "label": [0] * len(human_texts) + [1] * len(ai_texts)}
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()