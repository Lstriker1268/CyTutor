import json
import math
from pathlib import Path
from collections import Counter

BASE_DIR = Path.home() / "cytutor-baseline"
FILES = [
    ("Original baseline", BASE_DIR / "baseline_results.jsonl"),
    ("Updated baseline", BASE_DIR / "baseline_results_v2.jsonl"),
    ("4-bit baseline", BASE_DIR / "baseline_results_4bit.jsonl"),
]

def tokenize(text):
    return text.lower().split()

def sentence_bleu(reference, prediction, max_n=4):
    ref = tokenize(reference)
    pred = tokenize(prediction)

    if not pred:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred[i:i+n]) for i in range(len(pred) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)
        )

        total = sum(pred_ngrams.values())
        matches = sum(
            min(count, ref_ngrams[gram])
            for gram, count in pred_ngrams.items()
        )

        # Add-one smoothing for short answers
        precisions.append((matches + 1) / (total + 1))

    brevity_penalty = (
        1.0 if len(pred) > len(ref)
        else math.exp(1 - len(ref) / len(pred))
    )

    return brevity_penalty * math.exp(
        sum(math.log(p) for p in precisions) / max_n
    )

for label, path in FILES:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    scores = [
        sentence_bleu(row["reference"], row["prediction"])
        for row in rows
    ]

    print(f"\n{label}: {len(scores)} questions")
    for i, score in enumerate(scores, 1):
        print(f"  Question {i}: BLEU-4 = {score:.4f}")
    print(f"  Mean BLEU-4: {sum(scores) / len(scores):.4f}")
