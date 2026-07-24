# src/evaluation/metrics.py

import numpy as np
from evaluate import load

cer_metric = load("cer")
wer_metric = load("wer")


def compute_metrics(predictions, labels):

    preds = [p.strip() for p in predictions]
    refs = [l.strip() for l in labels]

    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)

    exact_match = np.mean([p == r for p, r in zip(preds, refs)])

    return {
        "cer": cer,
        "wer": wer,
        "accuracy": exact_match
    }