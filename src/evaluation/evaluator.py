# src\evaluation\evaluator.py

from src.inference.batch_predict import batch_predict
from src.evaluation.metrics import compute_metrics
from src.evaluation.report import save_report


def evaluate(folder, alias=None, model_uri=None,save_reports=False):

    results = []

    folders = folder if isinstance(folder, list) else [folder]

    for f in folders:

        results.extend(
            batch_predict(f, alias=alias, model_uri=model_uri)
        )

    predictions = [
        r["prediction"]
        for r in results
    ]

    labels = [
        r["label"]
        for r in results
    ]

    metrics = compute_metrics(
        predictions,
        labels
    )

    # =========================
    # OPTIONAL REPORT SAVING
    # =========================

    if save_reports:

        report = {
            "model_alias": alias,
            "model_uri": model_uri,
            "num_samples": len(results),
            "metrics": metrics,
            "samples": results[:50]
        }

        save_report(report)

    print(
        "Evaluation Results:",
        metrics
    )

    return metrics