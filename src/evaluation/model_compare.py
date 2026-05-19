# src\evaluation\model_compare.py

from src.evaluation.evaluator import evaluate
from src.config import get_test_paths


def compare_models(model_uris):

    print("\n=== MODEL COMPARISON ===")

    test_paths = get_test_paths()

    results = []

    for model_uri in model_uris:

        print(f"\nEvaluating: {model_uri}")

        metrics = evaluate(
            test_paths,
            model_uri=model_uri,
            save_reports=False
        )

        results.append({
            "model_uri": model_uri,
            "metrics": metrics
        })

    ranked = sorted(
        results,
        key=lambda x: (
            x["metrics"]["cer"],
            -x["metrics"]["accuracy"]
        )
    )

    winner = ranked[0]

    print("\n=== FINAL RANKING ===")

    for r in ranked:

        print(r["model_uri"], r["metrics"])

    return winner, ranked