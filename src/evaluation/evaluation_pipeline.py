# src\evaluation\evaluation_pipeline.py

import mlflow
from mlflow import MlflowClient
from datetime import datetime

from src.config import REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI
from src.evaluation.model_compare import compare_models
from src.evaluation.promote_model import promote_model


def run_evaluation_pipeline(
    additional_model_uri=None
):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_uris = set()

    ranked_versions = []

    # =========================
    # CHAMPION
    # =========================

    champion_version = None

    try:

        champion_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "champion"
        ).version

        champion_uri = (
            f"models:/{REGISTERED_MODEL_NAME}/{champion_version}"
        )

        model_uris.add(champion_uri)

    except:
        pass

    # =========================
    # PENDING MODELS
    # =========================

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    for mv in versions:

        status = mv.tags.get(
            "lifecycle_status",
            ""
        )

        if status != "pending_evaluation":
            continue

        uri = (
            f"models:/{REGISTERED_MODEL_NAME}/{mv.version}"
        )

        model_uris.add(uri)

    # =========================
    # OPTIONAL EXTRA MODEL
    # =========================

    if (
        additional_model_uri is not None
        and additional_model_uri not in model_uris
    ):

        model_uris.add(additional_model_uri)

    # =========================
    # NOTHING TO EVALUATE
    # =========================
    
    model_uris = list(model_uris)

    if len(model_uris) <= 1:

        print("\nNo pending models to evaluate.")

        return

    # =========================
    # COMPARE
    # =========================

    winner, ranked = compare_models(model_uris)

    winner_uri = winner["model_uri"]

    winner_version = winner_uri.split("/")[-1]

    for r in ranked:

        version = r["model_uri"].split("/")[-1]

        ranked_versions.append(version)

        metrics = r["metrics"]

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "latest_benchmark_cer",
            str(metrics["cer"])
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "latest_benchmark_wer",
            str(metrics["wer"])
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "latest_benchmark_accuracy",
            str(metrics["accuracy"])
        )
        
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "latest_benchmark_timestamp",
            datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        )

    # =========================
    # PROMOTE
    # =========================

    promote_model(
        winner_version=winner_version,
        ranked_versions=ranked_versions,
        champion_version=champion_version
    )