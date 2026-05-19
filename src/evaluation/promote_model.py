# src\evaluation\promote_model.py

from mlflow import MlflowClient

from src.config import REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI


def promote_model(winner_version, ranked_versions, champion_version=None):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    print(f"\nPromoting v{winner_version} to champion")

    # =========================
    # ARCHIVE OLD CHAMPION
    # =========================

    if (
        champion_version is not None
        and champion_version != winner_version
    ):

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            champion_version,
            "lifecycle_status",
            "archived"
        )

    # =========================
    # SET NEW CHAMPION
    # =========================

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        winner_version
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        winner_version,
        "lifecycle_status",
        "champion"
    )

    # =========================
    # REJECT OTHERS
    # =========================

    for version in ranked_versions:

        # skip winner
        if version == winner_version:
            continue

        # old champion already archived above
        if version == champion_version:
            continue

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_status",
            "rejected"
        )