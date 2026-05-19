# scripts\run_evaluation.py

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.evaluation.evaluation_pipeline import (
    run_evaluation_pipeline
)

from src.config import REGISTERED_MODEL_NAME


if __name__ == "__main__":

    additional_model_uri = None

    if len(sys.argv) > 1:

        version = sys.argv[1]

        additional_model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}/{version}"
        )

    run_evaluation_pipeline(
        additional_model_uri=additional_model_uri
    )