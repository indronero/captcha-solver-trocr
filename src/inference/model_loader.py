# src\inference\model_loader.py

import torch
import mlflow.transformers

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from src.config import REGISTERED_MODEL_NAME


def load_model(alias="champion", model_uri=None):

    if model_uri is None:

        model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}@{alias}"
        )

    print("Loading model from MLflow:", model_uri)

    # LOAD TRANSFORMERS MODEL
    components = mlflow.transformers.load_model(
        model_uri,
        return_type="components"
    )

    # components is dict-like
    model = components["model"]

    tokenizer = components["tokenizer"]

    image_processor = components["image_processor"]

    # rebuild processor
    processor = TrOCRProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    model.eval()

    return model, processor, device