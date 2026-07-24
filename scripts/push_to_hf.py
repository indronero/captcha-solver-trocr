# scripts/push_to_hf.py

from huggingface_hub import login

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor
)

from src.config import REGISTERED_MODEL_NAME

from mlflow import MlflowClient

import mlflow.transformers
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

login(token=HF_TOKEN)

client = MlflowClient()

champion_version = client.get_model_version_by_alias(
    REGISTERED_MODEL_NAME,
    "champion"
).version

model_uri = f"models:/{REGISTERED_MODEL_NAME}/{champion_version}"

components = mlflow.transformers.load_model(
    model_uri,
    return_type="components"
)

model = components["model"]

tokenizer = components["tokenizer"]

image_processor = components["image_processor"]

processor = TrOCRProcessor(
    image_processor=image_processor,
    tokenizer=tokenizer
)

model.push_to_hub(HF_MODEL_ID)

processor.push_to_hub(HF_MODEL_ID)

print("Champion pushed to HF")