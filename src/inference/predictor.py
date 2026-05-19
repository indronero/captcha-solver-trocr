# src\inference\predictor.py

import torch
from PIL import Image

from src.inference.model_loader import load_model
from src.config import MAX_LENGTH, NUM_BEAMS


model = None
processor = None
device = None

current_model_key = None


def predict(image_path, alias="champion", model_uri=None):

    global model
    global processor
    global device
    global current_model_key

    model_key = model_uri if model_uri else alias

    # First load
    if model is None:

        model, processor, device = load_model(
            alias=alias,
            model_uri=model_uri
        )

        current_model_key = model_key

    # Reload if model changes
    elif model_key != current_model_key:

        model, processor, device = load_model(
            alias=alias,
            model_uri=model_uri
        )

        current_model_key = model_key

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=MAX_LENGTH,
            num_beams=NUM_BEAMS
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text