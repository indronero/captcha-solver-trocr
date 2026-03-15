import torch
from PIL import Image

from src.inference.model_loader import load_model
from src.config import MAX_LENGTH, NUM_BEAMS


model, processor, device = load_model()


def predict(image_path, version=None):

    global model, processor, device

    if version is not None:
        model, processor, device = load_model(version)

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