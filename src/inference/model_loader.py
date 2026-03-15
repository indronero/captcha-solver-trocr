import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.config import MODEL_BASE_DIR
from src.utils.model_versioning import get_latest_version


def load_model(version=None):

    if version is None:
        model_path = get_latest_version(MODEL_BASE_DIR)
    else:
        model_path = f"{MODEL_BASE_DIR}/{version}"

    processor = TrOCRProcessor.from_pretrained(model_path)

    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    model.eval()

    print("Loaded model:", model_path)

    return model, processor, device