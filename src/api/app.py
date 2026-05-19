# src\api\app.py

import io
import os
import torch

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from PIL import Image

from huggingface_hub import login

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from src.config import (
    MAX_LENGTH,
    NUM_BEAMS
)


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID")

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL_REVISION = os.getenv(
    "HF_MODEL_REVISION",
    "main"
)


# =========================
# APP LIFECYCLE
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n=================================")
    print("STARTING OCR API")
    print("=================================")

    try:

        # =========================
        # VALIDATE ENV
        # =========================

        if not HF_MODEL_ID:

            raise ValueError(
                "HF_MODEL_ID missing in environment"
            )

        # =========================
        # HF AUTH
        # =========================

        if HF_TOKEN:

            print("Authenticating with Hugging Face...")

            login(
                token=HF_TOKEN,
                add_to_git_credential=False
            )

        # =========================
        # DEVICE
        # =========================

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Device:", device)

        # =========================
        # LOAD MODEL
        # =========================

        print("Loading model from Hugging Face...")

        model = VisionEncoderDecoderModel.from_pretrained(
            HF_MODEL_ID,
            revision=HF_MODEL_REVISION
        )

        processor = TrOCRProcessor.from_pretrained(
            HF_MODEL_ID,
            revision=HF_MODEL_REVISION
        )

        model.to(device)

        model.eval()

        # =========================
        # APP STATE
        # =========================

        app.state.model = model
        app.state.processor = processor
        app.state.device = device

        print("Model loaded successfully")

        yield

    except Exception as e:

        print("Startup failed:", str(e))

        raise e

    finally:

        print("\nShutting down API...")

        app.state.model = None
        app.state.processor = None
        app.state.device = None

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print("Resources cleaned up")


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="OCR MLOps API",
    lifespan=lifespan
)


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "device": str(app.state.device)
    }


# =========================
# MODEL INFO
# =========================

@app.get("/model")
def model_info():

    return {
        "model_source": "huggingface",
        "model_id": HF_MODEL_ID,
        "revision": HF_MODEL_REVISION,
        "device": str(app.state.device)
    }


# =========================
# PREDICT ENDPOINT
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        print("\n=== NEW REQUEST ===")

        # =========================
        # VALIDATE MODEL
        # =========================

        if app.state.model is None:

            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        model = app.state.model
        processor = app.state.processor
        device = app.state.device

        # =========================
        # READ IMAGE
        # =========================

        print("Reading image...")

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print("Image loaded:", image.size)

        # =========================
        # PREPROCESS
        # =========================

        print("Preprocessing...")

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(device)

        # =========================
        # INFERENCE
        # =========================

        print("Running inference...")

        with torch.no_grad():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=MAX_LENGTH,
                num_beams=NUM_BEAMS
            )

        # =========================
        # DECODE
        # =========================

        print("Decoding...")

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print("Prediction:", text)

        return {
            "prediction": text,
            "model_source": "huggingface",
            "model_id": HF_MODEL_ID,
            "revision": HF_MODEL_REVISION
        }

    except HTTPException:
        raise

    except Exception as e:

        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )