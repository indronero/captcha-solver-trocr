# src\config.py

import os

# Check if subdirs exist and join
def get_subdirs(base_path):
    if not os.path.exists(base_path):
        raise ValueError(f"{base_path} does not exist")

    subdirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    if len(subdirs) == 0:
        raise ValueError(f"No datasets found in {base_path}")

    return subdirs

MODEL_NAME = "microsoft/trocr-small-printed"

NUM_EXAMPLES = 15000

RAW_PATH = "datasets/raw"

TRAIN_BASE = "datasets/train"
VAL_BASE = "datasets/val"
TEST_BASE = "datasets/test"

def get_train_paths():
    return get_subdirs(TRAIN_BASE)

def get_val_paths():
    return get_subdirs(VAL_BASE)

def get_test_paths():
    return get_subdirs(TEST_BASE)

EVAL_OUT = "outputs/evaluation_outputs"


TRAIN_CACHE = "cache/train_cache.arrow"
VAL_CACHE = "cache/val_cache.arrow"

MODEL_BASE_DIR = "model/trocr_finetuned_captcha_model"
BOOTSTRAP_MODEL = "model/trocr_finetuned_captcha_model/v1"

MAX_LENGTH = 10
NUM_BEAMS = 4

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# =========================
# MLFLOW
# =========================

MLFLOW_EXPERIMENT = "captcha_ocr"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "captcha_ocr_model"

MLRUNS_DIR = "mlruns"

MLFLOW_DB = "mlflow.db"

# To run : mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 --allowed-hosts "*"
