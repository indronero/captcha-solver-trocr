MODEL_NAME = "microsoft/trocr-small-printed"

NUM_EXAMPLES = 15000

DATASET_PATHS = [
    "datasets\dataset_1"
]

TRAIN_CACHE = "cache/train_cache.arrow"
VAL_CACHE = "cache/val_cache.arrow"

MODEL_BASE_DIR = "model/trocr_finetuned_captcha_model"

MAX_LENGTH = 10
NUM_BEAMS = 4

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")