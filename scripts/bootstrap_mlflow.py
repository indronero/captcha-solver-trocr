# scripts\bootstrap_mlflow.py

import os
import shutil
import subprocess
import mlflow
import mlflow.transformers
from datetime import datetime

from mlflow import MlflowClient
from src.config import BOOTSTRAP_MODEL, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, REGISTERED_MODEL_NAME, MLRUNS_DIR, MLFLOW_DB


from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor
)

# =========================
# CLEAN OLD MLFLOW FILES
# =========================

if os.path.exists(MLRUNS_DIR):
    shutil.rmtree(MLRUNS_DIR)
    print("Deleted old mlruns/")

if os.path.exists(MLFLOW_DB):
    os.remove(MLFLOW_DB)
    print("Deleted old mlflow.db")


# =========================
# SET EXPERIMENT
# =========================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# =========================
# START RUN
# =========================

with mlflow.start_run(run_name="bootstrap_v1") as run:

    # =========================
    # GIT METADATA
    # =========================

    try:

        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True
        ).strip()

        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()

        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True
            ).strip()
        )

    except:

        git_branch = "NA"

        git_commit = "NA"

        git_dirty = False

    mlflow.set_tag(
        "git_branch",
        git_branch
    )

    mlflow.set_tag(
        "git_commit",
        git_commit
    )

    mlflow.set_tag(
        "git_dirty",
        str(git_dirty)
    )

    # =========================
    # RUN DESCRIPTION
    # =========================

    run_description = f"""
Bootstrap run for initial OCR model registration.

Source:
- Local pretrained checkpoint

Model path:
- {BOOTSTRAP_MODEL}

Git branch:
- {git_branch}

Timestamp:
- {datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}
"""

    mlflow.set_tag(
        "mlflow.note.content",
        run_description
    )

    print("Loading local v1 model...")

    model = VisionEncoderDecoderModel.from_pretrained(
        BOOTSTRAP_MODEL
    )

    processor = TrOCRProcessor.from_pretrained(
        BOOTSTRAP_MODEL
    )

    # =========================
    # MODEL CONFIG
    # =========================

    model.config.decoder_start_token_id = (
        processor.tokenizer.eos_token_id
    )

    model.config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        processor.tokenizer.eos_token_id
    )

    model.config.vocab_size = (
        processor.tokenizer.vocab_size
    )

    if (
        model.config.decoder.vocab_size
        != processor.tokenizer.vocab_size
    ):

        model.decoder.resize_token_embeddings(
            len(processor.tokenizer)
        )

    print("Logging model to MLflow...")

    model_info = mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "image_processor": processor.image_processor,
            "tokenizer": processor.tokenizer,
        },
        processor=processor,
        name="model",
        task="image-to-text",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    # =========================
    # RUN TAGS
    # =========================

    mlflow.set_tag(
        "baseline",
        "true"
    )

    mlflow.set_tag(
        "environment",
        "production"
    )

    mlflow.set_tag(
        "model_type",
        "trocr"
    )

    mlflow.set_tag(
        "source",
        "bootstrap_local_v1"
    )

    mlflow.set_tag(
        "training_source",
        "bootstrap_local_model"
    )

    print("Model logged successfully")


# =========================
# MODEL REGISTRY SETUP
# =========================

client = MlflowClient()

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

latest_version = max(
    versions,
    key=lambda v: int(v.version)
)

version = latest_version.version

try:

    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description="""
Captcha OCR model registry.

Tracks all OCR model versions, metrics, datasets,
git metadata, evaluation reports, and promotion status.

Aliases:
- champion = current production model
- candidate = latest pending model

Lifecycle tags:
- pending_evaluation
- champion
- archived
- rejected
"""
    )

except:
    pass

# =========================
# VERSION TAGS
# =========================

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "source_run_id",
    run.info.run_id
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "lifecycle_status",
    "champion"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "trained_from",
    "bootstrap"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "bootstrap_model_path",
    BOOTSTRAP_MODEL
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "benchmark_test_sources",
    "bootstrap_unknown"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "benchmark_test_samples",
    "unknown"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "training_benchmark_cer",
    "not_evaluated"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "training_benchmark_wer",
    "not_evaluated"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "training_benchmark_accuracy",
    "not_evaluated"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "training_benchmark_timestamp",
    "not_evaluated"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "git_branch",
    git_branch
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "git_commit",
    git_commit
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version,
    "git_dirty",
    str(git_dirty)
)

# =========================
# CHAMPION ALIAS
# =========================

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    version
)

# =========================
# VERSION DESCRIPTION
# =========================

client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=version,
    description=f"""
Bootstrap model version v{version}

Source:
- Local pretrained model

Path:
- {BOOTSTRAP_MODEL}

Run ID:
- {run.info.run_id}

Git branch:
- {git_branch}

Git commit:
- {git_commit}

Timestamp:
- {datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}
"""
)

print("\n=================================")
print("BOOTSTRAP COMPLETE")
print("=================================")

print(f"Registered Model: {REGISTERED_MODEL_NAME}")

print(f"Version: {version}")

print("Alias assigned: champion")