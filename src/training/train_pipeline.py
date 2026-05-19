# src\training\train_pipeline.py

import os
import torch
import mlflow
import mlflow.transformers
import subprocess
from datetime import datetime

from mlflow import MlflowClient

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor
)

from src.config import (
    MODEL_NAME,
    NUM_EXAMPLES,
    get_train_paths,
    get_val_paths,
    get_test_paths,
    MLFLOW_EXPERIMENT,
    REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI
)

from src.training.dataset_loader import load_datasets
from src.training.preprocess import preprocess_function
from src.training.trainer import build_trainer
from src.evaluation.evaluator import evaluate

def train():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = MlflowClient()

    with mlflow.start_run(run_name="training_run") as run:     

        # =========================
        # LOAD CURRENT CHAMPION
        # =========================

        try:

            champion_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

            print("Loading champion model...")

            components = mlflow.transformers.load_model(
                champion_uri,
                return_type="components"
            )

            model = components["model"]

            tokenizer = components["tokenizer"]

            image_processor = components["image_processor"]

            processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer
            )

            champion_version = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                "champion"
            ).version

            mlflow.set_tag(
                "training_source",
                f"champion_v{champion_version}"
            )

        except Exception as e:

            print(e)

            print("No champion found. Loading base model.")
            
            champion_version = "base"

            model = VisionEncoderDecoderModel.from_pretrained(
                MODEL_NAME
            )

            processor = TrOCRProcessor.from_pretrained(
                MODEL_NAME
            )

            mlflow.set_tag(
                "training_source",
                "base_model"
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

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model.to(device)

        # =========================
        # DATASETS
        # =========================

        print("Loading datasets...")

        train_dataset = load_datasets(
            get_train_paths(),
            NUM_EXAMPLES
        )

        val_dataset = load_datasets(
            get_val_paths()
        )
        
        test_dataset = load_datasets(
            get_test_paths()
        )
        
        # =========================
        # DATASET LINEAGE
        # =========================

        train_df = train_dataset.to_pandas()

        val_df = val_dataset.to_pandas()

        test_df = test_dataset.to_pandas()

        train_sources = ",".join(
            sorted(train_df["source"].unique())
        )

        val_sources = ",".join(
            sorted(val_df["source"].unique())
        )
        
        test_sources = ",".join(
            sorted(test_df["source"].unique())
        )

        train_data = mlflow.data.from_pandas(
            train_df,
            source=train_sources,
            name="captcha_train_dataset"
        )

        val_data = mlflow.data.from_pandas(
            val_df,
            source=val_sources,
            name="captcha_val_dataset"
        )
        
        test_data = mlflow.data.from_pandas(
            test_df,
            source=test_sources,
            name="captcha_test_dataset"
        )

        mlflow.log_input(
            train_data,
            context="training"
        )

        mlflow.log_input(
            val_data,
            context="validation"
        )
        
        mlflow.log_input(
            test_data,
            context="testing"
        )

        # =========================
        # PARAMS
        # =========================

        actual_train_samples = len(train_dataset)

        actual_val_samples = len(val_dataset)

        mlflow.log_param(
            "total_samples",
            actual_train_samples + actual_val_samples
        )

        mlflow.log_param(
            "train_samples",
            actual_train_samples
        )

        mlflow.log_param(
            "val_samples",
            actual_val_samples
        )

        mlflow.log_param(
            "base_model",
            MODEL_NAME
        )

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
        
        run_description = f"""
Captcha OCR training run

Base model: {MODEL_NAME}
Training source: {champion_version}
Git branch: {git_branch}

Datasets:
- Train: {train_sources}
- Validation: {val_sources}
- Test: {test_sources}

Timestamp: {datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}
"""

        mlflow.set_tag(
            "mlflow.note.content",
            run_description
        )

        print("Preprocessing datasets...")

        train_dataset = train_dataset.map(
            lambda ex: preprocess_function(ex, True),
            batched=True,
            num_proc=2,
            remove_columns=train_dataset.column_names
        )

        val_dataset = val_dataset.map(
            lambda ex: preprocess_function(ex, False),
            batched=True,
            num_proc=2,
            remove_columns=val_dataset.column_names
        )

        train_dataset.set_format("torch")

        val_dataset.set_format("torch")

        # =========================
        # TRAINER
        # =========================

        trainer = build_trainer(
            model,
            train_dataset,
            val_dataset,
            "temp_training_output"
        )

        print("Starting training...")

        trainer.train()

        # =========================
        # LOG MODEL
        # =========================

        print("Logging candidate model to MLflow...")

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

        latest_versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        latest_version = max(
            latest_versions,
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
- candidate = latest evaluated model

Lifecycle tags:
- pending_evaluation
- champion
- archived
- rejected
"""
            )

        except:
            pass
        
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "source_run_id",
            run.info.run_id
        )

        # =========================
        # CLEAN OLD CANDIDATE TAGS
        # =========================

        # for mv in latest_versions:

        #     try:

        #         client.delete_model_version_tag(
        #             REGISTERED_MODEL_NAME,
        #             mv.version,
        #             "status"
        #         )

        #     except:
        #         pass

        # =========================
        # INITIAL TAGS
        # =========================

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_status",
            "pending_evaluation"
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "trained_from",
            f"v{champion_version}"
        )
        
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "benchmark_test_sources",
            test_sources
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "benchmark_test_samples",
            str(len(test_dataset))
        )

        # =========================
        # CANDIDATE ALIAS
        # =========================

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "candidate",
            version
        )

        # =========================
        # EVALUATE
        # =========================

        model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}/{version}"
        )

        metrics = evaluate(get_test_paths(), model_uri=model_uri, save_reports=False)
        
        version_description = f"""
Captcha OCR model version v{version}

Training source:
- v{champion_version}

Datasets:
- Train: {train_sources}
- Validation: {val_sources}
- Test: {test_sources}

Run ID:
- {run.info.run_id}

Timestamp:
- {datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}
"""

        client.update_model_version(
            name=REGISTERED_MODEL_NAME,
            version=version,
            description=version_description
        )

        # =========================
        # LOG RUN METRICS
        # =========================

        mlflow.log_metric(
            "cer",
            metrics["cer"],
            model_id=model_info.model_id,
            dataset=test_data
        )

        mlflow.log_metric(
            "wer",
            metrics["wer"],
            model_id=model_info.model_id,
            dataset=test_data
        )

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
            model_id=model_info.model_id,
            dataset=test_data
        )

        # =========================
        # LOG VERSION METRICS
        # =========================

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "training_benchmark_cer",
            str(metrics["cer"])
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "training_benchmark_wer",
            str(metrics["wer"])
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "training_benchmark_accuracy",
            str(metrics["accuracy"])
        )
        
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "training_benchmark_timestamp",
            datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        )

        print("\nTraining pipeline complete.")