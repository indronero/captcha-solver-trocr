# TrOCR CAPTCHA Solver (MLOps)

Production-grade CAPTCHA OCR system built using Microsoft's TrOCR architecture with a complete MLOps workflow including:

- dataset cleaning and splitting
- training and fine-tuning
- MLflow model registry
- automated model evaluation
- champion/candidate promotion workflow
- batch inference
- FastAPI serving
- Hugging Face deployment support

The system fine-tunes:

```text
microsoft/trocr-small-printed
```

for CAPTCHA recognition using strong augmentation pipelines and automated benchmarking.

---

# Features

## OCR + Training

- TrOCR fine-tuning for CAPTCHA decoding
- Multi-dataset training support
- Heavy augmentation for robustness
- Automatic preprocessing pipeline
- GPU + CPU support

## MLOps

- MLflow experiment tracking
- MLflow model registry
- Champion / candidate workflow
- Automatic benchmark evaluation
- Automated model promotion
- Dataset lineage tracking
- Git metadata logging
- Versioned model lifecycle management

## Inference + Serving

- Single image inference
- Batch prediction
- FastAPI REST API
- Hugging Face model loading
- Alias-based model loading (`champion`, `candidate`)

---

# Base Model

```text
microsoft/trocr-small-printed
```

Original model:

https://huggingface.co/microsoft/trocr-small-printed

---

# Project Structure

```text
captcha_solver_mlops/

├── datasets/
│   ├── raw/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   │
│   ├── train/
│   ├── val/
│   └── test/
│
├── model/
│   └── trocr_finetuned_captcha_model/
│       └── v1/
│
├── outputs/
│   └── evaluation_outputs/
│
├── cache/
│
├── scripts/
│   ├── bootstrap_mlflow.py
│   ├── run_clean_split.py
│   ├── run_training.py
│   ├── run_inference.py
│   └── run_evaluation.py
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   └── clean_split.py
│   │
│   ├── training/
│   │   ├── dataset_loader.py
│   │   ├── preprocess.py
│   │   ├── trainer.py
│   │   └── train_pipeline.py
│   │
│   ├── inference/
│   │   ├── model_loader.py
│   │   ├── predictor.py
│   │   └── batch_predict.py
│   │
│   ├── evaluation/
│   │   ├── evaluation_pipeline.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── model_compare.py
│   │   ├── promote_model.py
│   │   └── report.py
│   │
│   └── config.py
│
├── mlruns/
├── mlflow.db
│
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

# Dataset Format

CAPTCHA labels are derived directly from filenames.

Example:

```text
A9K2L.png
```

Label:

```text
A9K2L
```

Dataset example:

```text
dataset/
├── A9K2L.png
├── X2P8T.png
└── T7Q9D.png
```

---

# Dataset Pipeline

This project follows a strict workflow:

```text
RAW DATA
   ↓
clean_split.py
   ↓
TRAIN / VAL / TEST SPLITS
   ↓
TRAINING + EVALUATION
```

---

# Dataset Cleaning & Splitting

Run:

```bash
python scripts/run_clean_split.py
```

Pipeline automatically:

- removes corrupted images
- removes invalid labels
- removes duplicate labels within datasets
- converts images to RGB
- normalizes outputs to `.png`
- creates deterministic train/val/test splits

Generated structure:

```text
datasets/
├── train/
├── val/
└── test/
```

---

# MLflow Setup

This project uses MLflow for:

- experiment tracking
- model registry
- model lineage
- benchmarking
- promotion workflows

---

# Bootstrap Initial Registry Model

You can either train from scratch (in which case skip this step) or bootstrap an existing model as the first version:

```bash
python scripts/bootstrap_mlflow.py
```

This will:

- create MLflow experiment
- register initial model
- create version `v1`
- assign `champion` alias
- initialize registry metadata

---

# Start MLflow UI

```bash
mlflow ui
```

Default:

```text
http://127.0.0.1:5000
```

---

# Training

Run training:

```bash
python scripts/run_training.py
```

Training pipeline:

- loads current champion model
- loads datasets
- applies augmentations
- fine-tunes TrOCR
- evaluates on benchmark test set
- logs metrics to MLflow
- registers candidate model automatically

---

# Training Workflow

```text
Champion Model
      ↓
Fine-tuning
      ↓
Candidate Model
      ↓
Benchmark Evaluation
      ↓
Promotion Decision
```

---

# Model Registry Workflow

## Aliases

| Alias     | Purpose                     |
| ---------- | --------------------------- |
| champion   | Current production model    |
| candidate  | Latest trained model        |

---

## Lifecycle Tags

| Status              | Meaning                     |
| ------------------- | --------------------------- |
| champion            | Production model            |
| pending_evaluation  | Awaiting benchmark testing  |
| archived            | Older champion              |
| rejected            | Failed benchmark comparison |

---

# Evaluation

Run evaluation pipeline:

```bash
python scripts/run_evaluation.py
```

Or evaluate a specific version:

```bash
python scripts/run_evaluation.py 3
```

Pipeline:

```text
evaluation_pipeline.py
    ↓
model_compare.py
    ↓
evaluator.py
    ↓
metrics.py
    ↓
promote_model.py
```

---

# Metrics

Evaluation computes:

- CER (Character Error Rate)
- WER (Word Error Rate)
- Exact Match Accuracy

Example:

```json
{
  "cer": 0.021,
  "wer": 0.097,
  "accuracy": 0.91
}
```

---

# Model Promotion Logic

Models are ranked using:

1. Lowest CER
2. Highest Accuracy

Best model automatically becomes:

```text
champion
```

Older champion becomes:

```text
archived
```

---

# Inference

Single image prediction:

```bash
python scripts/run_inference.py image.png
```

Example:

```bash
python scripts/run_inference.py samples/captcha.png
```

Output:

```text
AB12C
```

---

# Batch Prediction

Batch inference:

```python
from src.inference.batch_predict import batch_predict

results = batch_predict("datasets/test/dataset_1")
```

---

# FastAPI Inference Server

Run API:

```bash
uvicorn src.api.app:app --reload
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```http
GET /health
```

---

## Model Info

```http
GET /model
```

---

## Predict CAPTCHA

```http
POST /predict
```

Upload image file and receive prediction.

---

# Hugging Face Deployment

The API loads models directly from Hugging Face.

Environment variables:

```env
HF_MODEL_ID=your-username/captcha-ocr
HF_TOKEN=your_hf_token
HF_MODEL_REVISION=main
```

---

# Training Configuration

| Parameter                     | Value                         |
| ----------------------------- | ----------------------------- |
| Base Model                    | microsoft/trocr-small-printed |
| Train Batch Size              | 16                            |
| Eval Batch Size               | 16                            |
| Gradient Accumulation         | 2                             |
| Learning Rate                 | 1.5e-5                        |
| Scheduler                     | Cosine                        |
| Warmup Steps                  | 500                           |
| Weight Decay                  | 0.01                          |
| Beam Search                   | 4                             |
| Max Tokens                    | 10                            |
| Gradient Checkpointing        | Enabled                       |

---

# Data Augmentation

Albumentations pipeline includes:

- affine transforms
- optical distortion
- elastic transforms
- blur
- Gaussian noise
- coarse dropout
- brightness/contrast changes
- hue/saturation shifts

Designed specifically for CAPTCHA robustness.

---

# Hardware

Recommended:

- NVIDIA GPU with CUDA

Supported fallback:

- CPU inference/training

Automatic device detection included.

---

# Environment Setup

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Important Git Ignore Rules

The following are intentionally excluded from Git:

- datasets
- trained models
- MLflow runs
- MLflow database
- `.env`
- caches
- outputs

This keeps the repository lightweight and production-safe.

---

# Scripts

| Script                     | Purpose                                  |
| -------------------------- | ---------------------------------------- |
| bootstrap_mlflow.py        | Initialize MLflow registry               |
| run_clean_split.py         | Clean and split raw datasets             |
| run_training.py            | Train and register candidate model       |
| run_inference.py           | Single image inference                   |
| run_evaluation.py          | Benchmark and promotion pipeline         |

---

# MLOps Principles Used

This project follows production-grade MLOps practices:

- experiment tracking
- reproducible training
- automated evaluation
- model registry workflows
- dataset lineage tracking
- modular architecture
- alias-based deployments
- versioned model lifecycle management
- production-ready API serving

The architecture is designed for easy extension with:

- Docker
- CI/CD
- Kubernetes
- GPU cloud deployments
- scheduled retraining
- monitoring systems

---

# License

This repository uses Microsoft's TrOCR model.

Base model license and details:

https://huggingface.co/microsoft/trocr-small-printed