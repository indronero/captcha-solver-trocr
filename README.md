---
title: Captcha Solver Api
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Fine Tuned TrOCR CAPTCHA OCR API with FastAPI, CI/CD & MLOps
---

# TrOCR CAPTCHA Solver 

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![MLflow](https://img.shields.io/badge/MLflow-Registry-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-style Optical Character Recognition (OCR) system for CAPTCHA recognition built around Microsoft's TrOCR architecture and a complete end-to-end MLOps workflow.

Unlike a standalone training notebook, this project implements the complete machine learning lifecycle including dataset preparation, model training, experiment tracking, model registry, automated evaluation, champion promotion, production deployment, API serving, monitoring, and CI/CD automation.

The model is fine-tuned from:

```text
microsoft/trocr-small-printed
```

using custom CAPTCHA datasets with an enterprise-inspired deployment workflow built around MLflow and Hugging Face.


### Production APISwagger UI
https://indronel-captcha-solver-api.hf.space/docs


### Model Repository
https://huggingface.co/Indronel/captcha-ocr-model


---

# Production Architecture

```text
                   Raw CAPTCHA Datasets
                           │
                           ▼
                Dataset Cleaning & Validation
                           │
                           ▼
                 Train / Validation / Test Split
                           │
                           ▼
                  TrOCR Fine-tuning Pipeline
                           │
                           ▼
                MLflow Experiment Tracking
                           │
                           ▼
                  MLflow Model Registry
                           │
                 Champion / Candidate Aliases
                           │
                           ▼
              Push Champion Model to Hugging Face
                           │
                           ▼
              Hugging Face Model Repository
                           │
                           ▼
              FastAPI Production Inference API
                           │
                           ▼
                    Docker Container
                           │
                           ▼
               Hugging Face Spaces Deployment
                           │
                           ▼
                  Production REST API
```

---

# Project Highlights

## OCR & Training

- Fine-tuned Microsoft TrOCR model for CAPTCHA recognition
- Automatic dataset cleaning and validation
- Multi-dataset training support
- Albumentations-based augmentation pipeline
- GPU and CPU support
- Batch inference utilities
- Configurable training pipeline

## MLOps

- MLflow experiment tracking
- MLflow Model Registry
- Champion / Candidate model lifecycle
- Automated evaluation pipeline
- Model comparison and promotion workflow
- Dataset lineage
- Git metadata logging
- Versioned model management
- Reproducible training pipeline

## Deployment

- FastAPI inference service
- Dockerized deployment
- Hugging Face Model Hub integration
- Hugging Face Spaces hosting
- GitHub Actions CI
- GitHub Actions CD
- Automated production deployment

## Monitoring

- Prometheus metrics
- Grafana dashboard provisioning
- Request latency monitoring
- Request count monitoring
- Health endpoints

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| OCR | TrOCR, VisionEncoderDecoderModel |
| Framework | PyTorch, Transformers |
| Dataset Processing | Pillow, Albumentations |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Registry |
| Production Model Storage | Hugging Face Model Hub |
| API | FastAPI |
| Deployment | Docker, Hugging Face Spaces |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |
| Testing | Pytest |

---

# Base Model

```text
microsoft/trocr-small-printed
```

Original model:

https://huggingface.co/microsoft/trocr-small-printed

---

# Production Workflow

The project follows a production-oriented workflow where MLflow manages experimentation while Hugging Face serves as the production model repository.

```text
Developer
    │
    ▼
Train Model
    │
    ▼
MLflow Tracking
    │
    ▼
Register Candidate
    │
    ▼
Evaluation Pipeline
    │
    ▼
Champion Promotion
    │
    ▼
Push Champion
    │
    ▼
Hugging Face Model Repository
    │
    ▼
FastAPI Startup
    │
    ▼
Download Production Model
    │
    ▼
Serve Predictions
```

---

# Repository Structure

```text
captcha-solver-trocr/

├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       └── sync_hf.yml
│
├── datasets/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│
├── scripts/
│   ├── bootstrap_mlflow.py
│   ├── push_to_hf.py
│   ├── run_clean_split.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_inference.py
│
├── src/
│   ├── api/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── monitoring/
│   └── config.py
│
├── tests/
│   ├── sample_data/
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_training_smoke.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.cpu.txt
├── LICENSE
└── README.md
```

---

# Repository Components

| Directory | Purpose |
|-----------|---------|
| `datasets/` | Raw and processed CAPTCHA datasets |
| `scripts/` | Entry points for each pipeline stage |
| `src/training` | Model training pipeline |
| `src/evaluation` | Evaluation, comparison and promotion logic |
| `src/inference` | Local and production inference |
| `src/api` | FastAPI application |
| `src/monitoring` | Prometheus metrics |
| `monitoring/` | Grafana and Prometheus configuration |
| `tests/` | Automated test suite |
| `.github/workflows` | Continuous Integration and Deployment |

---

# Dataset Format

CAPTCHA labels are derived directly from image filenames.

Example:

```text
AB12C.png
```

Label:

```text
AB12C
```

Example dataset:

```text
dataset/
├── AB12C.png
├── X8Y4P.png
└── K7LM9.png
```

---

# Dataset Pipeline

```text
Raw Dataset
      │
      ▼
Validation
      │
      ▼
Cleaning
      │
      ▼
Duplicate Removal
      │
      ▼
RGB Conversion
      │
      ▼
PNG Normalization
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Training Pipeline
```

---

# Dataset Cleaning & Splitting

Prepare the dataset by running:

```bash
python scripts/run_clean_split.py
```

The preprocessing pipeline automatically:

- validates dataset integrity
- removes corrupted images
- removes invalid labels
- removes duplicate labels within datasets
- converts all images to RGB
- normalizes images to PNG format
- generates deterministic train/validation/test splits

Output structure:

```text
datasets/
├── train/
├── val/
└── test/
```

---

# Training Pipeline

The project follows a reproducible training pipeline from dataset preparation through model registration.

```text
Raw Dataset
      │
      ▼
Clean & Split
      │
      ▼
Dataset Loader
      │
      ▼
Image Preprocessing
      │
      ▼
Albumentations
      │
      ▼
Fine-tune TrOCR
      │
      ▼
Evaluation
      │
      ▼
MLflow Logging
      │
      ▼
Register Candidate
```

Training automatically:

- loads datasets
- prepares processors
- applies augmentations
- fine-tunes the TrOCR model
- evaluates performance
- logs metrics and artifacts to MLflow
- registers the trained model as a Candidate

Run training:

```bash
python scripts/run_training.py
```

---

# MLflow Integration

MLflow is used as the experiment tracking and model registry layer throughout the training lifecycle.

Implemented features include:

- Experiment tracking
- Parameter logging
- Metric logging
- Artifact logging
- Model Registry
- Champion/Candidate aliases
- Version management
- Git metadata tracking
- Model lineage
- Benchmark history

---

# Bootstrap Registry

When starting with an existing fine-tuned checkpoint, bootstrap the registry using:

```bash
python scripts/bootstrap_mlflow.py
```

The bootstrap process:

- creates a fresh MLflow experiment
- initializes the Model Registry
- registers Version 1
- assigns the `champion` alias
- records Git metadata
- stores registry metadata

This provides an initial production model before subsequent training iterations.

---

# MLflow UI

Launch the tracking server:

```bash
mlflow ui
```

Default address:

```text
http://127.0.0.1:5000
```

The UI provides access to:

- experiments
- metrics
- artifacts
- registered models
- model versions
- aliases
- lineage

---

# Model Lifecycle

The repository implements a Champion / Candidate workflow inspired by production MLOps systems.

```text
Champion
     │
     ▼
Fine-tuning
     │
     ▼
Candidate
     │
     ▼
Evaluation
     │
     ▼
Comparison
     │
     ▼
Promotion Decision
```

---

# Registry Aliases

| Alias | Description |
|--------|-------------|
| `champion` | Current production model |
| `candidate` | Newly trained model awaiting promotion |

---

# Model Status

During evaluation, model versions move through the following lifecycle:

| Status | Description |
|---------|-------------|
| `champion` | Current production model |
| `candidate` | Newly registered model |
| `pending_evaluation` | Awaiting benchmark evaluation |
| `archived` | Previous production model |
| `rejected` | Candidate failed comparison |

---

# Evaluation Pipeline

Evaluate the latest candidate:

```bash
python scripts/run_evaluation.py
```

Evaluate a specific model version:

```bash
python scripts/run_evaluation.py <version>
```

Evaluation pipeline:

```text
Candidate Model
       │
       ▼
Prediction
       │
       ▼
Metric Computation
       │
       ▼
Champion Comparison
       │
       ▼
Promotion Decision
```

Pipeline components:

```text
evaluation_pipeline.py
        │
        ▼
model_compare.py
        │
        ▼
evaluator.py
        │
        ▼
metrics.py
        │
        ▼
promote_model.py
```

---

# Evaluation Metrics

Each evaluation computes:

- Character Error Rate (CER)
- Word Error Rate (WER)
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

# Promotion Strategy

Models are compared using benchmark metrics.

Primary ranking:

1. Lowest Character Error Rate (CER)
2. Highest Exact Match Accuracy

If the candidate outperforms the current production model:

```text
Candidate
      │
      ▼
Promoted to Champion
      │
      ▼
Previous Champion Archived
```

Otherwise:

```text
Candidate
      │
      ▼
Rejected
```

This ensures only the best-performing model is promoted to production.

---

# Hugging Face Model Synchronization

Once a Champion model is selected, it can be published to the production model repository.

Run:

```bash
python scripts/push_to_hf.py
```

The synchronization process:

- retrieves the current Champion from MLflow
- loads the model and processor
- authenticates with Hugging Face
- uploads the production model
- updates the Hugging Face Model Repository

The uploaded repository contains:

```text
model.safetensors
config.json
generation_config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
sentencepiece.bpe.model
```

This repository acts as the production model source used by the deployed API.

---

# Production Model Flow

```text
Training
     │
     ▼
MLflow Tracking
     │
     ▼
Register Candidate
     │
     ▼
Evaluate
     │
     ▼
Promote Champion
     │
     ▼
Push Champion
     │
     ▼
Hugging Face Model Repository
```

Unlike embedding models inside the Docker image, the production API downloads the latest production model directly from the Hugging Face Model Repository during application startup, allowing model updates without rebuilding the inference container.

---

# Data Augmentation

The training pipeline uses Albumentations to improve robustness against CAPTCHA distortions.

Augmentations include:

- affine transformations
- optical distortion
- elastic transforms
- Gaussian blur
- Gaussian noise
- coarse dropout
- brightness and contrast adjustments
- hue and saturation modifications

These augmentations improve generalization across varying CAPTCHA styles and noise patterns.

---

# Training Configuration

| Parameter | Value |
|------------|-------|
| Base Model | `microsoft/trocr-small-printed` |
| Train Batch Size | 16 |
| Evaluation Batch Size | 16 |
| Gradient Accumulation | 2 |
| Learning Rate | 1.5e-5 |
| Scheduler | Cosine |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |
| Beam Search | 4 |
| Maximum Generation Length | 10 |
| Gradient Checkpointing | Enabled |

---

# Inference

Run inference on a single image:

```bash
python scripts/run_inference.py image.png
```

Example:

```bash
python scripts/run_inference.py samples/captcha.png
```

Example output:

```text
AB12C
```

The inference pipeline:

```text
Input Image
      │
      ▼
PIL Image
      │
      ▼
TrOCR Processor
      │
      ▼
VisionEncoderDecoderModel.generate()
      │
      ▼
Tokenizer Decode
      │
      ▼
Predicted CAPTCHA
```

---

# Batch Inference

Batch prediction is supported for evaluating multiple images.

Example:

```python
from src.inference.batch_predict import batch_predict

results = batch_predict("datasets/test/dataset_1")
```

This utility processes an entire directory and returns predictions for every image.

---

# FastAPI Inference API

The project includes a production-ready FastAPI application for serving predictions.

Run locally:

```bash
uvicorn src.api.app:app --reload
```

Production Docker deployment uses:

```bash
uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 7860
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Production:

```text
https://<your-space>.hf.space/docs
```

---

# Application Startup

The FastAPI application uses the lifespan API to initialize the inference service.

Startup sequence:

```text
Application Startup
        │
        ▼
Validate Environment Variables
        │
        ▼
Authenticate with Hugging Face
        │
        ▼
Download Production Model
        │
        ▼
Load TrOCR Processor
        │
        ▼
Load VisionEncoderDecoderModel
        │
        ▼
Store Objects in app.state
        │
        ▼
Serve Requests
```

This approach ensures that the production model is loaded once during startup rather than on every request.

---

# API Endpoints

## Health Check

```http
GET /health
```

Returns the service health status.

---

## Model Information

```http
GET /model
```

Returns information about the currently loaded production model.

---

## Predict CAPTCHA

```http
POST /predict
```

Upload a CAPTCHA image using multipart/form-data and receive the predicted text.

Example response:

```json
{
    "prediction": "AB12C"
}
```

---

# Production Deployment

The deployed inference service follows the architecture below.

```text
Client
   │
   ▼
FastAPI API
   │
   ▼
Hugging Face Authentication
   │
   ▼
Download Champion Model
   │
   ▼
Load TrOCR
   │
   ▼
Prediction
   │
   ▼
JSON Response
```

The Docker image contains only:

- application source code
- dependencies

The production model is **not** bundled into the image.

Instead, the application downloads the latest Champion model directly from the Hugging Face Model Repository during startup.

Benefits include:

- smaller Docker images
- independent model updates
- version-controlled production models
- cleaner deployment workflow

---

# Hugging Face Deployment

The project separates model storage from application deployment.

## Hugging Face Model Repository

Stores the production model artifacts:

- model weights
- tokenizer
- processor
- configuration
- generation configuration

The Champion model is uploaded after evaluation using:

```bash
python scripts/push_to_hf.py
```

---

## Hugging Face Space

Hosts the Dockerized FastAPI application.

During startup the application:

1. authenticates with Hugging Face
2. downloads the production model
3. loads the processor
4. loads the model
5. begins serving predictions

This architecture allows model updates without rebuilding the deployment container.

---

# Environment Variables

## Local Development

```env
HF_MODEL_ID=your-username/captcha-ocr
HF_TOKEN=your_huggingface_token
HF_MODEL_REVISION=main
```

---

## Deployment

Additional deployment secret:

```env
HF_SPACE_ID=your-username/captcha-api
```

---

# Docker

Build locally:

```bash
docker build -t captcha-ocr-api .
```

Run:

```bash
docker run -p 7860:7860 captcha-ocr-api
```

The Docker image is based on Python 3.11 Slim and is configured for Hugging Face Spaces Docker deployments.

---

# Docker Compose

A local monitoring stack is included.

```text
Docker Compose
      │
      ├── OCR API
      ├── Prometheus
      └── Grafana
```

This enables local observability during development.

---

# Monitoring

The FastAPI application exposes Prometheus metrics.

Collected metrics include:

- request count
- request latency
- HTTP metrics

Metrics endpoint:

```http
GET /metrics
```

Prometheus scrapes the API while Grafana can be configured to visualize collected metrics.

---

# Continuous Integration

GitHub Actions automatically validates every push.

Pipeline:

```text
Push
   │
   ▼
Checkout Repository
   │
   ▼
Install Dependencies
   │
   ▼
Compile Source
   │
   ▼
Run Tests
   │
   ▼
Docker Build
```

The CI workflow includes:

- Python environment setup
- dependency installation
- source compilation
- automated tests
- Docker build verification

Model loading inside tests is mocked, allowing the CI pipeline to execute without downloading production models from Hugging Face.

---

# Continuous Deployment

Deployment is automated using GitHub Actions.

Workflow:

```text
GitHub Push
      │
      ▼
Build Docker Image
      │
      ▼
Synchronize Repository
      │
      ▼
Hugging Face Space
      │
      ▼
Automatic Rebuild
      │
      ▼
Production API
```

The deployment workflow synchronizes the repository directly to the Hugging Face Space, triggering a rebuild of the production application.

---

# Testing

Automated tests cover the primary project components.

Current test suite includes:

```text
tests/
├── test_api.py
├── test_inference.py
└── test_training_smoke.py
```

Coverage includes:

- API functionality
- inference pipeline
- training smoke tests

To execute locally:

```bash
pytest tests
```

---

# Hardware Support

Recommended:

- NVIDIA GPU with CUDA for training

Supported:

- CPU inference
- CPU training
- Automatic device detection

---

# Project Scripts

| Script | Purpose |
|----------|---------|
| `bootstrap_mlflow.py` | Initialize a new MLflow registry |
| `run_clean_split.py` | Clean and split datasets |
| `run_training.py` | Train and register Candidate model |
| `run_evaluation.py` | Evaluate and promote models |
| `push_to_hf.py` | Upload Champion model to Hugging Face |
| `run_inference.py` | Single-image inference |

---

# MLOps Principles

This project follows production-inspired MLOps practices including:

- reproducible training
- modular architecture
- experiment tracking
- versioned model registry
- automated evaluation
- Champion/Candidate lifecycle
- production model promotion
- artifact versioning
- Dockerized deployment
- CI/CD automation
- API monitoring
- production-ready inference

The architecture is designed to be easily extended with:

- Kubernetes deployments
- scheduled retraining
- automated rollback
- cloud monitoring
- GPU inference
- deployment smoke tests
- production authentication and rate limiting

---

# License

This repository fine-tunes Microsoft's TrOCR model.

Base model:

https://huggingface.co/microsoft/trocr-small-printed

Please refer to the original model repository for licensing information.