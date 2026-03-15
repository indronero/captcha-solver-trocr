# TrOCR CAPTCHA Solver (MLOps)

This project implements a **CAPTCHA recognition system using Microsoft's TrOCR model** with a full **training, inference, and evaluation pipeline** structured for MLOps workflows.

The system fine-tunes:

microsoft/trocr-small-printed

on multiple CAPTCHA datasets using strong augmentations and evaluates performance using **Character Error Rate (CER)**.

The repository supports:

- multi-dataset training
- heavy augmentation for CAPTCHA robustness
- automatic model versioning
- automatic loading of the latest trained model
- manual version selection for inference
- batch prediction
- sorting predictions into **correct** and **incorrect**

This structure allows the model to be easily integrated into **DevOps pipelines and production systems**.

---

# Model

Base model:

```
microsoft/trocr-small-printed
```

Fine-tuned for CAPTCHA decoding.

---

# Project Structure

```
captcha_solver_mlops/

model/
   trocr_finetuned_captcha_model/
      v1/
      v2/
      v3/

samples/
   captchas/

datasets/
   

cache/

src/

   config.py

   utils/
      model_versioning.py

   inference/
      model_loader.py
      predictor.py
      batch_predict.py

   evaluation/
      sort_predictions.py

   training/
      dataset_loader.py
      preprocess.py
      trainer.py
      train_pipeline.py

scripts/

   run_training.py
   run_inference.py
   run_sorting.py

requirements.txt
README.md
```

---

# Dataset Format

CAPTCHA labels are derived from **image filenames**.

Example:

```
A9K2L.png
```

Label extracted:

```
A9K2L
```

Example dataset:

```
dataset/
   A9K2L.png
   X2P8T.png
   T7Q9D.png
```

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training

Run training:

```
python scripts/run_training.py
```

Training performs:

- dataset loading from multiple sources
- dataset balancing
- augmentation using Albumentations
- preprocessing with TrOCR processor
- model fine-tuning
- CER evaluation
- model checkpointing

---

# Model Versioning

Each training run automatically creates a **new version** of the model.

Example:

```
model/
   trocr_finetuned_captcha_model/
      v1/
      v2/
      v3/
```

Version numbers increment automatically.

Example:

```
First training → v1
Second training → v2
Third training → v3
```

Each version contains:

```
config.json
pytorch_model.bin
tokenizer.json
preprocessor_config.json
special_tokens_map.json
```

---

# Inference

Run prediction on a single image:

```
python scripts/run_inference.py image.png
```

Example:

```
python scripts/run_inference.py samples/captchas/AB12C.png
```

Output:

```
AB12C
```

---

# Model Version Selection

If no version is specified, the system automatically loads the **latest model version**.

Example:

```
python scripts/run_inference.py captcha.png
```

Loads:

```
model/trocr_finetuned_captcha_model/v3
```

To specify a version:

```
python scripts/run_inference.py captcha.png v1
```

Loads:

```
model/trocr_finetuned_captcha_model/v1
```

---

# Batch Prediction

Batch prediction is handled by:

```
src/inference/batch_predict.py
```

It processes all images inside a folder.

---

# Evaluation: Correct vs Incorrect Sorting

To evaluate predictions and automatically sort results:

```
python scripts/run_sorting.py input_folder output_folder
```

Example:

```
python scripts/run_sorting.py samples/captchas output
```

Output structure:

```
output/

   correct/
      ABC12.png
      KJ2P8.png

   incorrect/
      A8C12.png
      KJP28.png
```

Accuracy will also be printed:

```
{'correct': 84, 'incorrect': 16, 'accuracy': 0.84}
```

---

# Training Configuration

| Parameter | Value |
|--------|------|
| Base Model | microsoft/trocr-small-printed |
| Epochs | 10 |
| Train Batch Size | 16 |
| Gradient Accumulation | 2 |
| Scheduler | Cosine |
| Learning Rate | 1.5e-5 |
| Beam Search | 4 |
| Max Tokens | 10 |

---

# Data Augmentation

Training images use Albumentations with:

- Affine transformations
- Optical distortion
- Grid distortion
- Elastic transforms
- Gaussian noise
- Blur
- Coarse dropout
- Brightness and contrast changes
- Hue / saturation shifts

These augmentations improve robustness against CAPTCHA distortions.

---

# Evaluation Metric

Model performance is measured using:

**CER (Character Error Rate)**

Lower CER indicates better performance.

---

# Hardware

Recommended:

GPU with CUDA support.

The pipeline automatically falls back to CPU if CUDA is unavailable.

---

# Scripts

| Script | Purpose |
|------|------|
| run_training.py | Train a new model version |
| run_inference.py | Predict CAPTCHA text |
| run_sorting.py | Evaluate and sort predictions |

---

# MLOps Design

This project supports typical MLOps workflows:

- versioned models
- reproducible training
- automated evaluation
- modular inference pipeline
- easy integration with CI/CD

DevOps can add:

- Docker
- CI/CD pipelines
- REST APIs
- GPU deployments

without modifying the training or inference logic.

---

# License

This project uses the TrOCR model from Microsoft.

Original model:

https://huggingface.co/microsoft/trocr-small-printed