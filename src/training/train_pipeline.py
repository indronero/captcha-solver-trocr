import torch
from transformers import VisionEncoderDecoderModel
from src.config import MODEL_NAME, DATASET_PATHS, NUM_EXAMPLES, MODEL_BASE_DIR
from src.training.dataset_loader import load_datasets
from src.training.preprocess import preprocess_function, processor
from src.training.trainer import build_trainer
from src.utils.model_versioning import get_next_version, get_latest_version
import os



def train():

    if os.path.exists(MODEL_BASE_DIR) and os.listdir(MODEL_BASE_DIR):

        try:
            latest_model = get_latest_version(MODEL_BASE_DIR)
            print("Continuing training from:", latest_model)

            model = VisionEncoderDecoderModel.from_pretrained(latest_model)

        except:
            print("No valid version found. Loading base model.")
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    else:

        print("No trained models found. Loading base model.")

        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = processor.tokenizer.vocab_size
    
    if model.config.decoder.vocab_size != processor.tokenizer.vocab_size:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    print("Loading datasets...")

    train_dataset, val_dataset = load_datasets(DATASET_PATHS, NUM_EXAMPLES)

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

    print("Building trainer...")

    version_path = get_next_version(MODEL_BASE_DIR)

    print("Saving model to:", version_path)

    trainer = build_trainer(
        model,
        train_dataset,
        val_dataset,
        version_path
    )

    print("Starting training...")

    trainer.train()

    trainer.save_model(version_path)

    processor.save_pretrained(version_path)

    print("Training complete. Model saved to:", MODEL_BASE_DIR)