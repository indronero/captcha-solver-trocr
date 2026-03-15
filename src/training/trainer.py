import torch
import numpy as np
from evaluate import load
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from src.training.preprocess import processor

cer_metric = load("cer")


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {"cer": cer}



def data_collator(features):

    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def build_trainer(model, train_dataset, val_dataset, output_dir):

    training_args = Seq2SeqTrainingArguments(

        output_dir=output_dir,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        gradient_accumulation_steps=2,

        num_train_epochs=10,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=200,

        learning_rate=1.5e-5,

        warmup_steps=500,

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        save_total_limit=2,

        predict_with_generate=True,

        fp16=torch.cuda.is_available(),

        gradient_checkpointing=True,

        generation_max_length=10,
        generation_num_beams=4,

        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,

        report_to="none",

        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(

        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        tokenizer=processor.tokenizer,

        compute_metrics=compute_metrics,

        # CRITICAL (same as Kaggle)
        data_collator=data_collator
    )

    return trainer