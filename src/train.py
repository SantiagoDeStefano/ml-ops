import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.transformers

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
OUT_MODEL_DIR = "models/model"
MLFLOW_TRACKING_URI="http://localhost:5000"

TEXT_COL = "review"
LABEL_COL = "sentiment"
SEED = 42
MAX_LEN = 256

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}

    train_df["label"] = train_df[LABEL_COL].map(label2id)
    val_df["label"] = val_df[LABEL_COL].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    train_ds = Dataset.from_pandas(train_df[[TEXT_COL, "label"]])
    val_ds = Dataset.from_pandas(val_df[[TEXT_COL, "label"]])

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds = train_ds.remove_columns([TEXT_COL])
    val_ds = val_ds.remove_columns([TEXT_COL])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
    )

    args = TrainingArguments(
        output_dir="./outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=3e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("review-sentiment")

    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("epochs", 2)

        trainer.train()
        metrics = trainer.evaluate()

        # save final model locally for DVC pipeline output
        os.makedirs(OUT_MODEL_DIR, exist_ok=True)
        trainer.save_model(OUT_MODEL_DIR)
        tokenizer.save_pretrained(OUT_MODEL_DIR)

        # log metrics
        mlflow.log_metrics(metrics)

        # log to MLflow model artifact
        mlflow.transformers.log_model( # type: ignore
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",
            pip_requirements=[
                "mlflow",
                "torch",
                "datasets",
                "transformers",
                "huggingface_hub"
            ],
            registered_model_name="review-sentiment-transformer"
        )

        print("Training done. Saved model to:", OUT_MODEL_DIR)
        print("Eval metrics:", metrics)

if __name__ == "__main__":
    main()
