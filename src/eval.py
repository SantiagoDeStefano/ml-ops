import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score


MODEL_DIR = "models/model"
TEXT_COL = "review"
LABEL_COL = "sentiment"
MAX_LEN = 256


def main():
    test_df = pd.read_csv("data/processed/test.csv")

    label2id = {"negative": 0, "positive": 1}
    test_df["label"] = test_df[LABEL_COL].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    test_ds = Dataset.from_pandas(test_df[[TEXT_COL, "label"]])
    test_ds = test_ds.map(tokenize, batched=True)
    test_ds = test_ds.remove_columns([TEXT_COL])
    test_ds.set_format("torch")

    trainer = Trainer(model=model)

    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "test_samples": int(len(test_df)),
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
