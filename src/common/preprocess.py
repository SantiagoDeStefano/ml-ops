import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = "data/raw/film_sentiment.csv"
OUT_DIR = "data/processed"

TEXT_COL = "review"
LABEL_COL = "sentiment"

SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # basic cleaning
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df = df.dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    # keep only valid labels
    df = df[df[LABEL_COL].isin(["positive", "negative"])]

    # remove empty texts
    df = df[df[TEXT_COL].str.len() > 0]

    # remove duplicates
    df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL])

    # split: train 80%, val 10%, test 10%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df[LABEL_COL]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df[LABEL_COL]
    )

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Saved:")
    print(f"- {OUT_DIR}/train.csv: {len(train_df)} rows")
    print(f"- {OUT_DIR}/val.csv:   {len(val_df)} rows")
    print(f"- {OUT_DIR}/test.csv:  {len(test_df)} rows")


if __name__ == "__main__":
    main()
