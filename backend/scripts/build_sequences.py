"""
Step 3 of the pipeline: turn the flat train interactions into per-user chronological
sequences, and turn val/test into per-user "ground truth" target sets.

This is the format every downstream model consumes:
  - baselines (popularity, ALS) just need the flat interaction list (already have it)
  - the sequential retrieval model (Step 5+) needs, per user, an ordered list of item_ids
    -- that's what we build here
  - evaluation (Step 8) needs, per user, "what did they actually interact with in
    val/test" to compare recommendations against

Output:
    data/processed/train_sequences.parquet   user_id, item_ids (list, chronological), invoice_ids (list)
    data/processed/val_targets.parquet       user_id, item_ids (list) -- ground truth for val
    data/processed/test_targets.parquet      user_id, item_ids (list) -- ground truth for test

Run:
    python backend/scripts/build_sequences.py
"""
import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")


def build_user_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"])
    grouped = df.groupby("user_id").agg(
        item_ids=("item_id", list),
        invoice_ids=("invoice_id", list),
        timestamps=("timestamp", list),
    ).reset_index()
    return grouped


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    # de-duplicate: for evaluation we only care about the *set* of items a user
    # interacted with, not how many times, since baskets can repeat an item.
    grouped = df.groupby("user_id")["item_id"].agg(lambda s: sorted(set(s))).reset_index()
    grouped = grouped.rename(columns={"item_id": "item_ids"})
    return grouped


def main():
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    train_seq = build_user_sequences(train)
    val_targets = build_targets(val)
    test_targets = build_targets(test)

    train_seq.to_parquet(os.path.join(PROCESSED_DIR, "train_sequences.parquet"))
    val_targets.to_parquet(os.path.join(PROCESSED_DIR, "val_targets.parquet"))
    test_targets.to_parquet(os.path.join(PROCESSED_DIR, "test_targets.parquet"))

    seq_lens = train_seq["item_ids"].apply(len)
    print(f"train sequences: {len(train_seq):,} users")
    print(f"  seq length -- median: {seq_lens.median():.0f}, mean: {seq_lens.mean():.1f}, "
          f"max: {seq_lens.max()}")
    print(f"val targets: {len(val_targets):,} users")
    print(f"test targets: {len(test_targets):,} users")


if __name__ == "__main__":
    main()
