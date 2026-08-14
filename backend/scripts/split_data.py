"""
Step 2 of the pipeline: time-based train/val/test split.

Why time-based, not random: a random split lets the model "see the future" (train on an
interaction from November, test on one from August) which silently inflates offline metrics
and doesn't reflect how the model is actually used in production -- always predicting
forward in time from what's known so far. See docs/02_split_and_eval.md.

Cutoffs are computed as a fraction of the dataset's own observed time span (not hardcoded
dates), so this script works unchanged regardless of which dataset produced
data/processed/interactions.parquet -- UCI Online Retail spans ~1 year, Amazon Reviews
categories can span decades. A fixed VAL_FRACTION/TEST_FRACTION of the *time axis* keeps
the "hold out the most recent slice of time" methodology consistent across datasets, even
though the actual cutoff dates differ.

    train : timestamp <  (max_ts - (VAL_FRACTION + TEST_FRACTION) * span)
    val   : timestamp in [train_cutoff, val_cutoff)
    test  : timestamp >= val_cutoff

Output: data/processed/{train,val,test}.parquet
        data/processed/split_stats.json  (sizes, cutoff dates, cold-start counts)

Run:
    python backend/scripts/split_data.py
"""
import json
import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

VAL_FRACTION = 0.08   # last 8% of the time span -> validation
TEST_FRACTION = 0.08  # last 8% of the time span -> test


def main():
    interactions = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))

    min_ts, max_ts = interactions["timestamp"].min(), interactions["timestamp"].max()
    span = max_ts - min_ts
    val_start = max_ts - (VAL_FRACTION + TEST_FRACTION) * span
    test_start = max_ts - TEST_FRACTION * span

    train = interactions[interactions["timestamp"] < val_start]
    val = interactions[(interactions["timestamp"] >= val_start) & (interactions["timestamp"] < test_start)]
    test = interactions[interactions["timestamp"] >= test_start]

    train_users = set(train["user_id"])
    train_items = set(train["item_id"])

    stats = {
        "min_timestamp": str(min_ts),
        "max_timestamp": str(max_ts),
        "val_start": str(val_start),
        "test_start": str(test_start),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_users": len(train_users),
        "val_users": val["user_id"].nunique(),
        "test_users": test["user_id"].nunique(),
        # "cold" = appears in val/test but was never seen in train.
        # These rows are impossible for a pure collaborative-filtering model to get right --
        # they're exactly the cases the content-based (Semantic ID) stage exists to handle.
        "val_cold_users": len(set(val["user_id"]) - train_users),
        "val_cold_items": len(set(val["item_id"]) - train_items),
        "test_cold_users": len(set(test["user_id"]) - train_users),
        "test_cold_items": len(set(test["item_id"]) - train_items),
    }

    train.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val.to_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    test.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    with open(os.path.join(PROCESSED_DIR, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
