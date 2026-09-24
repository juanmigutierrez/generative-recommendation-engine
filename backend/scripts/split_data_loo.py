"""
Step 10a: leave-one-out (LOO) split -- the evaluation protocol used by TIGER (Rajput et al.
2023), SASRec (Kang & McAuley 2018) and most sequential-recommendation papers, so that this
project's numbers can be compared with published ones.

Per user, interactions sorted by time:
    [ i_1, i_2, ..., i_{n-2} ]  -> LOO train history
    i_{n-1}                     -> val target   (context = i_1 .. i_{n-2})
    i_n                         -> test target  (context = i_1 .. i_{n-1}, i.e. val item included)

This is a *different task* from the global time split in split_data.py (see
docs/02_split_and_eval.md and docs/09_audit_sources_and_results.md): the context ends
immediately before the target and there are essentially no cold items, which is why paper
numbers (Recall@10 ~ 0.05-0.07 on Amazon 5-core sets) are an order of magnitude above the
time-split table. Both protocols are kept; this one is the paper-comparable one.

Repeat interactions with the same item are collapsed to the first occurrence, as in the
papers (a user's sequence is a sequence of distinct items).

Outputs (data/processed/):
    loo_train.parquet             flat rows (user_id, item_id, timestamp) for popularity / ALS
    loo_train_sequences.parquet   user_id, item_ids (chronological, all but the last 2)
    loo_val_sequences.parquet     user_id, item_ids (all but the last 1) -- test-time context
    loo_val_targets.parquet       user_id, item_ids = [second-to-last item]
    loo_test_targets.parquet      user_id, item_ids = [last item]
    loo_split_stats.json

Run:
    python backend/scripts/split_data_loo.py
"""
import json
import os

import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")
MIN_ITEMS = 3  # need >=1 train item + val + test; 5-core guarantees >=5 anyway


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    df = df.drop_duplicates(["user_id", "item_id"], keep="first")  # distinct items per user
    print(f"{len(df):,} distinct (user, item) interactions, {df.user_id.nunique():,} users")

    seqs = df.groupby("user_id").agg(item_ids=("item_id", list), timestamps=("timestamp", list)).reset_index()
    seqs = seqs[seqs["item_ids"].apply(len) >= MIN_ITEMS].reset_index(drop=True)
    print(f"{len(seqs):,} users with >= {MIN_ITEMS} distinct items")

    train_items = seqs["item_ids"].apply(lambda s: s[:-2])
    val_items = seqs["item_ids"].apply(lambda s: s[-2])
    test_items = seqs["item_ids"].apply(lambda s: s[-1])
    train_ts = seqs["timestamps"].apply(lambda s: s[:-2])

    train_seq = pd.DataFrame({"user_id": seqs["user_id"], "item_ids": train_items})
    val_seq = pd.DataFrame({"user_id": seqs["user_id"], "item_ids": seqs["item_ids"].apply(lambda s: s[:-1])})
    val_targets = pd.DataFrame({"user_id": seqs["user_id"], "item_ids": val_items.apply(lambda i: [i])})
    test_targets = pd.DataFrame({"user_id": seqs["user_id"], "item_ids": test_items.apply(lambda i: [i])})

    flat = pd.DataFrame({
        "user_id": np.repeat(seqs["user_id"].values, train_items.apply(len).values),
        "item_id": np.concatenate(train_items.values),
        "timestamp": np.concatenate(train_ts.values),
    })

    train_seq.to_parquet(os.path.join(PROCESSED_DIR, "loo_train_sequences.parquet"))
    val_seq.to_parquet(os.path.join(PROCESSED_DIR, "loo_val_sequences.parquet"))
    val_targets.to_parquet(os.path.join(PROCESSED_DIR, "loo_val_targets.parquet"))
    test_targets.to_parquet(os.path.join(PROCESSED_DIR, "loo_test_targets.parquet"))
    flat.to_parquet(os.path.join(PROCESSED_DIR, "loo_train.parquet"))

    seen_train = set(flat["item_id"])
    lens = train_items.apply(len)
    stats = {
        "n_users": int(len(seqs)),
        "n_train_rows": int(len(flat)),
        "train_len_median": float(lens.median()),
        "train_len_mean": float(lens.mean()),
        "train_len_p95": float(lens.quantile(0.95)),
        "val_targets_cold_pct": float(100 * (~val_items.isin(seen_train)).mean()),
        "test_targets_cold_pct": float(100 * (~test_items.isin(seen_train | set(val_items))).mean()),
    }
    with open(os.path.join(PROCESSED_DIR, "loo_split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"saved -> {PROCESSED_DIR}/loo_*.parquet")


if __name__ == "__main__":
    main()
