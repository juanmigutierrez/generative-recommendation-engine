"""
Step 7: train the ranking-stage model.

Pipeline:
  1. Fit ALS on train (the "retrieval" stage feeding the ranker -- see docs/07_ranking.md
     for why ALS rather than the Step 6 Transformer's beam search: ALS scores the whole
     catalog for thousands of users in under a second via one batched matmul, which is
     what "cheap, wide retrieval" is supposed to be; beam search is a forward pass per
     generation step, deliberately evaluated on a 2,000-user sample in Step 6 because it
     doesn't scale to ranker-training volumes in this sandbox).
  2. Sample a set of warm (non-cold) train users who also have val interactions, and pull
     ALS's top-100 candidates for each.
  3. Label candidates: 1 if the item is in that user's val_targets, 0 otherwise. Users
     whose val items are *not* among ALS's top-100 contribute no positives and are dropped
     from the training set (a LambdaRank group with no positive has zero gradient anyway).

     Why no "positive injection" any more (docs/09_audit_sources_and_results.md, bug #1):
     the first version appended the missing true items to each candidate list with a
     placeholder ALS score (the list minimum) -- and since 93% of all positives ended up
     being injected, the ranker learned "the lowest-ALS-score, never-seen item is the
     positive", the exact opposite of a useful reranking signal on real candidates.
     A reranker can only reorder what retrieval gives it, so it must be trained on
     retrieved candidates with their real feature values.
  4. Train a LightGBM LambdaMART ranker (objective="lambdarank"), grouped by user, scored
     by NDCG during training -- the standard learning-to-rank setup used in production
     recommender ranking stages.

Run:
    python backend/scripts/train_ranker.py
"""
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.als import ALSRecommender
from models.ranking_features import (
    FEATURE_COLUMNS,
    build_item_features,
    build_user_profile_embeddings,
    build_user_interaction_counts,
    assemble_features,
)

PROCESSED_DIR = os.path.join("data", "processed")
N_CANDIDATES = 100
N_TRAIN_USERS = 18000  # of ~26.8K eligible; only ~13% of them end up with a retrieved positive (see below)
SEED = 42


def load_targets(path: str) -> dict:
    df = pd.read_parquet(path)
    return {row.user_id: set(row.item_ids) for row in df.itertuples()}


def main():
    t_start = time.time()
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    user_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet"))
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    train_sequences = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_sequences.parquet"))
    val_targets = load_targets(os.path.join(PROCESSED_DIR, "val_targets.parquet"))
    item_embeddings = np.load(os.path.join(PROCESSED_DIR, "item_embeddings.npy"))

    n_users, n_items = len(user_catalog), len(item_catalog)
    print(f"train: {len(train):,} rows, {n_users:,} users, {n_items:,} items")

    print("fitting ALS...")
    als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(train, n_users, n_items)

    print("building item + user features...")
    item_features = build_item_features(train, n_items)
    user_profiles = build_user_profile_embeddings(train_sequences, item_embeddings)
    user_n_interactions = build_user_interaction_counts(train_sequences)

    # sample warm users who also show up in val (so we have real positives to train against)
    warm_users = set(train_sequences["user_id"])
    eligible = sorted(warm_users & set(val_targets.keys()))
    rng = np.random.RandomState(SEED)
    sample_users = rng.choice(eligible, size=min(N_TRAIN_USERS, len(eligible)), replace=False)
    sample_users = np.sort(sample_users)
    sampled_users_all = sample_users.copy()  # saved below so eval scripts can hold these users out
    print(f"eligible warm+val users: {len(eligible):,}, sampled: {len(sample_users):,}")

    print("generating ALS candidates (batched)...")
    t0 = time.time()
    user_items_batch = als.user_items[sample_users]
    cand_ids, cand_scores = als.model.recommend(
        sample_users, user_items_batch, N=N_CANDIDATES, filter_already_liked_items=True
    )
    print(f"  candidate generation: {time.time() - t0:.1f}s, shape {cand_ids.shape}")

    item_ids_per_user = [list(row) for row in cand_ids]
    als_scores_per_user = [list(row) for row in cand_scores]

    # keep only users for whom retrieval actually surfaced at least one true item -- no
    # injection of un-retrieved positives with placeholder features (see docstring)
    keep = [i for i, u in enumerate(sample_users) if val_targets.get(u, set()) & set(item_ids_per_user[i])]
    print(f"  users with >=1 true val item in ALS top-{N_CANDIDATES}: {len(keep):,} / {len(sample_users):,} "
          f"(retrieval recall@{N_CANDIDATES} is the ceiling the ranker works under)")
    sample_users = sample_users[keep]
    item_ids_per_user = [item_ids_per_user[i] for i in keep]
    als_scores_per_user = [als_scores_per_user[i] for i in keep]
    n_injected = 0

    print("assembling feature table...")
    t0 = time.time()
    feat_df = assemble_features(
        sample_users, item_ids_per_user, als_scores_per_user,
        item_features, user_profiles, user_n_interactions, item_embeddings,
    )
    print(f"  {len(feat_df):,} rows in {time.time() - t0:.1f}s")

    pos_pairs = {
        (u, it) for u in sample_users for it in val_targets.get(u, set())
    }
    feat_df["label"] = [
        1 if (u, it) in pos_pairs else 0
        for u, it in zip(feat_df["user_id"].values, feat_df["item_id"].values)
    ]
    n_pos = feat_df["label"].sum()
    print(f"  positives: {n_pos:,} / {len(feat_df):,} ({n_pos / len(feat_df):.4%})")

    # LightGBM's lambdarank objective needs rows grouped contiguously by user, sorted so
    # that group sizes line up with the groups array below.
    feat_df = feat_df.sort_values("user_id").reset_index(drop=True)
    groups = feat_df.groupby("user_id").size().values

    X = feat_df[FEATURE_COLUMNS].values
    y = feat_df["label"].values

    print("training LightGBM LambdaMART ranker...")
    t0 = time.time()
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=SEED,
        verbosity=-1,
    )
    ranker.fit(X, y, group=groups, eval_at=[10])
    print(f"  trained in {time.time() - t0:.1f}s")

    importances = dict(zip(FEATURE_COLUMNS, ranker.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    print("feature importances (gain-based split count):")
    for k, v in importances.items():
        print(f"  {k:<26} {v}")

    out_dir = PROCESSED_DIR
    with open(os.path.join(out_dir, "ranker_model.pkl"), "wb") as f:
        pickle.dump(ranker, f)
    # the evaluation scripts exclude these users from their val samples -- their val items
    # are this model's training labels, so scoring them would be a leak (audit bug #2)
    np.save(os.path.join(out_dir, "ranker_train_users.npy"), sampled_users_all)

    meta = {
        "n_sampled_users": int(len(sampled_users_all)),
        "n_train_users": int(len(sample_users)),  # sampled users that had >=1 retrieved positive
        "n_candidates": N_CANDIDATES,
        "n_rows": int(len(feat_df)),
        "n_positives": int(n_pos),
        "n_injected": int(n_injected),  # always 0 now, kept for comparison with the old meta
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": importances,
        "total_time_s": time.time() - t_start,
    }
    with open(os.path.join(out_dir, "ranker_train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved -> {out_dir}/ranker_model.pkl, ranker_train_meta.json")
    print(f"total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
