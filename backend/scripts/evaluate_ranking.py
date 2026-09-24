"""
Step 7: evaluate the ranking stage the fair way -- same exact candidate set, reordered
or not. This isolates what reranking alone contributes: recall@100 is fixed the moment
ALS generates the candidates (reranking can't add items that aren't already in the list),
so the honest question is whether reordering moves true positives up into the top 10/20.

Two samples, both held out from ranker training:
  - val (held-out): eligible val users NOT used to train the ranker (train_ranker.py used
    18,000 of ~26,838 eligible warm+val users -- evaluate on a fresh sample of the rest).
  - test: warm users with test targets, entirely untouched during ranker training --
    this is the number that matters most.

Run:
    python backend/scripts/evaluate_ranking.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.als import ALSRecommender
from models.metrics import evaluate
from models.ranking_features import (
    FEATURE_COLUMNS,
    build_item_features,
    build_user_profile_embeddings,
    build_user_interaction_counts,
    assemble_features,
)

PROCESSED_DIR = os.path.join("data", "processed")
N_CANDIDATES = 100
N_EVAL_USERS = 5000
K_VALUES = (10, 20)
SEED = 123


def load_targets(path: str) -> dict:
    df = pd.read_parquet(path)
    return {row.user_id: set(row.item_ids) for row in df.itertuples()}


def build_eval_set(user_ids, als, item_features, user_profiles, user_n_interactions, item_embeddings):
    user_items_batch = als.user_items[user_ids]
    cand_ids, cand_scores = als.model.recommend(
        user_ids, user_items_batch, N=N_CANDIDATES, filter_already_liked_items=True
    )
    item_ids_per_user = [list(row) for row in cand_ids]
    als_scores_per_user = [list(row) for row in cand_scores]
    feat_df = assemble_features(
        user_ids, item_ids_per_user, als_scores_per_user,
        item_features, user_profiles, user_n_interactions, item_embeddings,
    )
    return feat_df


def main():
    t_start = time.time()
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    user_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet"))
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    train_sequences = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_sequences.parquet"))
    val_targets = load_targets(os.path.join(PROCESSED_DIR, "val_targets.parquet"))
    test_targets = load_targets(os.path.join(PROCESSED_DIR, "test_targets.parquet"))
    item_embeddings = np.load(os.path.join(PROCESSED_DIR, "item_embeddings.npy"))

    with open(os.path.join(PROCESSED_DIR, "ranker_model.pkl"), "rb") as f:
        ranker = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "ranker_train_meta.json")) as f:
        train_meta = json.load(f)

    n_users, n_items = len(user_catalog), len(item_catalog)
    print("refitting ALS (same config as train_ranker.py)...")
    als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(train, n_users, n_items)

    item_features = build_item_features(train, n_items)
    user_profiles = build_user_profile_embeddings(train_sequences, item_embeddings)
    user_n_interactions = build_user_interaction_counts(train_sequences)

    warm_users = set(train_sequences["user_id"])
    rng = np.random.RandomState(SEED)

    results = {}
    sample_sizes = {}

    # ---- val: held out from the 12,000 users used to train the ranker ----
    eligible_val = sorted(warm_users & set(val_targets.keys()))
    train_sample_size = train_meta.get("n_sampled_users", train_meta["n_train_users"])
    # reproduce the exact same training sample (same seed/logic as train_ranker.py) so we
    # can exclude it cleanly
    rng_train = np.random.RandomState(42)
    rt_path = os.path.join(PROCESSED_DIR, "ranker_train_users.npy")
    if os.path.exists(rt_path):  # exact list saved by train_ranker.py (preferred)
        train_sample = set(np.load(rt_path).tolist())
    else:  # legacy: reproduce train_ranker.py's sampling
        train_sample = set(rng_train.choice(eligible_val, size=min(train_sample_size, len(eligible_val)), replace=False).tolist())
    held_out_val = sorted(set(eligible_val) - train_sample)
    val_sample = np.sort(rng.choice(held_out_val, size=min(N_EVAL_USERS, len(held_out_val)), replace=False))
    print(f"val: {len(eligible_val):,} eligible, {len(held_out_val):,} held out from ranker training, evaluating on {len(val_sample):,}")

    # ---- test: warm users with test targets, never touched during training ----
    eligible_test = sorted(warm_users & set(test_targets.keys()))
    test_sample = np.sort(rng.choice(eligible_test, size=min(N_EVAL_USERS, len(eligible_test)), replace=False))
    print(f"test: {len(eligible_test):,} eligible, evaluating on {len(test_sample):,}")

    for split_name, user_sample, targets_full in [("val", val_sample, val_targets), ("test", test_sample, test_targets)]:
        # IMPORTANT: restrict targets to the sampled users. evaluate() averages over every
        # key in `targets`, so passing the full val/test targets dict here (tens of
        # thousands of users) while candidates only exist for the 5,000 sampled users would
        # silently score every unsampled user as a 0 -- caught this via a direct comparison
        # against Step 4's baseline numbers, which this pipeline should reproduce exactly
        # for the "als_unranked" case since it's the same model, same top-k, same users.
        targets = {u: targets_full[u] for u in user_sample}

        print(f"\nbuilding candidates + features for {split_name}...")
        t0 = time.time()
        feat_df = build_eval_set(user_sample, als, item_features, user_profiles, user_n_interactions, item_embeddings)
        print(f"  {len(feat_df):,} rows in {time.time() - t0:.1f}s")

        feat_df["rerank_score"] = ranker.predict(feat_df[FEATURE_COLUMNS].values)

        # group candidates per user, sorted by als_score (unranked / retrieval order) and
        # by rerank_score (the ranking stage's output), same candidate pool both times
        unranked_lists = {}
        reranked_lists = {}
        for user_id, grp in feat_df.groupby("user_id"):
            unranked_lists[user_id] = grp.sort_values("als_score", ascending=False)["item_id"].tolist()
            reranked_lists[user_id] = grp.sort_values("rerank_score", ascending=False)["item_id"].tolist()

        def make_recommend_fn(lists):
            def fn(user_id, k):
                return lists.get(user_id, [])[:k]
            return fn

        results.setdefault(split_name, {})
        results[split_name]["als_unranked"] = evaluate(make_recommend_fn(unranked_lists), targets, K_VALUES)
        results[split_name]["als_reranked"] = evaluate(make_recommend_fn(reranked_lists), targets, K_VALUES)
        sample_sizes[split_name] = len(user_sample)

    print("\n" + "=" * 70)
    print(f"{'split':<6}{'model':<16}" + "".join(f"{m:<12}" for m in results["val"]["als_unranked"]))
    print("-" * 70)
    for split_name in results:
        for model_name, metrics in results[split_name].items():
            row = f"{split_name:<6}{model_name:<16}"
            row += "".join(f"{v:<12.4f}" for v in metrics.values())
            print(row)
    print("=" * 70)

    out = {
        "results": results,
        "n_candidates": N_CANDIDATES,
        "eval_sample_size": sample_sizes,
        "total_time_s": time.time() - t_start,
    }
    with open(os.path.join(PROCESSED_DIR, "ranking_eval_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {PROCESSED_DIR}/ranking_eval_results.json")
    print(f"total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
