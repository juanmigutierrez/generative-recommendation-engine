"""
Step 8: one script, every model tier, the same user sample, the same fallback rule for
cold-start users -- the single comparison table that's the actual evidence for "what did
each layer of complexity buy."

This closes a gap flagged honestly back in Step 6 and Step 7's docs: the Transformer's own
evaluation (`evaluate_retrieval.py`) and the ranker's own evaluation (`evaluate_ranking.py`)
each *excluded* cold-start users (no train history -> no beam-search context, no ALS
candidates), while the Step 4 baseline table *included* them via a popularity fallback. That
made those two tables not directly comparable to the baseline numbers. Here, every tier gets
the same popularity fallback for cold users, so the whole table is apples-to-apples:

    popularity            -- the floor
    als (+ fallback)      -- collaborative filtering
    transformer (+ fallback) -- generative retrieval over Semantic IDs (Steps 5-6)
    als + ranking (+ fallback) -- retrieval + LightGBM reranking (Step 7)

Beam search is expensive (a forward pass per generation step per user), so -- consistent
with Steps 6 and 7's own documented scope decisions -- this evaluates a fixed 2,000-user
random sample per split, drawn from *all* val/test users (not just warm ones), so the
sample's cold-start rate matches the real population instead of silently excluding it.

Run (each split separately -- the transformer tier alone takes ~65-90s per split):
    python backend/scripts/run_full_evaluation.py --split val
    python backend/scripts/run_full_evaluation.py --split test
"""
import argparse
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import transformer as tx
from models.als import ALSRecommender
from models.popularity import PopularityRecommender
from models.metrics import evaluate as evaluate_metrics
from models.ranking_features import (
    FEATURE_COLUMNS, build_item_features, build_user_profile_embeddings,
    build_user_interaction_counts, assemble_features,
)
import evaluate_retrieval as ev

PROCESSED_DIR = os.path.join("data", "processed")
K_VALUES = (10, 20)
EVAL_SAMPLE_SIZE = 2000
N_RANK_CANDIDATES = 100
SEED = 7


def load_targets(path: str) -> dict:
    df = pd.read_parquet(path)
    return {row.user_id: set(row.item_ids) for row in df.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = ap.parse_args()

    t_start = time.time()
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    user_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet"))
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    train_sequences = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_sequences.parquet"))
    item_embeddings = np.load(os.path.join(PROCESSED_DIR, "item_embeddings.npy"))
    val_targets = load_targets(os.path.join(PROCESSED_DIR, "val_targets.parquet"))
    test_targets = load_targets(os.path.join(PROCESSED_DIR, "test_targets.parquet"))
    n_users, n_items = len(user_catalog), len(item_catalog)

    print("fitting popularity + ALS...")
    pop_model = PopularityRecommender().fit(train)
    als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(train, n_users, n_items)

    print("loading transformer + trie...")
    tf_params, tf_meta, tf_epoch, trie, get_context, _, _ = ev.setup()

    print("loading ranker + building ranking features...")
    with open(os.path.join(PROCESSED_DIR, "ranker_model.pkl"), "rb") as f:
        ranker = pickle.load(f)
    item_features = build_item_features(train, n_items)
    user_profiles = build_user_profile_embeddings(train_sequences, item_embeddings)
    user_n_interactions = build_user_interaction_counts(train_sequences)

    print(f"setup done in {time.time() - t_start:.1f}s")

    # ---- per-tier recommend functions, each falling back to popularity for cold users ----

    def als_fn(user_id, k):
        if als.is_cold(user_id):
            return pop_model.recommend(user_id, k)
        return als.recommend(user_id, k)

    def make_transformer_fn(warm_recs: dict):
        def fn(user_id, k):
            if user_id in warm_recs:
                return warm_recs[user_id][:k]
            return pop_model.recommend(user_id, k)
        return fn

    def make_ranking_fn(warm_recs: dict):
        def fn(user_id, k):
            if user_id in warm_recs:
                return warm_recs[user_id][:k]
            return pop_model.recommend(user_id, k)
        return fn

    def build_transformer_recs(users):
        """Beam search only for users the transformer actually has a context for (warm, in
        train_sequences) -- cold users are left out of this dict and hit the fallback above."""
        warm_users = [u for u in users if get_context(u) is not None]
        contexts = [get_context(u) for u in warm_users]
        recs = {}
        batch = 50
        for start in range(0, len(warm_users), batch):
            bu = warm_users[start:start + batch]
            bc = contexts[start:start + batch]
            ranked_lists = ev.beam_search_batch(tf_params, bc, tf_meta, trie)
            for u, ranked in zip(bu, ranked_lists):
                recs[u] = [item_id for item_id, _ in ranked]
        return recs

    def build_ranking_recs(users):
        """ALS candidates + rerank, only for users ALS has a vector for -- cold users left
        out, same fallback pattern as every other tier."""
        warm_users = np.array([u for u in users if not als.is_cold(u)])
        if len(warm_users) == 0:
            return {}
        cand_ids, cand_scores = als.model.recommend(
            warm_users, als.user_items[warm_users], N=N_RANK_CANDIDATES, filter_already_liked_items=True
        )
        item_ids_per_user = [list(row) for row in cand_ids]
        als_scores_per_user = [list(row) for row in cand_scores]
        feat_df = assemble_features(
            warm_users, item_ids_per_user, als_scores_per_user,
            item_features, user_profiles, user_n_interactions, item_embeddings,
        )
        feat_df["rerank_score"] = ranker.predict(feat_df[FEATURE_COLUMNS].values)
        recs = {}
        for user_id, grp in feat_df.groupby("user_id"):
            recs[user_id] = grp.sort_values("rerank_score", ascending=False)["item_id"].tolist()
        return recs

    all_results = {}
    all_sample_sizes = {}
    splits = ["val", "test"] if args.split == "both" else [args.split]

    for split_name in splits:
        targets_full = val_targets if split_name == "val" else test_targets
        rng = np.random.RandomState(SEED)
        all_users = sorted(targets_full.keys())
        if split_name == "val":
            # the ranker's training labels are these users' val items -> exclude them here
            # (docs/09_audit_sources_and_results.md, bug #2)
            rt_path = os.path.join(PROCESSED_DIR, "ranker_train_users.npy")
            if os.path.exists(rt_path):
                ranker_train_users = set(np.load(rt_path).tolist())
                before = len(all_users)
                all_users = [u for u in all_users if u not in ranker_train_users]
                print(f"excluded {before - len(all_users):,} ranker-training users from the val pool")
        n = min(EVAL_SAMPLE_SIZE, len(all_users))
        sample_users = list(rng.choice(all_users, size=n, replace=False))
        targets = {u: targets_full[u] for u in sample_users}
        n_cold = sum(1 for u in sample_users if get_context(u) is None)
        print(f"\n=== {split_name}: {len(sample_users):,} users sampled from {len(all_users):,} "
              f"({n_cold} cold / {n_cold / len(sample_users):.1%}) ===")

        t0 = time.time()
        transformer_recs = build_transformer_recs(sample_users)
        print(f"  transformer beam search: {time.time() - t0:.1f}s ({len(transformer_recs)} warm users)")

        t0 = time.time()
        ranking_recs = build_ranking_recs(sample_users)
        print(f"  ranking candidates+rerank: {time.time() - t0:.1f}s ({len(ranking_recs)} warm users)")

        tiers = {
            "popularity": pop_model.recommend,
            "als": als_fn,
            "transformer": make_transformer_fn(transformer_recs),
            "als_ranked": make_ranking_fn(ranking_recs),
        }

        split_results = {}
        for name, fn in tiers.items():
            t0 = time.time()
            split_results[name] = evaluate_metrics(fn, targets, K_VALUES)
            print(f"  {name:<14} {split_results[name]}  ({time.time() - t0:.1f}s)")

        all_results[split_name] = split_results
        all_sample_sizes[split_name] = {"n_sampled": len(sample_users), "n_cold": n_cold}

    print("\n" + "=" * 90)
    header_metrics = list(next(iter(all_results[splits[0]].values())).keys())
    print(f"{'split':<6}{'model':<14}" + "".join(f"{m:<11}" for m in header_metrics))
    print("-" * 90)
    for split_name in all_results:
        for model_name, metrics in all_results[split_name].items():
            row = f"{split_name:<6}{model_name:<14}"
            row += "".join(f"{v:<11.4f}" for v in metrics.values())
            print(row)
    print("=" * 90)

    out_path = os.path.join(PROCESSED_DIR, "full_evaluation_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.setdefault("results", {}).update(all_results)
    existing.setdefault("eval_sample_size", {}).update(all_sample_sizes)
    existing["k_values"] = list(K_VALUES)
    existing["transformer_checkpoint_epoch"] = tf_epoch
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")
    print(f"total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
