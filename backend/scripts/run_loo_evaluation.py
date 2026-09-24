"""
Step 10c: paper-comparable evaluation -- every model on the leave-one-out split
(split_data_loo.py), the protocol TIGER / SASRec / most sequential-recsys papers report on.

Tiers:
    popularity      -- floor
    als             -- collaborative filtering (Hu, Koren & Volinsky 2008)
    sasrec          -- raw-item-ID Transformer, the reference baseline TIGER compares against
                       (needs data/processed/sasrec_checkpoint.pkl from train_sasrec.py)
    transformer     -- Semantic ID generative retrieval, constrained beam search
                       (needs transformer_checkpoint_loo.pkl from train_transformer.py --loo)

Protocol details, all matching the papers:
  * val:  context = user's items[:-2], target = items[-2]
    test: context = user's items[:-1] (val item included), target = items[-1]
  * every model's output has the user's context items filtered out (the papers rank the
    target against items the user has not interacted with)
  * metrics: Recall@K and NDCG@K for K in 5, 10, 20 (TIGER reports @5 and @10; with a single
    target per user Recall@K == HitRate@K, the number SASRec reports)
  * evaluated on a random sample of users (default 2,000, `--n-users 0` for all ~95K; beam
    search is the slow part, ~40s per 1,000 users per model on a laptop CPU)

Any tier whose checkpoint is missing is skipped with a note, so this can be run before the
neural models finish training.

Run:
    python backend/scripts/run_loo_evaluation.py --split both --n-users 2000
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.als import ALSRecommender
from models.popularity import PopularityRecommender
from models.metrics import evaluate as evaluate_metrics
from models import sasrec
import evaluate_retrieval as ev

PROCESSED_DIR = os.path.join("data", "processed")
K_VALUES = (5, 10, 20)
SEED = 7
EXTRA = 50  # over-fetch so that filtering the user's own history never leaves < max(K) items


def load_seq(path):
    df = pd.read_parquet(path)
    return {row.user_id: list(row.item_ids) for row in df.itertuples()}


def load_targets(path):
    df = pd.read_parquet(path)
    return {row.user_id: set(row.item_ids) for row in df.itertuples()}


def filtered(fn):
    """Wrap a recommend(user, k) function so the user's context items are removed."""
    def inner(user_id, k, seen):
        recs = fn(user_id, k + EXTRA)
        return [i for i in recs if i not in seen][:k]
    return inner


# ---------------------------------------------------------------- SASRec tier
def build_sasrec_recs(users, contexts, seen_sets, n_items, k_max):
    path = os.path.join(PROCESSED_DIR, "sasrec_checkpoint.pkl")
    if not os.path.exists(path):
        print("  [sasrec] no checkpoint -> skipped (run train_sasrec.py)")
        return None
    with open(path, "rb") as f:
        raw = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, raw["params"])
    cfg = raw["config"]
    pad = n_items
    print(f"  [sasrec] checkpoint epoch {raw['epoch']}, loss {raw['history'][-1]:.4f}")
    score_fn = jax.jit(lambda t: sasrec.score_batch(params, t, pad, cfg["n_heads"]))
    recs = {}
    batch = 256
    for start in range(0, len(users), batch):
        bu = users[start:start + batch]
        toks = np.full((len(bu), cfg["max_items"]), pad, dtype=np.int32)
        for i, u in enumerate(bu):
            items = contexts[u][-cfg["max_items"]:]
            toks[i, :len(items)] = items
        scores = np.array(score_fn(jnp.array(toks)))
        for i, u in enumerate(bu):
            s = scores[i]
            if seen_sets[u]:
                s[list(seen_sets[u])] = -np.inf
            top = np.argpartition(-s, k_max)[:k_max]
            recs[u] = top[np.argsort(-s[top])].tolist()
    return recs


# ---------------------------------------------------------------- Semantic ID transformer tier
def build_transformer_recs(users, contexts, seen_sets, k_max):
    ckpt = os.path.join(PROCESSED_DIR, "transformer_checkpoint_loo.pkl")
    meta_path = os.path.join(PROCESSED_DIR, "transformer_vocab_meta_loo.json")
    if not (os.path.exists(ckpt) and os.path.exists(meta_path)):
        print("  [transformer] no LOO checkpoint/meta -> skipped "
              "(run build_semantic_sequences.py --loo ... and train_transformer.py --loo ...)")
        return None
    with open(ckpt, "rb") as f:
        raw = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, raw["params"])
    cfg = raw.get("config", {"n_heads": 4})
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  [transformer] checkpoint epoch {raw['epoch']}, loss {raw['history'][-1]:.4f}")

    # evaluate_retrieval's beam search reads these module-level constants
    ev.N_HEADS = cfg["n_heads"]
    ev.BEAM_WIDTH = k_max + 10  # headroom for filtering the user's own history

    item_tokens = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_tokens.parquet"))
    trie = ev.build_trie(item_tokens)
    item_to_tok = {row.item_id: [row.tok1, row.tok2, row.tok3, row.tok4]
                   for row in item_tokens.itertuples(index=False)}
    user_buckets = meta.get("user_buckets", 0)
    item_vocab = meta.get("item_vocab", 789)
    max_ctx_items = (meta["max_len"] - (1 if user_buckets else 0)) // 4 - 1  # leave room for 4 generated tokens

    def context_tokens(u):
        toks = [item_vocab + int(u) % user_buckets] if user_buckets else []
        for it in contexts[u][-max_ctx_items:]:
            toks.extend(item_to_tok[it])
        return toks

    recs = {}
    batch = 50
    t0 = time.time()
    for start in range(0, len(users), batch):
        bu = users[start:start + batch]
        ranked = ev.beam_search_batch(params, [context_tokens(u) for u in bu], meta, trie)
        for u, r in zip(bu, ranked):
            recs[u] = [i for i, _ in r if i not in seen_sets[u]][:k_max]
        if start % (batch * 20) == 0 and start:
            print(f"    beam search {start:,}/{len(users):,} ({time.time() - t0:.0f}s)")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--n-users", type=int, default=2000, help="0 = all users")
    args = ap.parse_args()

    t_start = time.time()
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "loo_train.parquet"))
    n_users_total = len(pd.read_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet")))
    n_items = len(pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet")))
    train_seq = load_seq(os.path.join(PROCESSED_DIR, "loo_train_sequences.parquet"))
    val_seq = load_seq(os.path.join(PROCESSED_DIR, "loo_val_sequences.parquet"))
    val_targets = load_targets(os.path.join(PROCESSED_DIR, "loo_val_targets.parquet"))
    test_targets = load_targets(os.path.join(PROCESSED_DIR, "loo_test_targets.parquet"))
    k_max = max(K_VALUES)

    print("fitting popularity + ALS on LOO train...")
    pop = PopularityRecommender().fit(train)
    als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(train, n_users_total, n_items)
    pop_fn = filtered(pop.recommend)
    als_fn = filtered(lambda u, k: als.recommend(u, k) if not als.is_cold(u) else pop.recommend(u, k))
    print(f"setup done in {time.time() - t_start:.1f}s")

    all_results, sizes = {}, {}
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        targets_full = val_targets if split == "val" else test_targets
        contexts = train_seq if split == "val" else val_seq
        rng = np.random.RandomState(SEED)
        all_users = sorted(targets_full.keys())
        users = all_users if args.n_users == 0 else list(rng.choice(all_users, size=min(args.n_users, len(all_users)), replace=False))
        seen_sets = {u: set(contexts[u]) for u in users}
        targets = {u: targets_full[u] for u in users}
        print(f"\n=== {split}: {len(users):,} users ===")

        t0 = time.time()
        sas_recs = build_sasrec_recs(users, contexts, seen_sets, n_items, k_max)
        print(f"  sasrec scoring: {time.time() - t0:.1f}s")
        t0 = time.time()
        tf_recs = build_transformer_recs(users, contexts, seen_sets, k_max)
        print(f"  transformer beam search: {time.time() - t0:.1f}s")

        tiers = {
            "popularity": lambda u, k: pop_fn(u, k, seen_sets[u]),
            "als": lambda u, k: als_fn(u, k, seen_sets[u]),
        }
        if sas_recs is not None:
            tiers["sasrec"] = lambda u, k: sas_recs[u][:k]
        if tf_recs is not None:
            tiers["transformer"] = lambda u, k: tf_recs[u][:k]

        split_results = {}
        for name, fn in tiers.items():
            r = evaluate_metrics(fn, targets, K_VALUES)
            split_results[name] = {m: v for m, v in r.items() if not m.startswith("map")}
            print(f"  {name:<12} " + "  ".join(f"{m} {v:.4f}" for m, v in split_results[name].items()))
        all_results[split] = split_results
        sizes[split] = len(users)

    print("\n" + "=" * 100)
    metrics = list(next(iter(all_results[splits[0]].values())).keys())
    print(f"{'split':<6}{'model':<13}" + "".join(f"{m:<11}" for m in metrics))
    print("-" * 100)
    for split, res in all_results.items():
        for name, vals in res.items():
            print(f"{split:<6}{name:<13}" + "".join(f"{vals[m]:<11.4f}" for m in metrics))
    print("=" * 100)
    print("Reference (TIGER paper, Amazon Beauty, same protocol): SASRec R@10 0.0605 / N@10 0.0318, "
          "TIGER R@10 0.0648 / N@10 0.0384. Video_Games is a different category, so compare the "
          "transformer against *this table's* SASRec, not the paper's absolute numbers.")

    out = os.path.join(PROCESSED_DIR, "loo_evaluation_results.json")
    existing = {}
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
    existing.setdefault("results", {}).update(all_results)
    existing.setdefault("n_users", {}).update(sizes)
    existing["k_values"] = list(K_VALUES)
    existing["protocol"] = "leave-one-out (last item test, second-to-last val), context items filtered"
    with open(out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nsaved -> {out}  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
