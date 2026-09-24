"""
Step 6c: generate next-item candidates with the trained Transformer via constrained beam
search, and evaluate with the same Recall@K / NDCG@K used for the Step 4 baselines.

Constrained decoding: the model's raw output is a distribution over 4 tokens (a Semantic
ID), but not every combination of 4 tokens corresponds to a real catalog item. A "trie" (a
nested lookup built directly from every real item's Semantic ID) restricts each generation
step to only tokens that could complete a *real* item -- exactly the point the roadmap
called out ("how constrained/beam decoding maps generated token sequences back to actual
valid items, not just any token sequence corresponds to a real product"). This guarantees
every candidate this script generates is a genuine catalog item; nothing needs to be
discarded afterward for being invalid.

Beam search: at each of the 4 token positions, keep the top BEAM_WIDTH partial sequences by
cumulative log-probability (masked to trie-valid tokens only), extend each, repeat. After the
4th (dedup-digit) step every surviving beam maps to exactly one real item_id.

Because a full beam search forward pass is needed per user, and doing this for all ~55K
val+test users would be expensive on CPU, this evaluates a fixed random sample per split --
documented, not hidden (see docs/06_generative_retrieval.md for the sample size and why).

Run:
    python backend/scripts/evaluate_retrieval.py
"""
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import transformer as tx
from models.metrics import evaluate as evaluate_metrics

PROCESSED_DIR = os.path.join("data", "processed")
D_MODEL, N_HEADS, N_LAYERS, D_FF = 64, 4, 2, 128
BEAM_WIDTH = 20
K_VALUES = (10, 20)
EVAL_SAMPLE_SIZE = 2000
SEED = 42


def build_trie(item_tokens: pd.DataFrame):
    """tok1 -> tok2 -> tok3 -> {tok4: item_id}. Also per-level valid-token sets for masking."""
    trie = {}
    for row in item_tokens.itertuples(index=False):
        t1, t2, t3, t4 = row.tok1, row.tok2, row.tok3, row.tok4
        trie.setdefault(t1, {}).setdefault(t2, {}).setdefault(t3, {})[t4] = row.item_id
    return trie


def load_model():
    import pickle
    with open(os.path.join(PROCESSED_DIR, "transformer_checkpoint.pkl"), "rb") as f:
        raw = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, raw["params"])
    with open(os.path.join(PROCESSED_DIR, "transformer_vocab_meta.json")) as f:
        meta = json.load(f)
    print(f"loaded checkpoint at epoch {raw['epoch']}, last loss {raw['history'][-1]:.4f}")
    return params, meta, raw["epoch"]


def _last_logits(params, seqs, pad_token, n_heads):
    h = tx.forward_hidden(params, seqs, pad_token, n_heads)  # (N, T, D)
    last_pos = jnp.sum(seqs != pad_token, axis=1) - 1  # (N,)
    h_last = jnp.take_along_axis(h, last_pos[:, None, None], axis=1)[:, 0, :]  # (N, D)
    return h_last @ params["tok_emb"].T  # only project the one position we need


_last_logits_jit = jax.jit(_last_logits, static_argnums=(2, 3))  # compiled once per (batch, T) shape


def get_batch_logits(params, seqs, pad_token):
    """seqs: (N, T) int32, each row's real content ends before the first PAD. Returns logits
    (N, vocab) for the position right after each row's last real token."""
    return _last_logits_jit(params, jnp.array(seqs), pad_token, N_HEADS)


def beam_search_batch(params, contexts, meta, trie):
    """contexts: list of variable-length token lists (no padding). Returns, per context, a
    list of (item_id, cum_logprob) sorted best-first, length <= BEAM_WIDTH.

    Vectorized over each beam's valid continuations (numpy fancy-indexing + argpartition)
    instead of a Python loop over every candidate token -- same result, ~10x faster, which
    matters once the model is bigger and the eval sample is thousands of users."""
    pad = meta["pad_token"]
    max_len = meta["max_len"]
    n = len(contexts)

    # beams[i] = list of (token_list, cum_logprob, trie_node) for user i
    beams = [[(list(ctx), 0.0, trie)] for ctx in contexts]

    for step in range(4):
        # pad only to the longest sequence in this batch (rounded up to a multiple of 8 so
        # the jit doesn't recompile for every length) instead of always to max_len -- most
        # histories are far shorter than the cap, and attention cost is quadratic in T
        longest = max(len(toks) for blist in beams for toks, _, _ in blist)
        T = min(max_len, ((longest + 7) // 8) * 8)
        flat_seqs = []
        for blist in beams:
            for toks, _, _ in blist:
                flat_seqs.append((toks + [pad] * (T - len(toks)))[:T])
        logits = np.array(get_batch_logits(params, np.array(flat_seqs, dtype=np.int32), pad))
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)

        new_beams = []
        ptr = 0
        for blist in beams:
            cand_scores, cand_meta = [], []  # per-candidate: score; (beam index, token)
            for b_idx, (toks, cum_lp, node) in enumerate(blist):
                valid = _node_keys(node)
                sc = cum_lp + log_probs[ptr][valid]
                ptr += 1
                cand_scores.append(sc)
                cand_meta.append((b_idx, valid))
            scores = np.concatenate(cand_scores)
            owners = np.concatenate([np.full(len(v), b, dtype=np.int32) for b, v in cand_meta])
            toks_arr = np.concatenate([v for _, v in cand_meta])
            k = min(BEAM_WIDTH, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            nb = []
            for j in top:
                b_idx, tok = int(owners[j]), int(toks_arr[j])
                toks, _, node = blist[b_idx]
                nb.append((toks + [tok], float(scores[j]), node[tok]))
            new_beams.append(nb)
        beams = new_beams

    # after the 4th step each beam's "node" is the item_id stored at the trie leaf
    return [[(item_id, lp) for _, lp, item_id in blist] for blist in beams]


_node_keys_cache = {}


def _node_keys(node):
    """Sorted numpy array of the valid next tokens at a trie node (cached per node)."""
    key = id(node)
    arr = _node_keys_cache.get(key)
    if arr is None:
        arr = np.fromiter(node.keys(), dtype=np.int64, count=len(node))
        _node_keys_cache[key] = arr
    return arr


def _valid_next_tokens(trie, generated_suffix):
    """Walk the trie using only the tokens generated so far *for this item* (not the user's
    context), return the valid next-token options at this depth."""
    node = trie
    for tok in generated_suffix:
        node = node[tok]
    return node  # dict: token -> (subtrie, or item_id at the last level)


def evaluate_split(split_name, params, meta, trie, get_context, targets, sample_size=EVAL_SAMPLE_SIZE, batch=50):
    rng = np.random.RandomState(SEED)
    eligible = [u for u in targets if get_context(u) is not None]
    n = min(sample_size, len(eligible))
    users = list(rng.choice(eligible, size=n, replace=False))
    print(f"\n{split_name}: evaluating {len(users):,} users (of {len(targets):,} total)")

    contexts = [get_context(u) for u in users]
    recommend_cache = {}
    for start in range(0, len(users), batch):
        batch_users = users[start:start + batch]
        batch_contexts = contexts[start:start + batch]
        ranked_lists = beam_search_batch(params, batch_contexts, meta, trie)
        for u, ranked in zip(batch_users, ranked_lists):
            recommend_cache[u] = [item_id for item_id, _ in ranked]
        if start % (batch * 10) == 0:
            print(f"  {start:,}/{len(users):,}")

    def recommend_fn(user_id, k):
        return recommend_cache.get(user_id, [])[:k]

    subset_targets = {u: targets[u] for u in users}
    result = evaluate_metrics(recommend_fn, subset_targets, K_VALUES)
    print(f"{split_name} results ({len(users)} users): {result}")
    return result, len(users)


def setup():
    params, meta, checkpoint_epoch = load_model()
    item_tokens = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_tokens.parquet"))
    trie = build_trie(item_tokens)

    seq_data = np.load(os.path.join(PROCESSED_DIR, "transformer_train_sequences.npz"))
    train_tokens, train_lengths, train_user_ids = seq_data["tokens"], seq_data["lengths"], seq_data["user_ids"]
    user_to_row = {u: i for i, u in enumerate(train_user_ids)}
    max_len = meta["max_len"]
    context_cap = max_len - 4

    def get_context(user_id):
        row = user_to_row.get(user_id)
        if row is None:
            return None
        length = min(int(train_lengths[row]), context_cap)
        return list(train_tokens[row, :length])

    def load_targets(path):
        df = pd.read_parquet(path)
        return {row.user_id: set(row.item_ids) for row in df.itertuples()}

    val_targets = load_targets(os.path.join(PROCESSED_DIR, "val_targets.parquet"))
    test_targets = load_targets(os.path.join(PROCESSED_DIR, "test_targets.parquet"))
    return params, meta, checkpoint_epoch, trie, get_context, val_targets, test_targets


def save_results(new_results, checkpoint_epoch):
    out_path = os.path.join(PROCESSED_DIR, "transformer_eval_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.setdefault("results", {}).update(new_results)
    existing["eval_sample_size"] = EVAL_SAMPLE_SIZE
    existing["beam_width"] = BEAM_WIDTH
    existing["checkpoint_epoch"] = checkpoint_epoch
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = ap.parse_args()

    params, meta, checkpoint_epoch, trie, get_context, val_targets, test_targets = setup()

    new_results = {}
    if args.split in ("val", "both"):
        r, n = evaluate_split("val", params, meta, trie, get_context, val_targets)
        new_results["val"] = r
    if args.split in ("test", "both"):
        r, n = evaluate_split("test", params, meta, trie, get_context, test_targets)
        new_results["test"] = r

    save_results(new_results, checkpoint_epoch)
