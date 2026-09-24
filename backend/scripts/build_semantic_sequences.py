"""
Step 6a: turn each user's chronological item history into a sequence of Semantic ID tokens
the Transformer can train on.

Tokenization scheme: each item's 4-digit Semantic ID (level1, level2, level3, dedup -- see
Step 5 / docs/05_semantic_ids.md) is flattened into 4 consecutive tokens in a single shared
vocabulary, each level given its own disjoint offset range so one embedding table and one
output layer cover everything:

    level 1 codes -> token ids   0 .. 255   (256 codes)
    level 2 codes -> token ids 256 .. 511   (256 codes)
    level 3 codes -> token ids 512 .. 767   (256 codes)
    dedup digit   -> token ids 768 .. 788   (21 values -- max observed dedup digit is 20)
    PAD           -> token id  789

A user's history becomes: [item1_tok1, item1_tok2, item1_tok3, item1_tok4, item2_tok1, ...] --
one long token stream, exactly like a language model's token stream, just with a vocabulary
built from item content instead of words. This is what lets Step 6's Transformer treat next-
item prediction as next-token prediction.

Sequences are right-truncated to the most recent MAX_ITEMS items (recency matters most, same
reasoning as the time-based split) and right-padded to a fixed length for batching.

Run:
    python backend/scripts/build_semantic_sequences.py
    python backend/scripts/build_semantic_sequences.py --loo --max-items 20 --user-buckets 2000
        (leave-one-out variant for the paper-comparable protocol, see split_data_loo.py:
         20-item context like TIGER, plus a hashed user-ID token prepended to every sequence,
         which TIGER uses so the model can personalize beyond the item history alone)
"""
import argparse
import os

import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

LEVEL_SIZE = 256
N_LEVELS = 3
MAX_DEDUP = 20  # inclusive; verified against semantic_ids.parquet at build time
DEDUP_VOCAB = MAX_DEDUP + 1
ITEM_VOCAB = N_LEVELS * LEVEL_SIZE + DEDUP_VOCAB  # 789 item-digit tokens
PAD_TOKEN = ITEM_VOCAB  # 789 (default layout, no user tokens)
VOCAB_SIZE = PAD_TOKEN + 1  # 790

MAX_ITEMS = 10  # covers the median (6) and mean (7.5) train sequence length comfortably;
# a larger MAX_ITEMS (e.g. 20, covering the 95th percentile) was tried first but pushed one
# training epoch's compute past the sandbox's ~170s-per-call budget (roughly linear in total
# tokens processed) -- see docs/06_generative_retrieval.md for the timing numbers behind this.
MAX_LEN = MAX_ITEMS * 4  # 40 tokens


def item_to_tokens(sid_row):
    l1, l2, l3, dedup = sid_row
    assert dedup <= MAX_DEDUP, f"dedup digit {dedup} exceeds assumed max {MAX_DEDUP} -- widen DEDUP_VOCAB"
    return [l1, LEVEL_SIZE + l2, 2 * LEVEL_SIZE + l3, N_LEVELS * LEVEL_SIZE + dedup]


def main(loo=False, max_items=MAX_ITEMS, user_buckets=0):
    # layout: [item digit tokens][user-bucket tokens (optional)][PAD]
    pad_token = ITEM_VOCAB + user_buckets
    vocab_size = pad_token + 1
    max_len = max_items * 4 + (1 if user_buckets else 0)
    suffix = "_loo" if loo else ""
    seq_file = "loo_train_sequences.parquet" if loo else "train_sequences.parquet"

    semantic_ids = pd.read_parquet(os.path.join(PROCESSED_DIR, "semantic_ids.parquet"))
    assert semantic_ids["sid_dedup"].max() <= MAX_DEDUP, "MAX_DEDUP assumption violated by real data"

    item_to_tok = {}
    for row in semantic_ids.itertuples(index=False):
        item_to_tok[row.item_id] = item_to_tokens(
            (row.sid_level_1, row.sid_level_2, row.sid_level_3, row.sid_dedup)
        )
    print(f"tokenized {len(item_to_tok):,} items, vocab size {vocab_size}")

    train_seq = pd.read_parquet(os.path.join(PROCESSED_DIR, seq_file))
    n_users = len(train_seq)

    token_matrix = np.full((n_users, max_len), pad_token, dtype=np.int32)
    seq_lengths = np.zeros(n_users, dtype=np.int32)  # actual token count (before padding)
    user_ids = np.zeros(n_users, dtype=np.int64)

    for i, row in enumerate(train_seq.itertuples(index=False)):
        items = list(row.item_ids)[-max_items:]  # keep the MOST RECENT items if too long
        toks = [ITEM_VOCAB + user_token_bucket(row.user_id, user_buckets)] if user_buckets else []
        for it in items:
            toks.extend(item_to_tok[it])
        n = len(toks)
        token_matrix[i, :n] = toks
        seq_lengths[i] = n
        user_ids[i] = row.user_id

    print(f"{n_users:,} user sequences, shape {token_matrix.shape}")
    print(f"token length stats: min {seq_lengths.min()}, median {np.median(seq_lengths):.0f}, "
          f"mean {seq_lengths.mean():.1f}, max {seq_lengths.max()}")

    np.savez(
        os.path.join(PROCESSED_DIR, f"transformer_train_sequences{suffix}.npz"),
        tokens=token_matrix,
        lengths=seq_lengths,
        user_ids=user_ids,
    )
    print(f"saved -> {PROCESSED_DIR}/transformer_train_sequences{suffix}.npz")

    # also save the full item_id -> token list mapping (every catalog item, not just users'
    # histories) -- needed at eval time to build the trie for constrained decoding
    item_tok_df = pd.DataFrame(
        [(item_id, *toks) for item_id, toks in item_to_tok.items()],
        columns=["item_id", "tok1", "tok2", "tok3", "tok4"],
    )
    item_tok_df.to_parquet(os.path.join(PROCESSED_DIR, "item_tokens.parquet"))
    print(f"saved -> {PROCESSED_DIR}/item_tokens.parquet")

    meta = {
        "vocab_size": vocab_size,
        "pad_token": pad_token,
        "level_size": LEVEL_SIZE,
        "n_levels": N_LEVELS,
        "dedup_vocab": DEDUP_VOCAB,
        "item_vocab": ITEM_VOCAB,
        "user_buckets": user_buckets,
        "max_items": max_items,
        "max_len": max_len,
        "loo": loo,
    }
    import json
    with open(os.path.join(PROCESSED_DIR, f"transformer_vocab_meta{suffix}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved -> {PROCESSED_DIR}/transformer_vocab_meta{suffix}.json")


def user_token_bucket(user_id: int, n_buckets: int) -> int:
    """TIGER hashes user ids into a fixed number of buckets (2000 in the paper) so the user
    token vocabulary stays small; a plain modulo is a deterministic stand-in for that hash."""
    return int(user_id) % n_buckets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo", action="store_true", help="use the leave-one-out split (split_data_loo.py)")
    ap.add_argument("--max-items", type=int, default=MAX_ITEMS)
    ap.add_argument("--user-buckets", type=int, default=0, help="prepend a hashed user-ID token (0 = off)")
    args = ap.parse_args()
    main(loo=args.loo, max_items=args.max_items, user_buckets=args.user_buckets)
