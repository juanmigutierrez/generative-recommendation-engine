"""
SASRec (Kang & McAuley 2018, "Self-Attentive Sequential Recommendation", arXiv:1808.09781)
-- the reference baseline TIGER compares against. Same causal Transformer as
models/transformer.py, but over *raw item IDs* (one token per item) instead of Semantic ID
digits, and scored with a dot product between the last hidden state and the item embedding
table instead of beam search. That makes it the cleanest control for "does the Semantic ID
tokenization help?": identical backbone, different vocabulary.

Differences from the original paper, stated so the comparison is honest: the original
trains with a binary cross-entropy loss over one sampled negative per position; this uses
full softmax cross-entropy over the whole catalog (the "SASRec+ / CE" variant, which most
re-implementations use and which is usually a bit *stronger* than the sampled-BCE
original). Item embeddings are tied between input and output, as in the original.
"""
import jax.numpy as jnp
import numpy as np

from . import transformer as tx


def build_sequences(train_sequences, n_items: int, max_items: int):
    """train_sequences: DataFrame(user_id, item_ids). Returns (tokens (U, max_items) int32
    right-padded with PAD = n_items, lengths, user_ids)."""
    pad = n_items
    n_users = len(train_sequences)
    tokens = np.full((n_users, max_items), pad, dtype=np.int32)
    lengths = np.zeros(n_users, dtype=np.int32)
    user_ids = np.zeros(n_users, dtype=np.int64)
    for i, row in enumerate(train_sequences.itertuples(index=False)):
        items = list(row.item_ids)[-max_items:]
        tokens[i, :len(items)] = items
        lengths[i] = len(items)
        user_ids[i] = row.user_id
    return tokens, lengths, user_ids


def loss_fn(params, tokens, pad_token, n_heads, dropout=0.0, key=None):
    """Full-softmax next-item cross-entropy (same shape as the Semantic ID model's loss)."""
    return tx.loss_fn(params, tokens, pad_token, n_heads, dropout, key)


def score_batch(params, tokens, pad_token, n_heads):
    """tokens: (B, T) right-padded item histories. Returns (B, n_items) scores for the
    next item = last hidden state . item embedding table (the PAD row is excluded)."""
    h = tx.forward_hidden(params, tokens, pad_token, n_heads)  # (B, T, D)
    last_pos = jnp.sum(tokens != pad_token, axis=1) - 1
    h_last = jnp.take_along_axis(h, last_pos[:, None, None], axis=1)[:, 0, :]  # (B, D)
    return h_last @ params["tok_emb"][:pad_token].T  # (B, n_items)
