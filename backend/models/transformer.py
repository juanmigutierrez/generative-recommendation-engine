"""
Decoder-only Transformer (GPT-style) for generative sequential retrieval -- reads a user's
history as a sequence of Semantic ID tokens (see build_semantic_sequences.py) and learns to
predict the next token, exactly like a language model predicts the next word. This is the
architectural bet TIGER (Google DeepMind) and Meta's HSTU/generative-recommenders both make:
retrieval as generation, not nearest-neighbor search.

Hand-rolled in plain JAX, same philosophy as rqvae.py -- every layer (embedding, causal
multi-head self-attention, layernorm, feed-forward, weight-tied output projection) is code
you can point to and explain, not a library call. The one place this leans on a library
convention rather than deriving it fresh is multi-head attention's shape bookkeeping
(reshape into heads, scaled dot-product, reshape back) -- standard since "Attention Is All
You Need" (Vaswani et al., 2017), reimplemented directly rather than imported.

Architecture (GPT-2-style "pre-norm" block, one of a few standard variants):
    x = token_embedding(tokens) + position_embedding(positions)
    for each of n_layers blocks:
        x = x + causal_self_attention(layernorm(x))
        x = x + feed_forward(layernorm(x))
    x = layernorm(x)
    logits = x @ token_embedding.T   (weight tying: output layer shares the input embedding
                                       table, a standard GPT trick that halves the parameter
                                       count of the two largest matrices in a small model)
"""
import jax
import jax.numpy as jnp
import numpy as np


def init_params(key, vocab_size, max_len, d_model, n_heads, n_layers, d_ff):
    assert d_model % n_heads == 0
    keys = jax.random.split(key, 2 + n_layers)

    def linear_init(k, fan_in, fan_out):
        scale = jnp.sqrt(2.0 / fan_in)
        return jax.random.normal(k, (fan_in, fan_out)) * scale

    params = {
        "tok_emb": jax.random.normal(keys[0], (vocab_size, d_model)) * 0.02,
        "pos_emb": jax.random.normal(keys[1], (max_len, d_model)) * 0.02,
        "blocks": [],
        "ln_f_g": jnp.ones(d_model),
        "ln_f_b": jnp.zeros(d_model),
    }
    for l in range(n_layers):
        bk = jax.random.split(keys[2 + l], 6)
        params["blocks"].append({
            "ln1_g": jnp.ones(d_model), "ln1_b": jnp.zeros(d_model),
            "wq": linear_init(bk[0], d_model, d_model),
            "wk": linear_init(bk[1], d_model, d_model),
            "wv": linear_init(bk[2], d_model, d_model),
            "wo": linear_init(bk[3], d_model, d_model),
            "ln2_g": jnp.ones(d_model), "ln2_b": jnp.zeros(d_model),
            "w1": linear_init(bk[4], d_model, d_ff), "b1": jnp.zeros(d_ff),
            "w2": linear_init(bk[5], d_ff, d_model), "b2": jnp.zeros(d_model),
        })
    return params


def layernorm(x, g, b, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * g + b


def causal_self_attention(x, block, n_heads, attn_bias):
    """attn_bias: (T, T) additive mask -- 0 where attention is allowed, -1e9 where it's
    blocked (causal: can't see the future; padding: can't attend to PAD positions)."""
    B, T, D = x.shape
    hd = D // n_heads

    q = x @ block["wq"]
    k = x @ block["wk"]
    v = x @ block["wv"]

    def split_heads(t):
        return t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)  # (B, H, T, hd)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(hd)  # (B, H, T, T)
    scores = scores + attn_bias[None, None, :, :]
    attn = jax.nn.softmax(scores, axis=-1)
    out = attn @ v  # (B, H, T, hd)

    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ block["wo"]


def feed_forward(x, block):
    h = jax.nn.gelu(x @ block["w1"] + block["b1"])
    return h @ block["w2"] + block["b2"]


def _dropout(x, rate, key):
    if key is None or rate <= 0.0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def forward(params, tokens, pad_token, n_heads, dropout=0.0, key=None):
    """tokens: (B, T) int32. Returns logits (B, T, vocab_size).

    dropout/key: training-time regularization (TIGER uses 0.1). Pass key=None at inference
    (no dropout). Applied to the embedding sum and to each residual branch's output, the
    standard GPT-2 placement."""
    B, T = tokens.shape
    positions = jnp.arange(T)
    n_drop = 1 + 2 * len(params["blocks"])
    keys = jax.random.split(key, n_drop) if key is not None else [None] * n_drop

    x = params["tok_emb"][tokens] + params["pos_emb"][positions][None, :, :]
    x = _dropout(x, dropout, keys[0])

    causal = jnp.where(jnp.arange(T)[:, None] >= jnp.arange(T)[None, :], 0.0, -1e9)
    pad_mask = jnp.where(tokens == pad_token, -1e9, 0.0)  # (B, T) -- can't attend to PAD keys

    # Build one bias matrix per batch item (causal rule AND can't attend to PAD keys), then
    # run every example through the stack of blocks with jax.vmap over the batch dimension --
    # avoids a python-level loop over the batch.
    bias_bt = causal[None, :, :] + pad_mask[:, None, :]  # (B, T, T)

    def run_one(x_b, bias_b):
        h = x_b
        for l, block in enumerate(params["blocks"]):
            a = causal_self_attention(
                layernorm(h, block["ln1_g"], block["ln1_b"])[None], block, n_heads, bias_b
            )[0]
            h = h + _dropout(a, dropout, keys[1 + 2 * l])
            f = feed_forward(layernorm(h, block["ln2_g"], block["ln2_b"]), block)
            h = h + _dropout(f, dropout, keys[2 + 2 * l])
        h = layernorm(h, params["ln_f_g"], params["ln_f_b"])
        return h

    h = jax.vmap(run_one)(x, bias_bt)  # (B, T, D)
    logits = h @ params["tok_emb"].T  # weight tying
    return logits


def forward_hidden(params, tokens, pad_token, n_heads):
    """Same as forward() but returns the final hidden states (B, T, D) instead of logits --
    used by models/sasrec.py, which scores items with a dot product against the embedding
    table only at the last position (cheaper than materializing (B, T, n_items) logits)."""
    B, T = tokens.shape
    positions = jnp.arange(T)
    x = params["tok_emb"][tokens] + params["pos_emb"][positions][None, :, :]
    causal = jnp.where(jnp.arange(T)[:, None] >= jnp.arange(T)[None, :], 0.0, -1e9)
    pad_mask = jnp.where(tokens == pad_token, -1e9, 0.0)
    bias_bt = causal[None, :, :] + pad_mask[:, None, :]

    def run_one(x_b, bias_b):
        h = x_b
        for block in params["blocks"]:
            h = h + causal_self_attention(
                layernorm(h, block["ln1_g"], block["ln1_b"])[None], block, n_heads, bias_b
            )[0]
            h = h + feed_forward(layernorm(h, block["ln2_g"], block["ln2_b"]), block)
        return layernorm(h, params["ln_f_g"], params["ln_f_b"])

    return jax.vmap(run_one)(x, bias_bt)


def loss_fn(params, tokens, pad_token, n_heads, dropout=0.0, key=None):
    """Standard causal LM loss: predict token[t+1] from tokens[:t+1], averaged over all
    non-PAD target positions."""
    logits = forward(params, tokens, pad_token, n_heads, dropout, key)
    logits = logits[:, :-1, :]  # predictions for positions 0..T-2
    targets = tokens[:, 1:]  # actual next tokens for positions 1..T-1
    mask = (targets != pad_token).astype(jnp.float32)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    loss = jnp.sum(nll * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss


def make_adam(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    def init(params):
        return {"m": jax.tree_util.tree_map(jnp.zeros_like, params),
                "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": 0}

    def update(grads, state, params):
        t = state["t"] + 1
        m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
        v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * (g**2), state["v"], grads)
        m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
        )
        return new_params, {"m": m, "v": v, "t": t}

    return init, update


def next_token_logits(params, tokens, pad_token, n_heads):
    """For generation: given a (1, T) prefix, return logits for the position right after the
    last real (non-PAD) token."""
    logits = forward(params, tokens, pad_token, n_heads)
    last_pos = jnp.sum(tokens != pad_token) - 1
    return logits[0, last_pos, :]
