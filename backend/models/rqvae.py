"""
RQ-VAE (Residual-Quantized VAE) -- turns a continuous item content embedding into a short
tuple of discrete codes, the item's "Semantic ID". This is the mechanism TIGER (Google
DeepMind, NeurIPS 2023) uses to give the generative retrieval model in Step 6 something it
can predict autoregressively, like a language model predicting tokens.

Framework note: implemented in JAX rather than PyTorch. PyTorch's CPU-only wheel is only
published on download.pytorch.org, which this sandbox's network can't reach; the default
PyPI `torch` wheel pulls in several GB of CUDA dependencies that don't fit the sandbox's
disk quota. JAX's CPU wheel is self-contained and lives directly on PyPI. The autodiff
concepts (forward pass, straight-through gradient through the quantization step, backprop
into encoder/decoder/codebook weights) are the same either way.

Architecture:
    encoder:  Linear(D -> H) -> ReLU -> Linear(H -> d)          content embedding -> latent
    residual VQ: L codebooks of K vectors each, applied in sequence -- each codebook
                 quantizes what the previous one got wrong (its residual)
    decoder:  Linear(d -> H) -> ReLU -> Linear(H -> D)          quantized latent -> reconstruction

See docs/05_semantic_ids.md for the full walkthrough of why residual quantization (vs. a
single codebook) and how the straight-through estimator works.
"""
import jax
import jax.numpy as jnp
import numpy as np


def init_params(key, in_dim, hidden_dim, latent_dim, n_codebooks, codebook_size):
    keys = jax.random.split(key, 6)

    def linear_init(k, fan_in, fan_out):
        scale = jnp.sqrt(2.0 / fan_in)
        return jax.random.normal(k, (fan_in, fan_out)) * scale

    params = {
        "enc_w1": linear_init(keys[0], in_dim, hidden_dim),
        "enc_b1": jnp.zeros(hidden_dim),
        "enc_w2": linear_init(keys[1], hidden_dim, latent_dim),
        "enc_b2": jnp.zeros(latent_dim),
        "dec_w1": linear_init(keys[2], latent_dim, hidden_dim),
        "dec_b1": jnp.zeros(hidden_dim),
        "dec_w2": linear_init(keys[3], hidden_dim, in_dim),
        "dec_b2": jnp.zeros(in_dim),
        # one codebook per quantization level, each (codebook_size, latent_dim)
        "codebooks": jax.random.normal(keys[4], (n_codebooks, codebook_size, latent_dim)) * 0.05,
    }
    return params


def kmeans_init_codebooks(key, encoder_outputs, n_codebooks, codebook_size, n_iters=15):
    """Data-dependent codebook initialization via a few rounds of k-means, run level-by-level
    on the residual stream (same structure as residual_quantize itself).

    Why this exists: random-near-origin codebook init (the original init_params behavior)
    works fine when the encoder's output distribution happens to already be near the origin
    and reasonably spread out -- true for the TF-IDF/SVD embeddings this was first built for.
    It silently fails for embeddings from a real pretrained model like Sentence-T5, whose
    outputs are known to be anisotropic (cluster tightly in a narrow cone of space, a
    well-documented property of BERT-family sentence embeddings) and live at a different
    scale than a small-random-noise codebook init assumes. The result, observed directly
    when retraining on real embeddings: total codebook collapse -- every item mapped to the
    exact same code, because the codebook never started anywhere near the encoder's actual
    output region. Seeding each codebook from the real (randomly-initialized-encoder) output
    distribution via k-means fixes this at the source, before training even starts.
    """
    codebooks = []
    residual = encoder_outputs
    keys = jax.random.split(key, n_codebooks)
    for l in range(n_codebooks):
        n_points = residual.shape[0]
        init_idx = jax.random.choice(keys[l], n_points, shape=(codebook_size,), replace=False)
        centers = residual[init_idx]
        for _ in range(n_iters):
            dists = (
                jnp.sum(residual**2, axis=-1, keepdims=True)
                - 2 * residual @ centers.T
                + jnp.sum(centers**2, axis=-1)[None, :]
            )
            assign = jnp.argmin(dists, axis=-1)
            onehot = jax.nn.one_hot(assign, codebook_size)
            counts = jnp.sum(onehot, axis=0)
            sums = onehot.T @ residual
            # codes with zero points assigned this round keep their previous position
            # rather than collapsing to a zero vector
            centers = jnp.where(counts[:, None] > 0, sums / jnp.maximum(counts, 1)[:, None], centers)
        codebooks.append(centers)
        dists = (
            jnp.sum(residual**2, axis=-1, keepdims=True)
            - 2 * residual @ centers.T
            + jnp.sum(centers**2, axis=-1)[None, :]
        )
        idx = jnp.argmin(dists, axis=-1)
        residual = residual - centers[idx]
    return jnp.stack(codebooks, axis=0)


def encode(params, x):
    h = jax.nn.relu(x @ params["enc_w1"] + params["enc_b1"])
    z = h @ params["enc_w2"] + params["enc_b2"]
    return z


def decode(params, z_q):
    h = jax.nn.relu(z_q @ params["dec_w1"] + params["dec_b1"])
    x_hat = h @ params["dec_w2"] + params["dec_b2"]
    return x_hat


def residual_quantize(codebooks, z):
    """Quantize z one codebook at a time, each level correcting the previous level's
    leftover error (the "residual"). Returns per-level (quantized_vector, code_index,
    residual_input) so the caller can build both the commitment loss and the codebook loss.
    """
    n_codebooks = codebooks.shape[0]
    residual = z
    level_quantized = []
    level_indices = []
    level_inputs = []
    for l in range(n_codebooks):
        cb = codebooks[l]  # (K, d)
        # squared distance from this level's residual to every code in this level's codebook
        dists = (
            jnp.sum(residual**2, axis=-1, keepdims=True)
            - 2 * residual @ cb.T
            + jnp.sum(cb**2, axis=-1)[None, :]
        )
        idx = jnp.argmin(dists, axis=-1)
        q = cb[idx]
        level_inputs.append(residual)
        level_quantized.append(q)
        level_indices.append(idx)
        residual = residual - q
    return level_quantized, level_indices, level_inputs


def forward(params, x, beta=0.25):
    z = encode(params, x)
    level_quantized, level_indices, level_inputs = residual_quantize(params["codebooks"], z)

    z_q = sum(level_quantized)  # full quantized latent = sum of per-level codes

    # straight-through estimator: use z_q on the forward pass, but let gradient flow to the
    # encoder as if quantization were the identity function
    z_q_st = z + jax.lax.stop_gradient(z_q - z)
    x_hat = decode(params, z_q_st)

    recon_loss = jnp.mean((x_hat - x) ** 2)

    codebook_loss = 0.0
    commitment_loss = 0.0
    for q, inp in zip(level_quantized, level_inputs):
        codebook_loss += jnp.mean((q - jax.lax.stop_gradient(inp)) ** 2)
        commitment_loss += jnp.mean((jax.lax.stop_gradient(q) - inp) ** 2)

    loss = recon_loss + codebook_loss + beta * commitment_loss
    indices = jnp.stack(level_indices, axis=-1)  # (batch, n_codebooks)
    return loss, {"recon_loss": recon_loss, "codebook_loss": codebook_loss,
                  "commitment_loss": commitment_loss, "indices": indices}


def get_codes(params, x):
    """Inference-only: encode + quantize, return the (n_items, n_codebooks) code matrix."""
    z = encode(params, x)
    _, level_indices, _ = residual_quantize(params["codebooks"], z)
    return jnp.stack(level_indices, axis=-1)


def make_adam(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """A small hand-rolled Adam optimizer -- avoids pulling in optax as another dependency
    for something this project can implement directly."""

    def init(params):
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        return {"m": m, "v": v, "t": 0}

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
