"""
Step 5b: train the RQ-VAE on item content embeddings, then assign every item a Semantic ID.

Run:
    python backend/scripts/train_rqvae.py
"""
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import rqvae

PROCESSED_DIR = os.path.join("data", "processed")

HIDDEN_DIM = 256
LATENT_DIM = 32
N_CODEBOOKS = 3
CODEBOOK_SIZE = 256
EPOCHS = 400
BATCH_SIZE = 1024
LR = 2e-3
BETA = 0.25
SEED = 42


def main():
    embeddings = np.load(os.path.join(PROCESSED_DIR, "item_embeddings.npy"))
    index_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_embeddings_index.parquet"))
    item_ids = index_df["item_id"].to_numpy()
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet")).set_index("item_id")

    n_items, in_dim = embeddings.shape
    print(f"{n_items:,} items, {in_dim}-dim content embeddings")

    # Standardize (center + per-dimension unit std) before the encoder ever sees this data.
    # Necessary for real pretrained sentence embeddings (Sentence-T5, BERT-family in general):
    # they're anisotropic -- nearly all of the vector's magnitude sits in one direction shared
    # by every item (measured directly: random-pair cosine similarity was 0.76 on the raw
    # embeddings here, essentially indistinguishable, vs. ~0 after standardizing). Skipping
    # this step is what caused a total codebook collapse (every item mapped to the same code)
    # the first time this was trained on real embeddings -- see docs/05_semantic_ids.md.
    emb_mean = embeddings.mean(axis=0, keepdims=True)
    emb_std = embeddings.std(axis=0, keepdims=True)
    embeddings_std = ((embeddings - emb_mean) / (emb_std + 1e-8)).astype("float32")

    x = jnp.array(embeddings_std)
    key = jax.random.PRNGKey(SEED)
    init_key, codebook_key = jax.random.split(key)
    params = rqvae.init_params(init_key, in_dim, HIDDEN_DIM, LATENT_DIM, N_CODEBOOKS, CODEBOOK_SIZE)

    # Data-dependent codebook init (k-means on the freshly-initialized encoder's own outputs)
    # instead of small-random-noise init -- see rqvae.kmeans_init_codebooks for why.
    z0 = rqvae.encode(params, x)
    params["codebooks"] = rqvae.kmeans_init_codebooks(codebook_key, z0, N_CODEBOOKS, CODEBOOK_SIZE)

    opt_init, opt_update = rqvae.make_adam(lr=LR)
    opt_state = opt_init(params)

    loss_and_grad = jax.jit(jax.value_and_grad(lambda p, batch: rqvae.forward(p, batch, beta=BETA)[0]))

    n_batches = max(1, n_items // BATCH_SIZE)
    history = []
    rng = np.random.RandomState(SEED)

    for epoch in range(EPOCHS):
        perm = rng.permutation(n_items)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            batch = x[idx]
            loss, grads = loss_and_grad(params, batch)
            params, opt_state = opt_update(grads, opt_state, params)
            epoch_loss += float(loss)
        epoch_loss /= n_batches
        history.append(epoch_loss)
        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:4d}  loss {epoch_loss:.5f}")

    # ---- assign Semantic IDs to every item (full pass, no batching needed at this scale) ----
    codes = np.array(rqvae.get_codes(params, x))  # (n_items, N_CODEBOOKS)

    # collisions: how many items share the exact same (level-1, level-2, level-3) tuple
    tuples = [tuple(row) for row in codes]
    from collections import Counter

    counts = Counter(tuples)
    n_colliding_items = sum(c for c in counts.values() if c > 1)
    n_unique_codes = len(counts)
    print(f"\n{n_unique_codes:,} unique {N_CODEBOOKS}-level code tuples for {n_items:,} items")
    print(f"{n_colliding_items:,} items ({100*n_colliding_items/n_items:.1f}%) share a tuple with at least one other item")

    # TIGER's fix: append a 4th "disambiguation" digit -- just a running counter per
    # colliding tuple -- so every item's full Semantic ID is unique, even though the first
    # 3 (content-derived) digits can repeat.
    seen = {}
    dedup_digit = np.zeros(n_items, dtype=np.int32)
    for i, t in enumerate(tuples):
        dedup_digit[i] = seen.get(t, 0)
        seen[t] = seen.get(t, 0) + 1

    semantic_ids = np.concatenate([codes, dedup_digit[:, None]], axis=1)  # (n_items, N_CODEBOOKS+1)
    full_tuples = [tuple(row) for row in semantic_ids]
    assert len(set(full_tuples)) == n_items, "dedup digit failed to make IDs unique"
    print(f"after adding the dedup digit: all {n_items:,} Semantic IDs are unique")

    # ---- codebook utilization: how many of the K codes per level actually get used ----
    utilization = []
    for l in range(N_CODEBOOKS):
        used = len(set(codes[:, l].tolist()))
        utilization.append(used / CODEBOOK_SIZE)
        print(f"level {l+1}: {used}/{CODEBOOK_SIZE} codes used ({100*used/CODEBOOK_SIZE:.0f}%)")

    # ---- save outputs ----
    out_df = pd.DataFrame(
        semantic_ids, columns=[f"sid_level_{l+1}" for l in range(N_CODEBOOKS)] + ["sid_dedup"]
    )
    out_df.insert(0, "item_id", item_ids)
    out_df.to_parquet(os.path.join(PROCESSED_DIR, "semantic_ids.parquet"))
    print(f"\nsaved -> {PROCESSED_DIR}/semantic_ids.parquet")

    np.savez(
        os.path.join(PROCESSED_DIR, "rqvae_params.npz"),
        **{k: np.array(v) for k, v in params.items() if k != "codebooks"},
        codebooks=np.array(params["codebooks"]),
    )

    # standardization stats, needed to apply the same transform to any new item embedded later
    np.savez(
        os.path.join(PROCESSED_DIR, "rqvae_embedding_standardization.npz"),
        mean=emb_mean,
        std=emb_std,
    )

    metrics = {
        "n_items": n_items,
        "in_dim": in_dim,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "n_codebooks": N_CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE,
        "epochs": EPOCHS,
        "final_loss": history[-1],
        "n_unique_3digit_codes": n_unique_codes,
        "pct_items_colliding_before_dedup": 100 * n_colliding_items / n_items,
        "codebook_utilization": utilization,
        "embedding_source": "sentence-transformers/sentence-t5-base (via Colab GPU, see build_item_embeddings_local.py)",
        "input_standardized": True,
    }
    with open(os.path.join(PROCESSED_DIR, "rqvae_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {PROCESSED_DIR}/rqvae_metrics.json")

    # ---- plot: training curve ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, color="#4F46E5")
    ax.set_title("RQ-VAE training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (reconstruction + codebook + commitment)")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, "rqvae_training_curve.png"), dpi=110)
    print(f"saved -> {PROCESSED_DIR}/rqvae_training_curve.png")

    # ---- qualitative check: do items sharing the first code look related? ----
    print("\n--- qualitative check: 5 items sharing the same level-1 code ---")
    level1 = codes[:, 0]
    sample_code = pd.Series(level1).value_counts().index[0]
    members = np.where(level1 == sample_code)[0][:5]
    for i in members:
        print(f"  [{tuple(codes[i])}]", item_catalog.loc[item_ids[i], "description"])


if __name__ == "__main__":
    main()
