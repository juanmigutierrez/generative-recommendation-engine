"""
Step 5a (local variant) -- item content embeddings via a real pretrained sentence-embedding
model, run on your machine (not the sandbox), using PyTorch.

Why PyTorch here, when the RQ-VAE itself (backend/models/rqvae.py) is JAX: two different
constraints, two different answers. The RQ-VAE trains inside the sandbox, where PyTorch's
CPU wheel isn't reachable and the default wheel doesn't fit the disk quota -- JAX solves
that. This script runs on your own machine, which has neither restriction. It was briefly
written in JAX/Flax too, but `transformers` v5 (2025) removed Flax support from the library
entirely (confirmed directly in HuggingFace's own migration guide -- see
https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md), so keeping it
in JAX meant pinning an old, unsupported `transformers` version just for this one script.
Not worth it: PyTorch is the standard, fully-supported path, and only this input-prep step
uses it -- the RQ-VAE itself, which is the part actually worth explaining in an interview,
stays JAX, untouched.

Model: `sentence-t5-base` -- the same family TIGER's paper uses (Sentence-T5, XXL in their
main results; base is the practical CPU-sized version). Loaded via the `sentence-transformers`
library directly, which handles this model's exact pooling/projection setup as its author
defined it, rather than a hand-rolled mean-pooling guess.

Output format matches `build_item_embeddings.py` exactly, so `train_rqvae.py` needs zero
changes -- it reads the embedding dimension dynamically from the saved array's shape.

Setup (run once, on your machine):
    pip install sentence-transformers

Run:
    python backend/scripts/build_item_embeddings_local.py

Then hand data/processed/item_embeddings.npy and item_embeddings_index.parquet back and the
RQ-VAE gets retrained on top of real semantic embeddings instead of TF-IDF/SVD.
"""
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_NAME = "sentence-transformers/sentence-t5-base"  # ~110M params, 768-dim


def main():
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    item_catalog = item_catalog.sort_values("item_id").reset_index(drop=True)
    titles = item_catalog["description"].fillna("").tolist()
    print(f"{len(titles):,} item titles")

    print(f"loading {MODEL_NAME} (downloads on first run, then cached locally)...")
    model = SentenceTransformer(MODEL_NAME)

    print("encoding...")
    embeddings = model.encode(
        titles, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    print(f"embeddings: {embeddings.shape}")

    embeddings = normalize(embeddings, norm="l2", axis=1).astype("float32")

    out_path = os.path.join(PROCESSED_DIR, "item_embeddings.npy")
    np.save(out_path, embeddings)
    item_catalog[["item_id"]].to_parquet(os.path.join(PROCESSED_DIR, "item_embeddings_index.parquet"))
    print(f"saved -> {out_path}  shape={embeddings.shape}")

    # sanity check before trusting this as RQ-VAE input: do nearest neighbors look related?
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.RandomState(0)
    for idx in rng.choice(len(embeddings), 3, replace=False):
        sims = cosine_similarity(embeddings[idx : idx + 1], embeddings)[0]
        top = np.argsort(-sims)[1:6]
        print(f"\n'{titles[idx]}' nearest neighbors:")
        for j in top:
            print(f"  {sims[j]:.3f}  {titles[j]}")


if __name__ == "__main__":
    main()
