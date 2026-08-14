"""
Step 5a: turn each item's title into a dense content embedding.

Why this exists: the RQ-VAE (train_rqvae.py) needs a continuous vector per item to
compress into a Semantic ID. The roadmap (docs/00_roadmap.md) originally called for
sentence-transformers embeddings, but that library needs PyTorch, and PyTorch's CPU-only
wheel is only published on download.pytorch.org -- which this sandbox can't reach (its
network is allowlisted to a small set of domains). The default PyPI `torch` wheel pulls in
several GB of CUDA dependencies that don't fit the sandbox's disk quota either.

So: TF-IDF + TruncatedSVD (classic LSA) instead. It's a legitimate, well-understood content
embedding method (pre-dates transformers but is still widely used), needs nothing beyond
scikit-learn, and for short e-commerce titles ("Skylanders: Spyro's Adventure - Xbox 360")
the word-overlap signal TF-IDF captures is most of what a sentence embedding would give you
anyway. The RQ-VAE code downstream doesn't care how the input vector was produced -- swap in
a real sentence-transformer later and nothing else changes.

Run:
    python backend/scripts/build_item_embeddings.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROCESSED_DIR = os.path.join("data", "processed")
EMBED_DIM = 128


def main():
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    item_catalog = item_catalog.sort_values("item_id").reset_index(drop=True)
    titles = item_catalog["description"].fillna("").tolist()
    print(f"{len(titles):,} item titles")

    tfidf = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),  # unigrams + bigrams -- "grand theft" means more than "grand" or "theft" alone
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(titles)
    print(f"tfidf matrix: {tfidf_matrix.shape}")

    svd = TruncatedSVD(n_components=EMBED_DIM, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    explained = svd.explained_variance_ratio_.sum()
    print(f"SVD -> {EMBED_DIM} dims, explained variance: {explained:.3f}")

    embeddings = normalize(embeddings, norm="l2", axis=1).astype("float32")

    out_path = os.path.join(PROCESSED_DIR, "item_embeddings.npy")
    np.save(out_path, embeddings)
    item_catalog[["item_id"]].to_parquet(os.path.join(PROCESSED_DIR, "item_embeddings_index.parquet"))
    print(f"saved -> {out_path}  shape={embeddings.shape}")

    # sanity check: nearest neighbors of a random item, by title, should look related
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
