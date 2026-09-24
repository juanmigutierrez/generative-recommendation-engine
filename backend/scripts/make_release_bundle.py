"""
Collect the pre-computed artifacts the tutorial notebooks download, so they can be attached
to a GitHub release (data/processed/ is git-ignored on purpose -- it is ~250 MB).

    python backend/scripts/make_release_bundle.py
    gh release create v1.0-artifacts release/* --title "Pre-computed artifacts for the tutorials" \
        --notes "Processed data, embeddings and trained checkpoints used by notebooks/tutorial_0*.ipynb"

The notebooks fetch each file from
    https://github.com/juanmigutierrez/generative-recommendation-engine/releases/download/v1.0-artifacts/<file>
so the tag name must stay `v1.0-artifacts` (or update RELEASE_TAG in notebooks/_build_tutorials.py).
"""
import os
import shutil

FILES = [
    # data
    "interactions.parquet", "item_catalog.parquet", "user_catalog.parquet",
    "train.parquet", "val.parquet", "test.parquet", "train_sequences.parquet",
    "val_targets.parquet", "test_targets.parquet", "split_stats.json",
    "loo_train.parquet", "loo_train_sequences.parquet", "loo_val_sequences.parquet",
    "loo_val_targets.parquet", "loo_test_targets.parquet", "loo_split_stats.json",
    # semantic ids
    "item_embeddings.npy", "item_embeddings_index.parquet", "semantic_ids.parquet",
    "item_tokens.parquet", "rqvae_params.npz", "rqvae_embedding_standardization.npz", "rqvae_metrics.json",
    # generative retrieval (leave-one-out track)
    "transformer_train_sequences_loo.npz", "transformer_vocab_meta_loo.json", "transformer_checkpoint_loo.pkl",
    # sasrec + ranker
    "sasrec_checkpoint.pkl", "ranker_model.pkl", "ranker_train_users.npy", "ranker_train_meta.json",
    # results
    "loo_evaluation_results.json", "full_evaluation_results.json", "ranking_eval_results.json",
]

if __name__ == "__main__":
    src = os.path.join("data", "processed"); dst = "release"
    os.makedirs(dst, exist_ok=True)
    total = 0
    for f in FILES:
        p = os.path.join(src, f)
        if not os.path.exists(p):
            print(f"MISSING  {f}"); continue
        shutil.copy(p, os.path.join(dst, f)); total += os.path.getsize(p)
        print(f"{os.path.getsize(p)/1e6:7.1f} MB  {f}")
    print(f"\n{total/1e6:.0f} MB in {dst}/ -- now run:\n  gh release create v1.0-artifacts {dst}/* --title \"Pre-computed artifacts for the tutorials\"")
