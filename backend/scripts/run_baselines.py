"""
Step 4: train + evaluate the two baseline models (popularity, ALS) on the same val/test
sets, and print/save a comparison table. This table is the floor every later stage
(Steps 5-7) has to beat -- see docs/00_roadmap.md.

Run:
    python backend/scripts/run_baselines.py
"""
import json
import os
import sys

# Must be set before numpy/implicit import. `implicit`'s ALS uses its own internal
# threading (Cython/OpenMP); if numpy's BLAS backend also tries to multithread the same
# matrix ops, they fight over cores and training can silently take 10-50x longer. This is
# a known gotcha with the `implicit` library, not specific to this project.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.popularity import PopularityRecommender
from models.als import ALSRecommender
from models.metrics import evaluate

PROCESSED_DIR = os.path.join("data", "processed")
K_VALUES = (10, 20)


def load_targets(path: str) -> dict:
    df = pd.read_parquet(path)
    return {row.user_id: set(row.item_ids) for row in df.itertuples()}


def main():
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    user_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet"))
    item_catalog = pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    n_users = len(user_catalog)
    n_items = len(item_catalog)

    val_targets = load_targets(os.path.join(PROCESSED_DIR, "val_targets.parquet"))
    test_targets = load_targets(os.path.join(PROCESSED_DIR, "test_targets.parquet"))

    print(f"train: {len(train):,} rows, {n_users:,} users, {n_items:,} items")

    print("\nfitting popularity...")
    pop_model = PopularityRecommender().fit(train)

    print("fitting ALS...")
    als_model = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(
        train, n_users, n_items
    )

    def als_with_fallback(user_id: int, k: int) -> list:
        """ALS has no vector for cold-start users (never seen in train) -- fall back to
        popularity for those, rather than returning nothing. This hybrid fallback is a
        real, common production pattern, not just an evaluation-script convenience."""
        if als_model.is_cold(user_id):
            return pop_model.recommend(user_id, k)
        return als_model.recommend(user_id, k)

    results = {}
    for split_name, targets in [("val", val_targets), ("test", test_targets)]:
        print(f"\nevaluating on {split_name}...")
        results[split_name] = {
            "popularity": evaluate(pop_model.recommend, targets, K_VALUES),
            "als": evaluate(als_with_fallback, targets, K_VALUES),
        }

    print("\n" + "=" * 60)
    print(f"{'split':<6}{'model':<12}" + "".join(f"{m:<12}" for m in results["val"]["popularity"]))
    print("-" * 60)
    for split_name in results:
        for model_name, metrics in results[split_name].items():
            row = f"{split_name:<6}{model_name:<12}"
            row += "".join(f"{v:<12.4f}" for v in metrics.values())
            print(row)
    print("=" * 60)

    out_path = os.path.join(PROCESSED_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
