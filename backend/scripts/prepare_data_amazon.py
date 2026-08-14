"""
Converts the downloaded Amazon Reviews 2023 5-core files into the SAME schema that
prepare_data.py produces for the UCI dataset, so every downstream script
(split_data.py, build_sequences.py, baselines, RQ-VAE, ...) works unchanged
regardless of which dataset you're using.

Run after download_amazon_reviews.py:
    python backend/scripts/prepare_data_amazon.py --category Video_Games

Output (same filenames/schema as prepare_data.py):
    data/processed/interactions.parquet   (user_id, item_id, invoice_id, quantity, unit_price, timestamp)
    data/processed/item_catalog.parquet   (item_id, stock_code, description)
    data/processed/user_catalog.parquet   (user_id, customer_id)

Schema differences vs. UCI Online Retail worth knowing about (see docs/01_dataset.md):
  - "quantity" doesn't exist for ratings -- each rating counts as one interaction (quantity=1).
  - "invoice_id" has no real basket/session meaning here (a rating isn't a shopping basket).
    Each rating gets its own synthetic invoice_id, so the "session = basket" framing from
    the UCI dataset doesn't carry over -- session-based modeling on this data means a
    time-ordered sequence of items per user, not baskets.
  - The 5-core file is already filtered (every user/item has >=5 interactions) and
    de-duplicated by McAuley Lab -- no verified-purchase or min-interaction filtering
    needed on our end, unlike the UCI cleaning step.
"""
import argparse
import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw", "amazon")
PROCESSED_DIR = os.path.join("data", "processed")


def build_item_catalog(ratings: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    items = sorted(ratings["parent_asin"].unique())
    catalog = pd.DataFrame({"stock_code": items})
    catalog["item_id"] = range(len(catalog))

    meta_titles = meta.drop_duplicates(subset=["parent_asin"]).set_index("parent_asin")["title"]
    catalog["description"] = catalog["stock_code"].map(meta_titles).fillna("UNKNOWN")
    return catalog[["item_id", "stock_code", "description"]]


def build_user_catalog(ratings: pd.DataFrame) -> pd.DataFrame:
    users = sorted(ratings["user_id"].unique())
    return pd.DataFrame({"user_id": range(len(users)), "customer_id": users})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Video_Games")
    args = parser.parse_args()
    category = args.category

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings_path = os.path.join(RAW_DIR, f"{category}_5core_ratings.parquet")
    meta_path = os.path.join(RAW_DIR, f"{category}_meta.parquet")
    if not (os.path.exists(ratings_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"Missing raw files for '{category}'. Run first:\n"
            f"  python backend/scripts/download_amazon_reviews.py --category {category}"
        )

    ratings = pd.read_parquet(ratings_path)
    meta = pd.read_parquet(meta_path)
    print(f"5-core ratings: {len(ratings):,}")

    ratings = ratings.dropna(subset=["user_id", "parent_asin", "timestamp"])
    ratings = ratings.drop_duplicates(subset=["user_id", "parent_asin", "timestamp"])
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="ms")

    item_catalog = build_item_catalog(ratings, meta)
    user_catalog = build_user_catalog(ratings)
    print(f"items: {len(item_catalog):,}  users: {len(user_catalog):,}")

    item_map = dict(zip(item_catalog["stock_code"], item_catalog["item_id"]))
    user_map = dict(zip(user_catalog["customer_id"], user_catalog["user_id"]))
    price_map = meta.drop_duplicates(subset=["parent_asin"]).set_index("parent_asin")["price"]

    interactions = pd.DataFrame({
        "user_id": ratings["user_id"].map(user_map),
        "item_id": ratings["parent_asin"].map(item_map),
        "invoice_id": [f"r{i}" for i in ratings.index],  # synthetic -- see docstring
        "quantity": 1,
        "unit_price": ratings["parent_asin"].map(price_map),
        "timestamp": ratings["timestamp"],
    }).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    interactions.to_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    item_catalog.to_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    user_catalog.to_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet"))
    print(f"wrote interactions/item_catalog/user_catalog to {PROCESSED_DIR}/")
    print("\nNext step (unchanged from the UCI pipeline):")
    print("  python backend/scripts/split_data.py")
    print("  python backend/scripts/build_sequences.py")


if __name__ == "__main__":
    main()
