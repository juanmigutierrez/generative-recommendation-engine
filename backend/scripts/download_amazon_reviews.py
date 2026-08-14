"""
Downloads the "5-core" benchmark-ready subset of Amazon Reviews 2023 (McAuley Lab, UCSD):
every user has >=5 interactions and every item has >=5 interactions within the category.
This is the standard preprocessing academic recsys papers (SASRec, TIGER, HSTU-style models)
use when benchmarking on Amazon data.

Why not raw reviews (what an earlier version of this script downloaded): in the raw
All_Beauty category, 93.9% of users have exactly ONE review ever -- there's no "history" to
learn from for the vast majority of users, which breaks sequential/session-based modeling at
the root (628K reviews spread across 584K nearly-all-one-time users). 5-core filtering
removes that problem by construction: every user in the output has enough repeat
interactions to have a real, learnable sequence.

Setup:
    pip install pandas pyarrow

Usage:
    python backend/scripts/download_amazon_reviews.py --category Video_Games
    python backend/scripts/download_amazon_reviews.py --category Software

Category sizing after 5-core filtering (users / items / ratings):
    Video_Games               94.8K / 25.6K  / 814.6K  (default -- classic recsys benchmark category)
    Software                 146.4K / 17.6K  / 1.3M
    Office_Products          223.3K / 77.6K  / 1.8M
    Toys_and_Games           432.3K / 162.0K / 3.9M
    Grocery_and_Gourmet_Food 404.8K / 132.9K / 3.9M   (repeat-purchase heavy -- great sequential signal)
    Beauty_and_Personal_Care 729.6K / 207.6K / 6.6M   (large -- scale up later if you want)
    (All_Beauty -- what the old raw-review download used -- is only 253 users / 356 items
     after 5-core filtering. Too small on top of the one-review-per-user problem above.)

Full list + sizes: https://amazon-reviews-2023.github.io/data_processing/5core.html

Output:
    data/raw/amazon/{category}_5core_ratings.parquet   (user_id, parent_asin, rating, timestamp)
    data/raw/amazon/{category}_meta.parquet            (one row per product)
"""
import argparse
import gzip
import json
import os
import re
import ssl
import urllib.request

import pandas as pd


def _parse_price(x):
    """
    Meta 'price' is inconsistently typed across rows -- usually a plain number, but
    sometimes text like "from 14.99" (variable/bundle pricing) or null. pyarrow refuses to
    write a column that mixes floats and strings, so pull the first number out of whatever
    we got and treat anything unparseable as missing rather than erroring the whole write.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    match = re.search(r"[\d,]+\.?\d*", str(x))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
RAW_DIR = os.path.join("data", "raw", "amazon")

META_COLUMNS = ["parent_asin", "title", "main_category", "average_rating",
                "rating_number", "price", "store", "categories", "features", "description"]


def _build_opener():
    """
    Windows Python (especially the python.org installer) sometimes ships without the root CA
    that mcauleylab.ucsd.edu's certificate chains to, causing CERTIFICATE_VERIFY_FAILED even
    though the site is fine. Prefer certifi's CA bundle; fall back to unverified if needed so
    the download isn't blocked -- this is a public academic dataset file, not sensitive.
    """
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        print("  (certifi not available/working -- falling back to an unverified "
              "SSL context. `pip install certifi` to avoid this.)")
        ctx = ssl._create_unverified_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


_OPENER = _build_opener()


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  already downloaded: {dest}")
        return
    print(f"  downloading {url}")

    tmp_dest = dest + ".part"
    try:
        resp = _OPENER.open(url, timeout=60)
    except ssl.SSLCertVerificationError:
        print("  SSL verification still failing -- retrying without verification...")
        ctx = ssl._create_unverified_context()
        resp = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)).open(url, timeout=60)

    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024
    with resp, open(tmp_dest, "wb") as out:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB)", end="")
    print()
    os.replace(tmp_dest, dest)


def jsonl_gz_to_df(path: str, columns: list) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append({c: record.get(c) for c in columns})
    df = pd.DataFrame(rows)
    for col in ["categories", "features", "description"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: " ".join(x) if isinstance(x, (list, tuple)) else (x or "")
            )
    if "price" in df.columns:
        df["price"] = df["price"].apply(_parse_price)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", default="Video_Games",
        help="Amazon Reviews 2023 category name, e.g. Video_Games, Software, Toys_and_Games.",
    )
    args = parser.parse_args()
    category = args.category
    os.makedirs(RAW_DIR, exist_ok=True)

    ratings_gz = os.path.join(RAW_DIR, f"{category}_5core.csv.gz")
    meta_gz = os.path.join(RAW_DIR, f"meta_{category}.jsonl.gz")

    print(f"5-core ratings for category '{category}':")
    download(f"{BASE_URL}/benchmark/5core/rating_only/{category}.csv.gz", ratings_gz)
    print("  parsing...")
    with gzip.open(ratings_gz, "rt", encoding="utf-8") as f:
        ratings_df = pd.read_csv(f)
    ratings_path = os.path.join(RAW_DIR, f"{category}_5core_ratings.parquet")
    ratings_df.to_parquet(ratings_path)
    print(f"  {len(ratings_df):,} ratings, {ratings_df['user_id'].nunique():,} users, "
          f"{ratings_df['parent_asin'].nunique():,} items -> {ratings_path}")

    print(f"Item metadata for category '{category}':")
    download(f"{BASE_URL}/raw/meta_categories/meta_{category}.jsonl.gz", meta_gz)
    print("  parsing...")
    meta_df = jsonl_gz_to_df(meta_gz, META_COLUMNS)
    meta_path = os.path.join(RAW_DIR, f"{category}_meta.parquet")
    meta_df.to_parquet(meta_path)
    print(f"  {len(meta_df):,} products -> {meta_path}")

    print("\nDone. Next step:")
    print(f"  python backend/scripts/prepare_data_amazon.py --category {category}")


if __name__ == "__main__":
    main()
