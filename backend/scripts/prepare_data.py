"""
Step 1 of the pipeline: raw file -> cleaned, ID-encoded interactions + item catalog.

Input:  data/raw/online_retail.csv (or .xlsx as fallback)
Output: data/processed/interactions.parquet   (user_id, item_id, invoice_id, quantity, unit_price, timestamp)
        data/processed/item_catalog.parquet   (item_id, stock_code, description)
        data/processed/user_catalog.parquet   (user_id, customer_id)

Cleaning rules (see docs/01_dataset.md for rationale):
  - drop rows with missing CustomerID (guest checkouts -> can't personalize)
  - drop cancellations (InvoiceNo starting with "C")
  - drop non-positive Quantity (returns / data artifacts)

Run:
    python backend/scripts/prepare_data.py
"""
import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def load_raw() -> pd.DataFrame:
    csv_path = os.path.join(RAW_DIR, "online_retail.csv")
    xlsx_path = os.path.join(RAW_DIR, "online_retail.xlsx")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(
            f"No raw data found in {RAW_DIR}. Expected online_retail.csv or .xlsx "
            "(UCI Online Retail dataset)."
        )
    df.columns = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[df["CustomerID"].notnull()]
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def build_item_catalog(df: pd.DataFrame) -> pd.DataFrame:
    # a StockCode can have several slightly different Description strings
    # (typos, case) across rows -- take the most frequent one as canonical.
    canonical_desc = (
        df.groupby("StockCode")["Description"]
        .agg(lambda s: s.value_counts().idxmax() if s.notnull().any() else "UNKNOWN")
        .reset_index()
    )
    canonical_desc = canonical_desc.sort_values("StockCode").reset_index(drop=True)
    canonical_desc["item_id"] = canonical_desc.index
    return canonical_desc.rename(columns={"StockCode": "stock_code", "Description": "description"})[
        ["item_id", "stock_code", "description"]
    ]


def build_user_catalog(df: pd.DataFrame) -> pd.DataFrame:
    customers = sorted(df["CustomerID"].unique())
    return pd.DataFrame({"user_id": range(len(customers)), "customer_id": customers})


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw = load_raw()
    print(f"raw rows: {len(raw):,}")

    df = clean(raw)
    print(f"clean rows: {len(df):,} "
          f"({len(df) / len(raw):.1%} kept)")

    item_catalog = build_item_catalog(df)
    user_catalog = build_user_catalog(df)
    print(f"items: {len(item_catalog):,}  users: {len(user_catalog):,}")

    item_map = dict(zip(item_catalog["stock_code"], item_catalog["item_id"]))
    user_map = dict(zip(user_catalog["customer_id"], user_catalog["user_id"]))

    interactions = pd.DataFrame({
        "user_id": df["CustomerID"].map(user_map),
        "item_id": df["StockCode"].map(item_map),
        "invoice_id": df["InvoiceNo"],
        "quantity": df["Quantity"],
        "unit_price": df["UnitPrice"],
        "timestamp": df["InvoiceDate"],
    }).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    interactions.to_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    item_catalog.to_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet"))
    user_catalog.to_parquet(os.path.join(PROCESSED_DIR, "user_catalog.parquet"))
    print(f"wrote interactions/item_catalog/user_catalog to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
