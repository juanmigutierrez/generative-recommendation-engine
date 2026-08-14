"""
Exports the processed parquet files as .csv siblings, purely so they're easy to open and
eyeball in Excel/a text editor -- parquet is the format the pipeline actually uses (faster,
preserves types), CSV here is just for human inspection.

List-valued columns (the per-user sequences/targets files) get flattened to a single
space-separated string column so they still open sensibly as a spreadsheet.

Run:
    python backend/scripts/export_csv.py
"""
import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

FILES = [
    "interactions.parquet",
    "item_catalog.parquet",
    "user_catalog.parquet",
    "train.parquet",
    "val.parquet",
    "test.parquet",
    "train_sequences.parquet",
    "val_targets.parquet",
    "test_targets.parquet",
]


def flatten_lists(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            df[col] = df[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple)) else x
            )
    return df


def main():
    for fname in FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"skip (not found): {fname}")
            continue
        df = pd.read_parquet(path)
        df = flatten_lists(df)
        csv_path = path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"{fname} -> {os.path.basename(csv_path)}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
