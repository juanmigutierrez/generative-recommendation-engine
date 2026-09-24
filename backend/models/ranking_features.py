"""
Step 7: shared feature-building code for the ranking stage.

Retrieval (ALS here, standing in for the Transformer -- see docs/07_ranking.md for why)
gives a wide shortlist of candidates. Ranking's job is to reorder that shortlist using
signals retrieval couldn't afford to compute for the whole 25,612-item catalog:
  - als_score            -- retrieval's own confidence, carried forward as a feature
  - item_popularity_log  -- how often this item is bought overall (log1p of train count)
  - item_price_imputed   -- price, median-imputed where missing (see docs/01_dataset.md
                             "Known caveat: missing prices" -- 24.7% of rows are NaN)
  - has_price             -- 1 if this item had a real observed price, 0 if imputed
  - item_recency_norm    -- how recently, on average, this item has been bought (trending
                             vs. stale), normalized to [0, 1] over the train time span
  - content_sim          -- cosine similarity between the item's Sentence-T5 embedding
                             (Step 5) and the user's profile embedding (mean of their
                             train-history item embeddings) -- ties the ranking stage back
                             to the same content signal the generative retrieval model uses
  - user_n_interactions_log -- log1p of the user's train history length (a rough proxy for
                             how much signal exists to personalize with at all)

Every function here is deliberately vectorized (no python loops over users/items) so
candidate generation + feature assembly for thousands of users stays fast.
"""
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "als_score",
    "item_popularity_log",
    "item_price_imputed",
    "has_price",
    "item_recency_norm",
    "content_sim",
    "user_n_interactions_log",
]


def build_item_features(train: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """One row per item_id (0..n_items-1), static features derived only from train."""
    counts = train.groupby("item_id").size()
    popularity_log = np.log1p(counts.reindex(range(n_items), fill_value=0).values)

    price_by_item = train.groupby("item_id")["unit_price"].median()
    global_median_price = train["unit_price"].median()
    has_price = price_by_item.reindex(range(n_items)).notna().values.astype("float32")
    price_imputed = price_by_item.reindex(range(n_items)).fillna(global_median_price).values

    # recency: mean interaction time per item, normalized to [0, 1] over train's own span
    # (same "percentage of the dataset's own timespan" trick used for the train/val/test
    # split itself -- see docs/02_split_and_eval.md).
    t = train["timestamp"].astype("int64")  # ns since epoch
    t_min, t_max = t.min(), t.max()
    span = max(t_max - t_min, 1)
    mean_t_by_item = t.groupby(train["item_id"]).mean()
    recency_norm = ((mean_t_by_item.reindex(range(n_items)) - t_min) / span).values
    global_recency = ((t.mean() - t_min) / span)
    recency_norm = np.where(np.isnan(recency_norm), global_recency, recency_norm)

    return pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "item_popularity_log": popularity_log.astype("float32"),
            "item_price_imputed": price_imputed.astype("float32"),
            "has_price": has_price,
            "item_recency_norm": recency_norm.astype("float32"),
        }
    ).set_index("item_id")


def build_user_profile_embeddings(train_sequences: pd.DataFrame, item_embeddings: np.ndarray) -> dict:
    """user_id -> mean-pooled, L2-normalized embedding of their train-history items."""
    profiles = {}
    for row in train_sequences.itertuples(index=False):
        vecs = item_embeddings[row.item_ids]
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        profiles[row.user_id] = mean_vec / norm if norm > 0 else mean_vec
    return profiles


def build_user_interaction_counts(train_sequences: pd.DataFrame) -> dict:
    return {row.user_id: len(row.item_ids) for row in train_sequences.itertuples(index=False)}


def assemble_features(
    user_ids: np.ndarray,
    item_ids_per_user: list,
    als_scores_per_user: list,
    item_features: pd.DataFrame,
    user_profiles: dict,
    user_n_interactions: dict,
    item_embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Flattens (user, [candidate items], [als scores]) triples into one long feature table,
    one row per (user_id, item_id) candidate pair.
    """
    n_pairs = sum(len(items) for items in item_ids_per_user)
    out_user = np.empty(n_pairs, dtype="int64")
    out_item = np.empty(n_pairs, dtype="int64")
    out_als = np.empty(n_pairs, dtype="float32")
    out_content_sim = np.empty(n_pairs, dtype="float32")

    pos = 0
    default_profile = np.zeros(item_embeddings.shape[1], dtype="float32")
    for u, items, scores in zip(user_ids, item_ids_per_user, als_scores_per_user):
        n = len(items)
        if n == 0:
            continue
        out_user[pos:pos + n] = u
        out_item[pos:pos + n] = items
        out_als[pos:pos + n] = scores
        profile = user_profiles.get(u, default_profile)
        item_vecs = item_embeddings[items]
        item_norms = np.linalg.norm(item_vecs, axis=1)
        sims = (item_vecs @ profile) / np.where(item_norms > 0, item_norms, 1.0)
        out_content_sim[pos:pos + n] = sims
        pos += n

    out_user = out_user[:pos]
    out_item = out_item[:pos]
    out_als = out_als[:pos]
    out_content_sim = out_content_sim[:pos]

    df = pd.DataFrame({"user_id": out_user, "item_id": out_item, "als_score": out_als, "content_sim": out_content_sim})
    df = df.merge(item_features, on="item_id", how="left")
    df["user_n_interactions_log"] = df["user_id"].map(user_n_interactions).fillna(0).apply(lambda x: np.log1p(x)).astype("float32")
    return df
