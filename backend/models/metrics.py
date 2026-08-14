"""
Recall@K and NDCG@K -- the two metrics used to score every model in this project (see
docs/02_split_and_eval.md and docs/00_roadmap.md Step 8 for the full rationale).

Both take the same inputs: a list of recommended item_ids (ranked, best first) and the set
of item_ids the user actually interacted with in the held-out period ("relevant" items).
"""
import numpy as np


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Of the items the user actually interacted with, what fraction appear anywhere in the
    top-K recommendations? Doesn't care about order within the top-K, only presence.
    """
    if not relevant:
        return np.nan
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Like recall, but rewards true positives that rank HIGHER in the list more than ones
    buried at position K. A model that gets the right answer at rank 1 scores higher than
    one that gets it at rank 10, even though both would score identically on recall@10.

    DCG@K = sum over ranked list positions i=1..K of (relevance at i) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K, where IDCG@K is the DCG of the best possible ordering
             (all true positives ranked first) -- this normalizes the score to [0, 1]
             regardless of how many relevant items the user actually has.
    """
    if not relevant:
        return np.nan
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)  # +2 because i is 0-indexed and log2(1)=0 would divide by zero
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else np.nan


def evaluate(recommend_fn, targets: dict, k_values=(10, 20)) -> dict:
    """
    Runs recall_at_k / ndcg_at_k for every user in `targets` (user_id -> set of relevant
    item_ids) against whatever `recommend_fn(user_id, k)` returns, and averages across users.

    recommend_fn: callable(user_id, k) -> ranked list of item_ids (already excludes items
                  the user saw in train -- each model's `recommend()` method handles that).
    """
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})

    for user_id, relevant in targets.items():
        if not relevant:
            continue
        max_k = max(k_values)
        recs = recommend_fn(user_id, max_k)
        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(recs, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(recs, relevant, k))

    return {metric: float(np.nanmean(vals)) for metric, vals in results.items()}
