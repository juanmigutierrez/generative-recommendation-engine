"""
Recall@K, NDCG@K, and MAP@K -- the metrics used to score every model in this project (see
docs/02_split_and_eval.md and docs/08_evaluation.md for the full rationale).

All three take the same inputs: a list of recommended item_ids (ranked, best first) and the
set of item_ids the user actually interacted with in the held-out period ("relevant" items).
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


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    MAP@K's per-user term. Precision@K and Recall@K each ignore *where* inside the top-K a
    hit lands (a hit at rank 1 counts the same as a hit at rank 10); NDCG@K cares about rank
    but discounts smoothly (log2). Average Precision cares about rank differently: it's the
    mean of precision@i evaluated only at the ranks where a hit actually occurs, so getting
    several relevant items *early* compounds -- each early hit raises the precision every
    later hit gets credited at, too.

    AP@K = (1 / min(|relevant|, K)) * sum_{i=1..K} [ precision@i * rel(i) ]
    where rel(i) = 1 if the item at rank i (1-indexed) is relevant, else 0, and precision@i
    is (# relevant items in the top i) / i. Dividing by min(|relevant|, K) rather than the
    number of hits normalizes to [0, 1]: a perfect ranking (all relevant items first) scores
    1.0 regardless of how many relevant items the user has.
    """
    if not relevant:
        return np.nan
    top_k = recommended[:k]
    hits = 0
    precisions = []
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            precisions.append(hits / i)
    denom = min(len(relevant), k)
    return sum(precisions) / denom if denom > 0 else np.nan


def evaluate(recommend_fn, targets: dict, k_values=(10, 20)) -> dict:
    """
    Runs recall_at_k / ndcg_at_k / average_precision_at_k for every user in `targets`
    (user_id -> set of relevant item_ids) against whatever `recommend_fn(user_id, k)`
    returns, and averages across users.

    recommend_fn: callable(user_id, k) -> ranked list of item_ids (already excludes items
                  the user saw in train -- each model's `recommend()` method handles that).
    """
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})
    results.update({f"map@{k}": [] for k in k_values})

    for user_id, relevant in targets.items():
        if not relevant:
            continue
        max_k = max(k_values)
        recs = recommend_fn(user_id, max_k)
        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(recs, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(recs, relevant, k))
            results[f"map@{k}"].append(average_precision_at_k(recs, relevant, k))

    return {metric: float(np.nanmean(vals)) for metric, vals in results.items()}
