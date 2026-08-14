"""
ALS collaborative filtering baseline, using the `implicit` library
(https://github.com/benfred/implicit), which implements the confidence-weighted implicit-
feedback formulation from Hu, Koren & Volinsky (2008) -- see docs/00_roadmap.md Step 4 and
the animated walkthrough earlier in this build for the math/intuition.
"""
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(self, factors=64, regularization=0.05, iterations=15, alpha=40.0, random_state=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha  # confidence scaling: c_ui = 1 + alpha * r_ui (see docs)
        self.random_state = random_state
        self.model = None
        self.user_items = None  # sparse CSR (n_users x n_items), confidence-weighted
        self.n_users = None

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        self.n_users = n_users
        # collapse repeat interactions into a count per (user, item), then scale by alpha
        # to get the confidence value the paper calls c_ui.
        counts = train.groupby(["user_id", "item_id"]).size().reset_index(name="count")
        user_items = sp.coo_matrix(
            (counts["count"].astype("double") * self.alpha, (counts["user_id"], counts["item_id"])),
            shape=(n_users, n_items),
        ).tocsr()
        self.user_items = user_items

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model.fit(user_items)
        return self

    def is_cold(self, user_id: int) -> bool:
        """True if this user has zero train interactions -- ALS has no learned vector for
        them (ALS can only place a user in the latent space using interactions it saw)."""
        return user_id >= self.n_users or self.user_items[user_id].nnz == 0

    def recommend(self, user_id: int, k: int) -> list:
        if self.is_cold(user_id):
            return []  # caller is expected to fall back to popularity -- see run_baselines.py
        ids, _scores = self.model.recommend(
            user_id, self.user_items[user_id], N=k, filter_already_liked_items=True
        )
        return list(ids)
