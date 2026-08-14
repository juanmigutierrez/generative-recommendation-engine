"""
Popularity baseline: recommend the same globally-most-interacted-with items to everyone,
minus whatever that particular user already saw in train. No personalization at all --
this is the floor every other model in the project has to clear.
"""
import pandas as pd


class PopularityRecommender:
    def __init__(self):
        self.ranked_items = None       # item_ids sorted by train popularity, most popular first
        self.user_seen = None          # user_id -> set of item_ids seen in train

    def fit(self, train: pd.DataFrame):
        self.ranked_items = (
            train.groupby("item_id").size().sort_values(ascending=False).index.tolist()
        )
        self.user_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id: int, k: int) -> list:
        seen = self.user_seen.get(user_id, set())
        recs = []
        for item_id in self.ranked_items:
            if item_id not in seen:
                recs.append(item_id)
                if len(recs) == k:
                    break
        return recs
