# Dataset

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD)
— category: **Video_Games**, **5-core** benchmark subset (every user has ≥5 interactions,
every item has ≥5 interactions, guaranteed by construction).

## Why this dataset (and why it changed twice)

This project went through three dataset choices, each rejected/accepted for a concrete,
checkable reason -- worth knowing the history since it's a realistic picture of how dataset
selection actually goes:

1. **Amazon Reviews 2023 (planned first)** -- the modern standard recsys research dataset,
   used by the TIGER/HSTU-style papers this project's architecture is based on. Blocked
   initially by network restrictions in the build environment (Hugging Face unreachable);
   worked once downloaded directly via McAuley Lab's file server instead.
2. **UCI Online Retail (used temporarily)** -- real transaction data, used as a placeholder
   while the Amazon download path was being fixed. See git history / `docs/` for its own
   writeup if useful for comparison; it's no longer the primary dataset.
3. **Amazon Reviews 2023, raw reviews (tried, rejected)** -- the raw `All_Beauty` category
   downloaded cleanly, but **93.9% of users had exactly one review ever** (628K reviews
   across 584K users). A sequential recommender needs *history* to predict from; with almost
   no repeat users, there's nothing to learn. This is a real property of review data, not a
   bug: most people review a product once and never review anything else in that category.
4. **Amazon Reviews 2023, 5-core Video_Games (final)** -- McAuley Lab publishes a "5-core"
   filtered version specifically for recsys benchmarking: every user/item is guaranteed
   ≥5 interactions by construction. `Video_Games` was chosen over the default `All_Beauty`
   because `All_Beauty` collapses to only 253 users after 5-core filtering (too small);
   `Video_Games` gives 94,762 users / 25,612 items / 814,586 ratings -- a real, commonly
   benchmarked category in the recsys literature (SASRec, TIGER, and others report on it),
   which also makes this project's results comparable to published numbers.

## Key stats (5-core Video_Games)

- 94,762 users, 25,612 items, 814,586 interactions
- Every user has ≥5 interactions by construction; median 6, mean 8.6 per user
- Item descriptions are real product titles (0 missing/unknown out of 25,612 items)
- **Popularity is long-tailed**: top 10 items are only 2.4% of interactions; the bottom 50%
  of items are 11.5% -- less extreme than a raw retail dataset but still a real long tail,
  still motivating the content-based (Semantic ID) retrieval stage for items with sparse
  interaction history.
- Timestamps span 1999–2023 (review timestamps, not necessarily purchase dates -- see
  caveat below).

## Known caveat: this is review data, not transaction data

Unlike the UCI dataset (real purchase invoices), Amazon Reviews are *reviews*, which
correlate with but aren't identical to purchases (a review can lag the purchase by months;
not every purchase gets reviewed). There's also no real "basket"/session grouping --
`invoice_id` in `interactions.parquet` is a synthetic per-row placeholder, not a real
shopping session. This is disclosed in `backend/scripts/prepare_data_amazon.py`'s docstring
and is a legitimate limitation worth stating plainly in an interview: it's still real user-
item interaction signal at meaningful scale, but "session-based" framing from the original
architecture doc refers to sequences of reviews over time, not baskets.

## Known caveat: missing prices

24.7% of interaction rows have `unit_price = NaN`. Verified this is a genuine source-data
gap, not a processing bug: all 25,612 items have a matching metadata record, but 8,478 of
them (33%) have `price: null` in Amazon's own metadata (delisted products, third-party
sellers, prices not captured at crawl time). Left as NaN rather than imputed at this stage
-- no model consumes price yet. When the ranking stage (Step 7) adds price as a feature,
handle it explicitly there (e.g. category-median imputation + a `has_price` flag), not by
silently filling it in upstream.

## Files

- `data/raw/amazon/Video_Games_5core_ratings.parquet` -- interactions (user_id, parent_asin, rating, timestamp)
- `data/raw/amazon/Video_Games_meta.parquet` -- item metadata (title, price, store, category, etc.)
- `data/processed/interactions.parquet`, `item_catalog.parquet`, `user_catalog.parquet` --
  unified schema, produced by `backend/scripts/prepare_data_amazon.py`

Next: `data/processed/{train,val,test}.parquet` -- see `docs/02_split_and_eval.md`.
