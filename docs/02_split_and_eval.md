# Splitting & evaluation methodology

## Why time-based, not random

A random 80/10/10 split lets a model train on a 2022 interaction and get evaluated on one
from 2005 -- it has effectively seen the future. Every real recommender predicts *forward*
in time from what it currently knows, so evaluation has to mirror that or the offline
metrics won't predict real performance. This is one of the most common mistakes in recsys
portfolio projects, and calling it out explicitly is worth doing in an interview.

## Cutoffs (dataset-agnostic by design)

`backend/scripts/split_data.py` computes cutoffs as a **fraction of the dataset's own
observed time span**, not hardcoded dates -- the same script produced a sensible split for
UCI Online Retail (spans ~1 year) and now for Amazon Reviews (spans 1999–2023) with zero
changes. Last 8% of the time span -> test, the 8% before that -> val, everything earlier ->
train.

Current split (5-core Video_Games):

| Split | Range | Rows | Users |
|---|---|---|---|
| Train | before 2019-11-08 | 645,982 | 85,757 |
| Val | 2019-11-08 → 2021-10-05 | 96,009 | 32,852 |
| Test | 2021-10-05 → 2023-09-02 | 72,595 | 22,722 |

Note the val/test windows are ~2 years each here (vs. 4 weeks for UCI) -- because the
dataset spans 24 years instead of 1, and the split is proportional to time span, not row
count. This is worth being able to explain: it's a deliberate tradeoff (consistent
methodology across datasets) vs. hand-tuning window size per dataset for "realism."

## Cold-start counts (why the architecture has a content-based stage)

| | Val | Test |
|---|---|---|
| Cold users (never in train) | 6,014 / 32,852 (18%) | 6,476 / 22,722 (28%) |
| Cold items (never in train) | 2,877 | 4,884 |

A pure collaborative-filtering model (ALS, matrix factorization) has *no signal at all* for
these rows -- it can only fall back to popularity. Roughly a fifth to a quarter of
evaluation users are cold-start, which is large enough that the baseline-vs-hybrid
comparison in the eval report (Step 8) will show a real, measurable gap. This is the direct
motivation for the RQ-VAE / Semantic ID stage: it derives item representations from
*content* (title/description text), so a cold item with zero interactions still gets a
usable embedding at recommendation time.

## Sequence stats (for the sequential/generative retrieval model)

Per-user chronological item sequences built from train: median length 6, mean 7.5, max 473
(a long tail of highly active reviewers). Every user has ≥5 interactions by construction (a
property of the 5-core filtering), so unlike a raw interaction log, there's no "1-interaction
users are most of the dataset" problem here -- see `docs/01_dataset.md` for why that
mattered and how it ruled out the raw (non-5-core) version of this same dataset.

## Metrics (implemented in Step 8)

- **Recall@K** -- of the items a user actually interacted with in val/test, what fraction
  appear in the top-K recommendations.
- **NDCG@K** -- like Recall@K but rewards ranking true positives higher up the list.
- Reported per model tier (popularity → ALS → generative retrieval → +ranking) so the
  incremental value of each added stage of complexity is measured, not assumed.
