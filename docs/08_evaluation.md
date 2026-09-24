# Step 8 — Unified Offline Evaluation Framework

## Why this step exists

Every model tier built so far (Steps 4, 6, 7) was evaluated in its own script, with its own
scope decisions -- all defensible individually, but not directly comparable to each other:

- Step 4's baselines (`run_baselines.py`) evaluated on **all** val/test users, with ALS
  falling back to popularity for cold-start users.
- Step 6's Transformer (`evaluate_retrieval.py`) evaluated on a 2,000-user sample, but
  **excluded cold-start users entirely** -- no train history means no beam-search context.
- Step 7's ranker (`evaluate_ranking.py`) evaluated on a 5,000-user sample, also
  **warm-users-only** -- no ALS vector means no candidates to rerank.

Both docs said so explicitly at the time (`docs/06_generative_retrieval.md`: "these numbers
aren't perfectly apples-to-apples with the baseline table"). Step 8 is the fix: one script,
`backend/scripts/run_full_evaluation.py`, that evaluates all four tiers on the *same* user
sample per split, with the *same* fallback rule for cold-start users everywhere, so the
resulting table is an honest, single comparison -- the actual evidence for "what did each
layer of complexity buy," which is the whole point of building this project stage by stage.

## The fallback rule, applied consistently

Every tier now falls back to popularity for users it has no real way to personalize for:

| tier | has no signal for... | falls back to popularity when |
|---|---|---|
| popularity | (never — it's the floor) | — |
| ALS | users with zero train interactions | `als.is_cold(user)` |
| transformer | users with zero train interactions (no beam-search context) | user not in `train_sequences` |
| ALS + ranking | users ALS has no vector for (candidates come from ALS) | `als.is_cold(user)` |

This matters because cold-start users are a large, real fraction of the data: **18.9% of
the val sample, 29.3% of the test sample** (measured directly on the actual 2,000-user
samples drawn below, consistent with the 18-28% figures from `docs/02_split_and_eval.md`).
Excluding them, like Steps 6 and 7 originally did, silently drops the exact population the
Semantic ID work was built to help — the comparison has to include them to mean anything.

## Sample: 2,000 users per split, drawn from everyone

Beam search is a forward pass per generation step per user, so full val/test evaluation
(tens of thousands of users) isn't tractable here -- the same constraint Step 6 hit.
Consistent with that precedent, this evaluates a fixed random sample of 2,000 users per
split. The one change from Step 6's version: the sample is drawn from **all** val/test
users, not just warm ones, so its cold-start rate matches the real population rather than
excluding it by construction.

## A third metric: MAP@K

Added to `backend/models/metrics.py` alongside Recall@K and NDCG@K. Recall@K only checks
presence in the top-K; NDCG@K rewards higher-ranked hits but discounts smoothly. MAP@K
(Mean Average Precision) rewards a specific pattern NDCG doesn't emphasize as strongly:
getting *several* relevant items early, since each early hit raises the precision every
later hit in the list gets credited at too. Standard IR metric, useful third angle on the
same ranked lists.

## Results

| split | model | recall@10 | recall@20 | ndcg@10 | ndcg@20 | map@10 | map@20 |
|---|---|---|---|---|---|---|---|
| val | popularity | 0.0181 | 0.0214 | 0.0144 | 0.0155 | 0.0084 | 0.0086 |
| val | als | 0.0141 | 0.0222 | 0.0112 | 0.0137 | 0.0062 | 0.0068 |
| val | **transformer** | **0.0184** | 0.0235 | 0.0143 | **0.0159** | **0.0086** | **0.0090** |
| val | als + ranking | 0.0157 | **0.0236** | 0.0122 | 0.0146 | 0.0065 | 0.0070 |
| test | popularity | 0.0075 | 0.0083 | 0.0042 | 0.0045 | 0.0021 | 0.0021 |
| test | als | 0.0074 | 0.0099 | 0.0046 | 0.0054 | 0.0024 | 0.0026 |
| test | transformer | 0.0067 | 0.0090 | 0.0044 | 0.0051 | 0.0023 | 0.0024 |
| test | **als + ranking** | **0.0082** | **0.0113** | **0.0053** | **0.0063** | **0.0033** | **0.0035** |

![All four tiers, val and test, same sample, same fallback rule](../data/processed/full_evaluation_comparison.png)

## Reading this honestly

These numbers are lower and shuffled compared to each stage's own isolated evaluation --
that's the fallback rule and the shared sample working as intended, not a regression. Once
every tier has to answer for the same ~19-29% of genuinely cold users (previously scored
100% correctly-excluded or 100% ALS-only-fallback depending on the script), rankings that
looked clean in isolation get noisier. That's what an honest apples-to-apples comparison is
supposed to do: it's a harder table to read, but it's the real one.

Four things worth stating plainly rather than picking whichever framing flatters most:

1. **Popularity is a genuinely strong floor here**, not a strawman -- it beats ALS outright
   on val, and is competitive with everything on test. With a median of 6 interactions per
   user, there just isn't much personalization signal to work with, and popularity captures
   the single strongest available pattern (what's broadly liked) without needing any of it.
2. **The transformer's edge is on val, not test** -- consistent with what Step 6's own doc
   already flagged (10 epochs, loss still descending, a real undertraining explanation
   worth more compute to resolve rather than something to paper over).
3. **ALS + ranking is the most consistent performer on test** -- winning every metric there,
   though behind the transformer and roughly tied with popularity on val. This tracks with
   Step 7's own finding: the ranker's held-out-val result was mixed, but its test result was
   a clean win across the board.
4. **No single tier dominates every metric on every split.** That's the actual finding, and
   it's worth saying directly instead of manufacturing a winner: this is what makes a
   two-stage architecture with several genuinely different mechanisms (co-occurrence,
   content-derived generation, learned reranking) worth having in the first place -- each
   catches cases the others miss, and a production system would likely want to blend or
   A/B test rather than pick one permanently from an offline table this size.

## What's next

Step 9 wraps the trained models behind a FastAPI service (`/recommendations/{user_id}` and
similar), the piece that turns "models that produce a metrics table" into something that
looks and behaves like a real system.

## Outputs

- `backend/models/metrics.py` -- now includes `average_precision_at_k` / MAP@K.
- `backend/scripts/run_full_evaluation.py` -- the unified evaluation script (run per split:
  `--split val`, `--split test`, or `--split both`).
- `data/processed/full_evaluation_results.json` -- all four tiers x both splits x six
  metrics, plus sample sizes and cold-start counts.
- `data/processed/full_evaluation_comparison.png` -- results plot.

## Update (Step 10): re-run after the ranker fix

The `als + ranking` rows above were produced by a ranker that had learned an artifact of
its own training data (see `docs/09_audit_sources_and_results.md`, bug #1) and, on val,
were partly scored on its own training users (bug #2). After retraining without positive
injection and excluding the 18,000 ranker-training users from the val pool (so the val
sample is 41.6% cold, higher than before — those users are disproportionately the ones
left), the same script gives:

| split | model | recall@10 | recall@20 | ndcg@10 | ndcg@20 | map@10 | map@20 |
|---|---|---|---|---|---|---|---|
| val | popularity | 0.0169 | 0.0187 | 0.0139 | 0.0146 | 0.0082 | 0.0083 |
| val | als | 0.0140 | 0.0210 | 0.0111 | 0.0134 | 0.0054 | 0.0059 |
| val | transformer | 0.0178 | 0.0208 | 0.0134 | 0.0146 | 0.0075 | 0.0077 |
| val | **als + ranking** | **0.0229** | **0.0313** | **0.0185** | **0.0212** | **0.0111** | **0.0117** |
| test | popularity | 0.0075 | 0.0083 | 0.0042 | 0.0045 | 0.0021 | 0.0021 |
| test | als | 0.0074 | 0.0099 | 0.0046 | 0.0054 | 0.0024 | 0.0026 |
| test | transformer | 0.0067 | 0.0090 | 0.0044 | 0.0051 | 0.0023 | 0.0024 |
| test | **als + ranking** | **0.0114** | **0.0136** | **0.0084** | **0.0091** | **0.0057** | **0.0059** |

The reranker is now the clear winner on both splits (+35-50% over the best retrieval-only
tier on recall@10). The other three rows are unchanged within noise. The transformer row
here is still the original Step 6 model (time split, 10 epochs); the paper-protocol
evaluation of the retrained model is in `docs/10_loo_protocol.md`, and is where the
generative model should be judged.
