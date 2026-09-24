# Roadmap

This is the build plan for the rest of the project: what gets built, in what order, why that
order, and what you should be able to explain about each piece afterward. Written to double
as an outline for the eventual blog post — each numbered stage below maps to one section.

Data pipeline (dataset selection, cleaning, time-based split, sequence building) is done —
see `docs/01_dataset.md` and `docs/02_split_and_eval.md`. Everything below builds on top of
`data/processed/{train,val,test}.parquet` and `train_sequences.parquet`.

## The shape of what we're building

A **two-stage recommender**, the pattern real systems (YouTube, Amazon, Meta) actually use:

```
                    ┌─────────────┐        ┌──────────┐
user history  ───▶  │  RETRIEVAL   │ ───▶   │ RANKING  │ ───▶  top-K recommendations
                    │ (cheap, wide)│ 100s   │(precise, │
                    └─────────────┘ candid. │ narrow)  │
                                             └──────────┘
```

Retrieval's job: narrow millions of items down to a few hundred plausible candidates,
cheaply. Ranking's job: take those candidates and order them accurately, using more
expensive per-candidate computation you couldn't afford to run on the full catalog. Every
stage below either builds one of these two stages, or builds something that measures /
serves / demos them.

---

## Steps 1–3 — already done

**Step 1 — Project scaffold.** Repo structure (`backend/`, `data/`, `docs/`, `frontend/`),
README stating the architecture up front, `requirements.txt`, `.gitignore`. Why bother with
this before any data or model code: it forces you to state the plan before writing anything,
and gives every later file a home instead of accumulating loose scripts.

**Step 2 — Dataset acquisition & exploration.** This took three attempts, which is worth
being able to talk about honestly in an interview rather than glossing over:
1. Amazon Reviews 2023 was the plan, but the build environment's network was locked to
   GitHub only — Hugging Face, Kaggle, and UCI's own site were all unreachable.
2. Landed on **UCI Online Retail** as a placeholder (real transaction data, reachable via a
   GitHub mirror) to keep the pipeline moving while the Amazon path got fixed.
3. Once Amazon Reviews became reachable (downloading directly from McAuley Lab's file
   server, run locally on your machine, not the sandbox), the *raw* `All_Beauty` category
   turned out to be 93.9% one-time reviewers — no history to model from. Switched to the
   **5-core `Video_Games`** subset (every user/item guaranteed ≥5 interactions by
   construction), which is what the project is actually built on now.

Full writeup: `docs/01_dataset.md`. The takeaway worth stating out loud: "real-world data
acquisition rarely works on the first try, and here's the concrete reason each attempt
failed" is a more credible interview answer than a project that pretends the first dataset
choice was always right.

**Step 3 — Data pipeline.** Three scripts, each with one job (`backend/scripts/`):
`prepare_data_amazon.py` (raw → cleaned, ID-encoded, unified schema), `split_data.py`
(time-based train/val/test — not random, see `docs/02_split_and_eval.md` for why that
matters), `build_sequences.py` (flat interactions → per-user chronological sequences for
the retrieval model, and per-user ground-truth target sets for evaluation). All three
produce the same output schema regardless of which raw dataset feeds them — the "adapter
pattern" reasoning for that is worth being able to explain too (asked about earlier in this
build).

---

## Step 4 — Baseline models

**Build:** a popularity model (recommend whatever's most interacted-with, same list for
everyone) and an ALS matrix-factorization model (`implicit` library) trained on the
train-split interactions.

**Why first, before anything sophisticated:** you can't claim a fancy model is "good"
without a number to compare it against. Baselines establish the floor. They also surface
the model comparison table (Recall@K / NDCG@K per model) that becomes the project's main
piece of evidence — "here's what each added layer of complexity actually bought us,"
measured, not assumed.

**Concepts to be able to explain:** implicit feedback (we only observe positive
interactions, never explicit "user dislikes this" signal, which changes how you train and
evaluate vs. explicit ratings); matrix factorization (representing users and items as
learned vectors in the same space, where dot product ≈ affinity); why popularity is a
surprisingly strong baseline in practice and hard to beat with sparse data.

## Step 5 — Semantic ID tokenizer (RQ-VAE) — done

Built and documented in full in `docs/05_semantic_ids.md`. Short version below; that file
has the math, the results, and an honest writeup of a torch → JAX framework pivot forced by
sandbox network/disk constraints.

**Build:** an RQ-VAE (residual-quantized variational autoencoder) trained on item content
embeddings (generated from item titles via `sentence-transformers`). Output: every item
gets a short tuple of discrete codes — its "Semantic ID" — instead of an arbitrary integer
ID.

**Why:** this is the mechanism from the TIGER paper (Google DeepMind) that the retrieval
stage depends on. A plain integer item ID (item #4821) carries no meaning — two similar
items could have wildly different IDs. A Semantic ID is *derived from what the item is*, so
similar items get similar codes, and — critically — a brand-new item with zero interaction
history still gets a valid, meaningful Semantic ID the moment it's added to the catalog.
That's the direct fix for the cold-start gap measured in `docs/02_split_and_eval.md`
(18–28% of eval users/items are cold-start).

**Concepts to be able to explain:** what a VAE does (learn a compressed latent
representation, encoder/decoder trained together); what "quantization" adds (snap the
continuous latent vector to the nearest of a fixed set of learned codebook vectors, so the
representation becomes discrete/tokenizable — necessary because Step 6 needs to *generate*
these IDs like a language model generates tokens); why "residual" quantization (multiple
codebooks applied in sequence, each correcting the previous one's error) gives finer-grained
codes than a single codebook could.

## Step 6 — Generative sequential retrieval model — done

Built and documented in full in `docs/06_generative_retrieval.md`. Short version below;
that file has the tokenization scheme, the timing constraint that shaped the context-length
choice, the honest mixed val/test result, and a worked example.

**Build:** a decoder-only Transformer (GPT-style, causal attention) trained on each user's
history as a sequence of Semantic IDs, predicting the next item's Semantic ID
autoregressively.

**Why:** this replaces the classic "embed the user, do a nearest-neighbor search over all
item embeddings" retrieval approach with direct generation — the model *writes out* the
tokens of the next item's ID rather than searching for it. This is the architectural bet
Meta's HSTU and Google's TIGER both make, and it's the centerpiece of why this project cites
2023–2026 research rather than a textbook SVD.

**Concepts to be able to explain:** why sequence order matters for recommendations (recency,
momentum — the 5th-most-recent purchase predicts the next one better than a random past
purchase); how constrained/beam decoding maps generated token sequences back to actual valid
items (not just any token sequence corresponds to a real product); how this compares to
two-tower embedding search and what each approach trades off (index-build cost vs. inference
cost, ease of adding new items, etc.).

## Step 7 — Ranking / reranking stage — done

Built and documented in full in `docs/07_ranking.md`. Short version below; that file has
the feature table, the LambdaMART training details, the honest val-vs-test result, and a
worked example.

**Build:** a LightGBM LambdaMART ranker (`LGBMRanker`, `objective="lambdarank"`) that takes
the ~100 candidates ALS retrieval produced for a user and reorders them, using features
retrieval couldn't afford to use for every item in the catalog (price with imputation,
recency, popularity, content similarity from the Step 5 embeddings, retrieval's own score).

**Why:** this is the second half of the two-stage pattern. Retrieval optimizes for coverage
and speed over the whole catalog; ranking optimizes for precision over a small candidate
set, where it's affordable to compute richer features per item.

**Concepts to be able to explain:** why you wouldn't just run the ranking model over the
entire catalog directly (compute cost at scale); feature engineering for ranking (what
signals were available and why they were chosen, including how the missing-price data gap
from `docs/01_dataset.md` got handled -- global-median imputation plus a `has_price` flag,
since no clean category taxonomy existed for the originally-planned category-median
approach); why reranking has to be evaluated on the *same* candidate set, reordered or not,
to isolate what the ranking stage itself contributes versus what retrieval already did;
learning-to-rank (LambdaMART/`lambdarank`) versus plain binary classification for this kind
of problem.

## Step 8 — Offline evaluation framework — done

Built and documented in full in `docs/08_evaluation.md`. Short version below; that file has
the full results table, the cold-start fallback rule applied consistently across every
tier, and an honest read of why no single tier wins everything.

**Build:** one script (`backend/scripts/run_full_evaluation.py`) that runs Recall@K,
NDCG@K, and MAP@K for every model tier (popularity → ALS → Semantic ID Transformer →
ALS + ranking) against the same 2,000-user sample per split, and outputs a single
comparison table.

**Why:** this produces the actual "results" evidence for both the interview story and the
blog post — a table showing each stage's measured contribution, not just "trust me, the
fancy model is better." It also closes a gap Steps 6 and 7 each flagged honestly at the
time: their own evaluations excluded cold-start users (no train history → no beam-search
context, no ALS candidates), while Step 4's baseline table included them via a popularity
fallback. Step 8 applies that same fallback to every tier, so the whole table is finally
apples-to-apples.

**Concepts to be able to explain:** why "evaluate every model the same way" is itself a
nontrivial design decision, not a formality — three earlier scripts each made reasonable,
individually-documented scope choices that turned out not to be comparable to each other;
MAP@K vs. NDCG@K (both reward early hits, but MAP rewards *several* early hits compounding,
NDCG discounts more smoothly by rank); why "no tier wins every metric on every split" is a
real, defensible finding for a multi-mechanism system rather than an unfinished result.

## Step 9 — FastAPI serving layer

**Build:** a small API exposing `/recommendations/{user_id}` and similar endpoints, wrapping
the trained models for real-time inference, with basic request logging.

**Why:** trained models sitting in a notebook aren't a "system." Serving them behind an API
is what makes this look like production work rather than a research exercise — and it's
what the frontend (Step 10) will call.

## Step 10 — React frontend

**Build:** a small demo UI — browse products, pick/simulate a user, see live recommendations
with a short "why this was recommended" explanation per item.

**Why:** this is the piece a non-technical interviewer or blog reader can actually look at
and understand in 10 seconds, without reading model code. It's also the most "showcase"-y
part of the project.

## Step 11 — Dockerize & deployment prep

**Build:** Docker Compose wiring the API + frontend together, plus instructions to deploy
somewhere public (Render/Fly.io/HF Spaces) so the project has a live link, not just a repo.

## Step 12 — Final verification & interview-ready docs

**Build:** an end-to-end run-through from raw data to live demo to confirm nothing's broken,
polish the README, and write a summary doc pulling together the "what/why/results" from
every stage — this becomes the blog post's skeleton directly.

---

## On the blog post

Everything in `docs/` has been written stage-by-stage as we go, each with a "why," not just
a "what" — that's deliberate, so those files can be lightly edited into blog post sections
rather than written from scratch at the end. Suggested structure when we get there: problem
framing → dataset & the 3 dataset attempts (a good "real engineering has false starts" story)
→ architecture diagram → each stage with its measured contribution → live demo link → what
you'd do differently at 10x scale.
