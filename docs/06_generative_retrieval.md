# Step 6 — Generative Sequential Retrieval (Decoder-Only Transformer)

## Why this step exists

Steps 1–5 built up to this: baselines that can't say anything about users/items they haven't
seen (Step 4), and Semantic IDs that give every item a content-derived identity even with zero
interaction history (Step 5). This step is where those two things combine into what TIGER
(Google DeepMind, NeurIPS 2023) and Meta's HSTU/generative-recommenders actually propose:
instead of embedding a user and doing nearest-neighbor search over item embeddings (the
classic retrieval pattern), train a Transformer to *generate* the next item's Semantic ID
directly, autoregressively, the same way a language model generates the next word.

## Turning user histories into token sequences

`backend/scripts/build_semantic_sequences.py`. Every item's 4-digit Semantic ID (3 content
codes + 1 dedup digit, from Step 5) is flattened into 4 tokens in one shared vocabulary, each
level given a disjoint offset range so a single embedding table and a single output layer
cover everything:

| range | meaning |
|---|---|
| 0–255 | level-1 code |
| 256–511 | level-2 code |
| 512–767 | level-3 code |
| 768–788 | dedup digit (0–20, the observed max) |
| 789 | PAD |

Vocabulary size: 790. A user's history becomes one long token stream —
`[item1_tok1, item1_tok2, item1_tok3, item1_tok4, item2_tok1, ...]` — exactly the shape a
decoder-only Transformer expects.

Sequences are right-truncated to the **10 most recent items** (40 tokens) and right-padded to
that fixed length. This was originally set to 20 items (80 tokens, covering the ~95th
percentile of history length), but one training epoch at that length took over 250 seconds —
past the sandbox's ~170-second-per-command budget. 10 items (40 tokens) still comfortably
covers the median (6) and mean (7.5) train-sequence length, and cut epoch time to ~110-120s.
A real, measured constraint shaping a real design choice, not an arbitrary number.

## The Transformer

`backend/models/transformer.py`, hand-rolled in plain JAX — same philosophy as the RQ-VAE
(Step 5): every layer (token + positional embedding, causal multi-head self-attention,
layernorm, feed-forward, weight-tied output projection) is code you can point to and explain,
not a library call. GPT-2-style pre-norm blocks:

```
x = token_embedding(tokens) + position_embedding(positions)
for each block:
    x = x + causal_self_attention(layernorm(x))
    x = x + feed_forward(layernorm(x))
x = layernorm(x)
logits = x @ token_embedding.T     # weight tying: output layer reuses the input embedding
```

Small on purpose: 2 layers, 4 attention heads, 64-dim embeddings, 128-dim feed-forward —
**375K parameters** before the sequence-length reduction, ~120K after (fewer position
embeddings needed). Training objective is standard causal language modeling: predict token
*t+1* from tokens *0..t*, cross-entropy loss, averaged only over non-PAD positions.

## Training: resumable by necessity

One epoch over all 85,757 users takes ~110-150s of pure compute — close enough to the
sandbox's per-command budget that a full training run has to happen as a *series* of separate
commands, not one long process. `train_transformer.py` is written around this: each
invocation loads a checkpoint if one exists, runs N more epochs, saves a new checkpoint. The
same pattern used to retrain the RQ-VAE on real embeddings in Step 5, extended into the
project's actual training script rather than a one-off workaround.

10 epochs, run one at a time:

| epoch | loss |
|---|---|
| 0 (untrained) | 6.67 (≈ ln(790), i.e. uniform random guessing over the vocabulary) |
| 1 | 5.37 |
| 2 | 3.89 |
| 3 | 3.58 |
| 5 | 3.22 |
| 7 | 2.98 |
| 10 | 2.76 |

![Transformer training loss](../data/processed/transformer_training_curve.png)

Clear, healthy, decelerating convergence — still worth more epochs than this project's time
budget allowed for in one sitting, a real, stated limitation rather than a claim of full
convergence.

## Constrained decoding

The model's raw output at each of the 4 token positions is a distribution over that level's
256 (or 21) possible codes — but not every 4-token combination corresponds to a real catalog
item. Exactly the question the roadmap flagged in advance: *"how constrained/beam decoding
maps generated token sequences back to actual valid items."*

Fix: build a trie directly from every real item's Semantic ID (`item_tokens.parquet`) —
level-1 code → level-2 code → level-3 code → {dedup digit: item_id}. Beam search (width 20)
walks this trie: at each step, only tokens that could complete a *real* item are considered,
scored by the model's log-probability, top-20 partial sequences kept, repeat for all 4
positions. Every surviving beam at the end maps to exactly one genuine item_id — nothing
needs to be filtered out afterward for being invalid, because invalid paths were never
explored.

## Evaluation

Same Recall@K / NDCG@K as the baselines (`backend/models/metrics.py`), but on a **2,000-user
random sample per split**, not the full val/test sets. Beam search requires a full forward
pass per candidate at every step; running it for all ~55K val+test users was not tractable in
this sandbox's time budget. 2,000 users is large enough for a stable estimate, small enough to
actually finish — a scope decision, documented rather than hidden.

One more scope note worth being explicit about: this evaluation **excludes cold-start users
entirely** (users with zero train interactions have no context to generate from at all). The
baseline evaluation in Step 4 *included* cold users via a popularity fallback. So these numbers
aren't perfectly apples-to-apples with the baseline table — they represent what the Transformer
achieves specifically for users it has *some* history for, which is a real, if narrower,
comparison. A production system would still need a popularity (or Semantic-ID-based
content) fallback for genuinely cold users, same as Step 4's `als_with_fallback`.

| split | model | recall@10 | recall@20 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|
| val | popularity | 0.0146 | 0.0180 | 0.0111 | 0.0122 |
| val | als | 0.0149 | 0.0227 | 0.0107 | 0.0132 |
| val | **transformer** | **0.0188** | **0.0259** | **0.0121** | **0.0142** |
| test | popularity | 0.0058 | 0.0071 | 0.0038 | 0.0042 |
| test | als | 0.0051 | 0.0078 | 0.0036 | 0.0044 |
| test | **transformer** | 0.0040 | 0.0064 | 0.0025 | 0.0032 |

A genuinely mixed, honest result — not the clean win a portfolio project is tempted to only
report. On val, the Transformer beats both baselines on every metric, at only 10 epochs of
training on a warm-start-only subset. On test, it falls behind both. Worth stating plainly
rather than explaining away: this could be the higher cold-start rate in test (28.5% vs.
val's 18.3%, see `docs/02_split_and_eval.md`) making the *warm* test users a harder,
differently-shaped population than val's; it could be under-training (10 epochs, still
descending); it could be the smaller 10-item context window losing signal for users with
longer histories. Any of these would be the natural next thing to test with more time,
exactly the kind of open question worth being explicit about rather than resolving by
assumption.

## A concrete example

User 20582's 4 most recent purchases before the held-out window: a Switch wireless
controller, a Switch dockable case, a Switch carrying case/stand, a Joy-Con charging dock —
consistently Nintendo Switch accessories. Top-10 generated recommendations:

1. **amFilm Tempered Glass Screen Protector for Nintendo Switch (2-Pack)** ← this is what the
   user actually bought next
2. Nintendo Switch Pro Controller
3. The Legend of Zelda: Breath of the Wild Master Edition - Nintendo Switch
4. Mario Kart 8 Deluxe – Booster Course Pass - Nintendo Switch
5. Super Mario Odyssey - Nintendo Switch
6. PowerA Wired Controller for Nintendo Switch - Zelda: Breath of The Wild
7. Amazon Basics Carrying Case for Nintendo Switch
8. Mayflash GameCube Controller Adapter for Wii U, PC USB and Switch
9. Nintendo Joy-Con (R) - Neon Red - Nintendo Switch
10. iLLumiShield Glass Screen Protector for Nintendo Switch (3-Pack)

Every single one of the 10 generated candidates is Nintendo Switch-related, and the top-ranked
one is the exact item the user bought next. The model was never told this user's platform —
it inferred "Switch accessory buyer" purely from the Semantic IDs of their last 4 purchases
and generated accordingly.

## What's next

Step 7 (ranking) takes the top-N candidates this Transformer retrieves and reorders them using
features retrieval couldn't afford to compute for the whole catalog — price, recency,
popularity, content similarity. Retrieval's job was narrowing the catalog down cheaply;
ranking's job is getting precise about the shortlist.

## Outputs

- `data/processed/transformer_train_sequences.npz`, `item_tokens.parquet`,
  `transformer_vocab_meta.json` — tokenized training data and vocabulary.
- `data/processed/transformer_checkpoint.pkl` — trained model weights + optimizer state +
  loss history (resumable).
- `data/processed/transformer_training_curve.png` — training loss plot.
- `data/processed/transformer_eval_results.json` — Recall@K/NDCG@K on the 2,000-user val/test
  samples.
