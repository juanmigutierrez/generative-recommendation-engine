# Step 5 — Semantic ID Tokenizer (RQ-VAE)

## Why this step exists

Both baselines (see the results table in Step 4 / the blog draft) share one blind spot: they
only know an item or user exists if it showed up during training. Popularity ranks by
interaction count; ALS learns a vector per user/item from the interaction matrix. Neither can
say anything about a user or item they've never seen.

Measured cost of that blind spot (`docs/02_split_and_eval.md`): **18–28% of val/test users
are cold-start**, and ALS falls back to popularity for all of them.

A Semantic ID fixes this for items specifically. Instead of an arbitrary integer ID (item
#4821 — a number that carries zero information about what the item *is*), every item gets a
short tuple of codes derived from its content. Similar items land on similar codes, and —
critically — a brand-new item gets a valid, meaningful Semantic ID the moment it enters the
catalog, before a single interaction is recorded. That's the direct fix, and it's also what
Step 6 (generative retrieval) needs: a discrete vocabulary it can generate from, the same way
a language model generates word tokens.

This is the mechanism from TIGER (Google DeepMind, NeurIPS 2023, ["Recommender Systems with
Generative Retrieval"](https://arxiv.org/abs/2305.05065)).

## Framework choices, and why they ended up split across two machines

The RQ-VAE itself (`backend/models/rqvae.py`) is JAX, and stays JAX. PyTorch's CPU-only build
is only published on `download.pytorch.org`, which this project's sandbox can't reach, and the
default PyPI `torch` wheel pulls in several GB of CUDA dependencies that don't fit the
sandbox's disk quota — confirmed by actually trying, not assumed. JAX's CPU wheel is
self-contained on PyPI (~90 MB). The Adam optimizer (`rqvae.make_adam`) is hand-written rather
than pulled from `optax`, partly to avoid one more dependency, partly because writing it out
is a better forcing function for actually understanding what Adam does than importing it.

Item *embeddings* (the RQ-VAE's input) are a different story, and went through several
iterations worth documenting honestly:

1. First attempt, in-sandbox: TF-IDF + TruncatedSVD (LSA) over item titles, no PyTorch needed
   at all. Worked, and nearest-neighbor checks looked reasonable, but it's word-overlap
   signal, not real semantic understanding — "PS4" and "PlayStation 4" are unrelated to it
   unless those exact strings co-occur enough for SVD to link them.
2. Second attempt: load `sentence-t5-base` (the actual model family TIGER's paper uses) into
   JAX/Flax, run locally. Blocked — `transformers` v5 (2025) removed Flax/TensorFlow support
   from the library entirely, confirmed directly in HuggingFace's own migration guide
   (`https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md`, "Removal of
   TensorFlow and Jax"). Pinning an old `transformers` version just for this one script wasn't
   worth it.
3. Final: `sentence-t5-base` via the standard `sentence-transformers` library (PyTorch), run
   locally, then moved to Google Colab (free GPU) once local CPU inference turned out to be
   slow for a 110M-parameter model over 25,612 items. Real semantic embeddings, produced on
   whichever machine could actually run them efficiently, fed back into the sandbox where the
   RQ-VAE (still JAX) consumes them.

The split — PyTorch for embedding a pretrained model, JAX for the from-scratch RQ-VAE — isn't
arbitrary. It's "use the tool each sub-task actually supports," which is a more honest
engineering story than forcing one framework everywhere for consistency's sake.

## Building the item content embeddings

`backend/scripts/build_item_embeddings_local.py`, run on Colab (see
`notebooks/colab_item_embeddings.ipynb`): `sentence-transformers/sentence-t5-base`
(~110M-parameter encoder, 768-dim output), mean-pooled per the model's own method, L2-normalized.

Sanity check before trusting this as RQ-VAE input — nearest neighbors by cosine similarity:

```
'Gunstar Heroes - Sega Genesis' nearest neighbors:
  0.935  Starflight - Sega Genesis
  0.930  Gunstar Super Heroes
  0.926  Gaiares - Sega Genesis
  0.925  Phantasy Star II - Sega Genesis
  0.918  Phantasy Star IV - Sega Genesis

'Nintendo Splatoon Series - Octoling Amiibo 3-pack - Switch' nearest neighbors:
  0.935  amiibo - Octoling (Blue) - Splatoon Series
  0.924  Nintendo Switch – OLED Model Splatoon 3 Special Edition
  0.921  amiibo - Inkling (Yellow) - Splatoon Series
  0.916  Nintendo Amiibo - Squirtle - Super Smash Bros. Series - Switch
```

Noticeably better than the TF-IDF version — "Phantasy Star II/IV" showing up for Gunstar
Heroes is a same-era, same-publisher, similar-genre association a pure word-overlap method
would never make, since the titles share almost no words.

## Anisotropy: a hidden landmine in pretrained sentence embeddings

The first attempt to train the RQ-VAE on these real embeddings failed completely — not a
crash, worse: it silently converged to a **totally collapsed codebook**. Every single one of
25,612 items mapped to the exact same 3-digit code. Loss looked great (flat and low) right up
until checking what the codes actually were.

Root cause, measured directly rather than guessed at: the average cosine similarity between
two *random, unrelated* items was **0.76**. In a well-spread embedding space, unrelated items
should sit close to orthogonal (~0 similarity). BERT-family sentence embeddings are known to
be anisotropic — nearly all of a vector's magnitude sits in one direction shared by every
item, a well-documented phenomenon in NLP literature (see the "BERT-flow" / "whitening-BERT"
line of work). That shared direction is exactly what a small encoder collapses onto: reconstructing
the *average* embedding minimizes loss almost as well as actually differentiating items,
so with no counter-pressure, that's the lazy solution gradient descent finds.

Two fixes, both standard and both necessary together — data-dependent codebook init alone
only partially helped (level 1 usage went from 1 code to 4 out of 256 — better, still broken):

1. **Standardize the input.** Center (subtract the mean vector) and scale to unit variance
   per dimension before the encoder ever sees the embeddings. This directly attacks the shared
   dominant direction. Measured effect: average random-pair cosine similarity dropped from
   0.76 to ~0.0005 after standardizing.
2. **Data-dependent codebook initialization.** Instead of small-random-noise codebook init
   (fine for the TF-IDF embeddings, which happened to already sit near the origin at a
   compatible scale), run the freshly-initialized encoder once over the real data and seed
   each codebook via k-means on those outputs (`rqvae.kmeans_init_codebooks`), level-by-level
   on the residual stream. This guarantees the codebook starts in the actual region of space
   the encoder produces, rather than hoping training finds it.

This is a real, worth-remembering lesson for anyone doing this with a real pretrained
embedding model rather than a hand-built one: **don't feed raw sentence-embedding-model output
straight into a from-scratch encoder without checking its geometry first.**

## The RQ-VAE

`backend/models/rqvae.py`. Three pieces:

**Encoder / decoder** — a small MLP on each side (768 → 256 → 32 latent, mirrored back out).
Nothing unusual: this is what makes it a *variational auto*encoder-style architecture —
compress, then reconstruct, and the quality of the reconstruction is the training signal.

**Residual quantization** — the "RQ" in RQ-VAE, and the part worth being able to explain
carefully. A single codebook (plain VQ-VAE) snaps the latent vector to its single nearest
neighbor among K learned code vectors — coarse, and K has to be huge to capture fine detail.
Residual quantization uses several small codebooks in sequence instead: codebook 1 quantizes
the latent vector, codebook 2 quantizes *what codebook 1 got wrong* (the residual), codebook 3
quantizes what's still left over after that. Three codebooks of size 256 give
256<sup>3</sup> ≈ 16.7M possible combinations from only 768 total learned vectors — far more
representational range than one codebook of any practical size, for a fraction of the
parameters.

**Straight-through estimator** — argmin (picking the nearest codebook vector) has no useful
gradient. The trick (standard since the original VQ-VAE paper, van den Oord et al. 2017): run
the quantized vector forward through the decoder, but on the backward pass, pretend
quantization was the identity function and let gradients flow straight back into the encoder
as if nothing was snapped to a codebook at all. Two extra loss terms keep the codebooks and
encoder honest: a codebook loss that pulls each codebook vector toward the encoder outputs
assigned to it, and a commitment loss that pulls the encoder toward the codebook vector it
picked, so the encoder doesn't keep wandering to a new point every step.

## Training

400 epochs, batch size 1024, Adam (lr 2e-3), full 25,612-item catalog, on the standardized
embeddings with k-means-initialized codebooks. Loss (reconstruction + codebook +
0.25·commitment) drops from ~3.4 to ~1.03 and plateaus:

![RQ-VAE training curve](../data/processed/rqvae_training_curve.png)

Codebook utilization — how many of the 256 codes per level actually get used:

| level | codes used |
|---|---|
| 1 | 73 / 256 (29%) |
| 2 | 247 / 256 (96%) |
| 3 | 246 / 256 (96%) |

An honest, real result worth stating plainly rather than rounding up: levels 2 and 3 are
healthy, level 1 is lower than the TF-IDF version's run got (95%). No sign of collapse (73
active codes, still climbing when training stopped, vs. 1), but level 1 apparently has fewer
natural "coarse" clusters in real semantic space than the codebook has room for — plausible,
since level 1 has to capture the broadest category distinctions first, before levels 2 and 3
refine within them. This would be worth revisiting with more training epochs or a smaller
level-1 codebook if this were going into production.

## Assigning Semantic IDs

Every item's raw 3-level code (from `residual_quantize`) is a tuple like `(62, 215, 102)`.
25,612 items produced **21,713 unique 3-level tuples** — 25.3% of items share a tuple with at
least one other item (down from 48.2% with the TF-IDF embeddings — real semantic embeddings
differentiate items better, so fewer collide before the theoretical 256<sup>3</sup> ceiling).

TIGER's fix, applied here: append a 4th digit — a running counter, per colliding tuple — so
every item's *full* Semantic ID (4 digits) is unique, even though the first 3 (content-derived)
digits can repeat between related items. After adding it: **all 25,612 Semantic IDs are
unique**, verified directly rather than assumed.

## Qualitative check: do the codes actually mean something?

Five items sharing the same level-1 code (`62, ...`):

- *Saitek Eclipse Keyboard (PZ30AU)*
- *DSI Left Handed Mechanical Keyboard Cherry MX Red KB-DCK-LH104-V2*
- *Creative Labs Fatal1ty 1010 Gaming Mouse*
- *Saitek Eclipse Backlit Keyboard - Red LED (PZ30AUR)*
- *Ideazon MERC Gaming Keyboard*

All five are gaming keyboards/mice — a tighter, more precise cluster than the TF-IDF version's
"PlayStation console generation" grouping, and again, the model wasn't told to group by
peripheral type. It learned that purely from title content and reconstruction pressure on real
semantic embeddings.

## What's next

Step 6 takes `semantic_ids.parquet` and `train_sequences.parquet`, converts every user's
interaction history into a sequence of Semantic IDs, and trains a decoder-only Transformer to
predict the next item's Semantic ID autoregressively — generation instead of nearest-neighbor
search, the architectural bet TIGER and Meta's HSTU both make.

## Outputs

- `data/processed/item_embeddings.npy`, `item_embeddings_index.parquet` — 768-dim
  `sentence-t5-base` content embeddings (generated on Colab GPU).
- `data/processed/rqvae_embedding_standardization.npz` — mean/std used to standardize
  embeddings before the encoder; needed to embed any new item consistently later.
- `data/processed/semantic_ids.parquet` — `item_id` → 4-digit Semantic ID (3 content codes +
  1 dedup digit).
- `data/processed/rqvae_params.npz` — trained encoder/decoder/codebook weights.
- `data/processed/rqvae_metrics.json`, `rqvae_training_curve.png` — training diagnostics.
