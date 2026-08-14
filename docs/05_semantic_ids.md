# Step 5 — Semantic ID Tokenizer (RQ-VAE)

## Why this step exists

Both baselines (`docs/03_baselines.md`... see the results table in Step 4 / the blog draft)
share one blind spot: they only know an item or user exists if it showed up during training.
Popularity ranks by interaction count; ALS learns a vector per user/item from the
interaction matrix. Neither can say anything about a user or item they've never seen.

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

## A framework detour worth documenting honestly

The original plan (`docs/00_roadmap.md`) called for `sentence-transformers` embeddings and a
PyTorch RQ-VAE. Neither worked in this sandbox:

- `sentence-transformers` requires PyTorch.
- PyTorch's CPU-only build is only published on `download.pytorch.org`, which the sandbox's
  network allowlist doesn't include.
- The default PyPI `torch` wheel pulls in several GB of CUDA dependencies (`cuda-toolkit`,
  `nvidia-cublas-*`, etc.) that don't fit the sandbox's disk quota — confirmed by actually
  trying: downloading the 526 MB torch wheel alone worked, but resolving its CUDA
  dependencies did not.

Two substitutions, both deliberate rather than incidental:

1. **Item embeddings: TF-IDF + TruncatedSVD (LSA)** instead of a sentence-transformer.
   Classic, well-understood, needs nothing beyond scikit-learn. For short e-commerce titles
   ("Skylanders: Spyro's Adventure - Xbox 360"), word-overlap signal is most of what a
   sentence embedding buys you anyway — confirmed by checking nearest neighbors before
   trusting it as RQ-VAE input (see below). Swapping in a real sentence-transformer later is
   a one-file change; nothing downstream cares how the input vector was produced.
2. **RQ-VAE: JAX** instead of PyTorch. JAX's CPU wheel is self-contained on PyPI (~90 MB, no
   proxy needed). The Adam optimizer is hand-written (`backend/models/rqvae.py`,
   `make_adam()`) rather than pulled from `optax`, partly to avoid one more dependency, partly
   because writing it out is a better forcing function for actually understanding what Adam
   does than importing it.

This is the same kind of story as the dataset saga in Step 2 — worth stating plainly rather
than glossing over, because "here's the constraint, here's the substitution, here's why it's
still a legitimate choice" is a more credible engineering narrative than pretending everything
went as originally planned.

## Building the item content embeddings

`backend/scripts/build_item_embeddings.py`:

1. TF-IDF over item titles (unigrams + bigrams, 20K vocab, English stopwords removed).
2. TruncatedSVD down to 128 dimensions (~22% explained variance — expected for LSA over a
   large, sparse vocabulary; the goal isn't reconstructing the text, just capturing enough
   structure that similar titles land near each other).
3. L2-normalize.

Sanity check before trusting this as RQ-VAE input — nearest neighbors by cosine similarity:

```
'Jeecoo V20 Stereo Gaming Headset for PS4 PS5 Xbox One...' nearest neighbors:
  0.968  ARKARTECH Gaming Headset with Mic for Xbox One PS4 PS5 PC Switch Tablet...
  0.964  Gaming Headset with Mic for Xbox One PS4 PS5 PC Switch Tablet Smartphone...
  0.956  Combatwing Pc Gaming Headset with Microphone & Led Light...

'Gunstar Heroes - Sega Genesis' nearest neighbors:
  0.931  RoboCop vs. Terminator - Sega Genesis
  0.923  Cyborg Justice - Sega Genesis
  0.920  Mystic Defender - Sega Genesis
```

Good enough — gaming headsets cluster with gaming headsets, Genesis-era games cluster with
Genesis-era games.

## The RQ-VAE

`backend/models/rqvae.py`. Three pieces:

**Encoder / decoder** — a small MLP on each side (128 → 256 → 32 latent, mirrored back out).
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

400 epochs, batch size 1024, Adam (lr 2e-3), full 25,612-item catalog. Loss (reconstruction +
codebook + 0.25·commitment) drops from 0.023 to 0.0039 and plateaus:

![RQ-VAE training curve](../data/processed/rqvae_training_curve.png)

Codebook utilization — how many of the 256 codes per level actually get used, rather than
collapsing onto a handful of them (a known VQ-VAE failure mode):

| level | codes used |
|---|---|
| 1 | 244 / 256 (95%) |
| 2 | 245 / 256 (96%) |
| 3 | 247 / 256 (96%) |

Healthy — no collapse.

## Assigning Semantic IDs

Every item's raw 3-level code (from `residual_quantize`) is a tuple like `(66, 89, 120)`.
25,612 items only produced **16,446 unique 3-level tuples** — 48.2% of items share a tuple
with at least one other item. Expected: 256<sup>3</sup> possible combinations is much larger
than 25,612 items, but codes aren't used uniformly (some regions of embedding space are
denser than others), so collisions happen well before the theoretical ceiling.

TIGER's fix, applied here: append a 4th digit — a running counter, per colliding tuple — so
every item's *full* Semantic ID (4 digits) is unique, even though the first 3 (content-derived)
digits can repeat between related items. After adding it: **all 25,612 Semantic IDs are
unique**, verified directly rather than assumed.

One actual collision, for concreteness — items 7 and 8 share `(79, 81, 86)`, disambiguated as
`(79, 81, 86, 0)` and `(79, 81, 86, 1)`:

- *Bulletstorm: Limited Edition*
- *Borderlands 2 Limited Edition Strategy Guide*

Makes sense: short titles, both dominated by "Limited Edition" in the TF-IDF signal, so they
land in the same neighborhood.

## Qualitative check: do the codes actually mean something?

Five items sharing the same level-1 code (`66, ...`):

- *Tekken - PlayStation*
- *Crash Bandicoot 3: Warped - PlayStation*
- *Syphon Filter - PlayStation*
- *R4: Ridge Racer Type 4 - PlayStation PS1*
- *Oddworld Abe's Oddysee - PlayStation*

All five are original PlayStation (PS1) titles — the model wasn't told "cluster by console
generation," it learned that grouping purely from title text and reconstruction pressure.
This is the whole point made concrete: a Semantic ID is not an arbitrary bucket, it's a
learned, content-derived neighborhood.

## What's next

Step 6 takes `semantic_ids.parquet` and `train_sequences.parquet`, converts every user's
interaction history into a sequence of Semantic IDs, and trains a decoder-only Transformer to
predict the next item's Semantic ID autoregressively — generation instead of nearest-neighbor
search, the architectural bet TIGER and Meta's HSTU both make.

## Outputs

- `data/processed/item_embeddings.npy`, `item_embeddings_index.parquet` — 128-dim TF-IDF/SVD
  content embeddings.
- `data/processed/semantic_ids.parquet` — `item_id` → 4-digit Semantic ID (3 content codes +
  1 dedup digit).
- `data/processed/rqvae_params.npz` — trained encoder/decoder/codebook weights.
- `data/processed/rqvae_metrics.json`, `rqvae_training_curve.png` — training diagnostics.
