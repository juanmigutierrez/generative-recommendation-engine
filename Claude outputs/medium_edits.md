# Medium draft — paste-ready edits

Keyed to the sections of the Medium draft ("Generative AI for Recommendations: What
YouTube, Netflix and Meta Are Moving To, Built From Scratch"). Each block below is either a
**replacement** for an existing passage (the old text is quoted so it's easy to find) or a
**new section** with where it goes. Numbers come from `docs/08_evaluation.md` (time split,
after the ranker fix) and `docs/10_loo_protocol.md` (leave-one-out, all 94,762 users).

`[COLAB: ...]` marks where an "Open in Colab" badge goes once the tutorial notebooks are
pushed (`notebooks/tutorial_01_baselines.ipynb` etc.).

---

## 1. Intro — replace the two clunky sentences

**Old:** "Now, possibly you have heard that before, what is generative AI recommender ? Its
AI that will use a generative model for recommend similar objects, it will work on unseen
data better."

**New:**

> That is the classic, *collaborative* way of recommending: compare people, and compare
> items, through who interacted with what. A generative recommender does something
> different. It treats your history as a sentence and *writes* the next item, the way a
> language model writes the next word — and because each item's "word" is built from its
> content, it can write items nobody has bought yet.

(Then the "This works well until something new shows up..." paragraph follows naturally.
Delete the sentence "That's where a generative recommender comes in. Instead of scoring
items by similarity, it treats your history like a sentence and generates the next item,
the way a language model generates the next word" — it now duplicates the bridge above; keep
its second half about the item's word being built from content if you like.)

Also in this section: "Editing" in the page title is Medium's editor label, not part of the
title.

---

## 2. Dataset — add one sentence at the end

> `[COLAB: tutorial_01 — download the data, build the split, run the baselines]`
>
> One thing I got wrong at first and only found out much later: being on a benchmark dataset
> does *not* make results comparable to the papers on its own. The evaluation protocol has
> to match too, and mine — described next — deliberately didn't. The section "Measuring it
> the way the papers do" is where that gets fixed.

---

## 3. Scoring the models — replace the comparability paragraph

**Old:** "Both are averaged over all test users. We report K = 10 and 20, the two cutoffs the
TIGER and HSTU papers use, so our numbers can be read against theirs.

One thing to know before the tables: the absolute numbers will look small (a Recall@10 of
0.02 means 2%). That's normal for this dataset. With 25,612 items and a median of 6
interactions per user, guessing the exact next purchase is hard, and published results on
the same category are in the same range. What matters is the comparison between models, not
the raw value."

**New:**

> Both are averaged over all evaluation users, at K = 10 and K = 20.
>
> One thing to know before the tables: the absolute numbers will look small — a Recall@10 of
> 0.02 means 2%. Part of that is the dataset (25,612 items, a median of 6 interactions per
> user). But most of it is the *question* this split asks. Splitting by time means a user's
> training history ends in 2019 and their test items are what they reviewed in 2021–2023 —
> a median gap of almost four years, and 72% of the test items didn't exist in the training
> data at all. That is the realistic production question ("what will this person want next
> year?"), and it is a brutal one. The papers ask an easier and more standard one, and later
> in the post I measure that way too so the two can be told apart. Until then, what matters
> is the comparison between models, not the raw value.

---

## 4. Semantic IDs — fix the "Add results" note

**Old:** the paragraph starting "*Add results*: 'Jeecoo V20 Stereo Gaming Headset ...'" (raw
nearest-neighbour dump).

**New** (drop it in after "…a grouping the model was never told to make."):

> The embedding step alone already knows what things are. Nearest neighbours of one
> product title, by cosine similarity in the Sentence-T5 space:
>
> **Jeecoo V20 Stereo Gaming Headset for PS4 PS5 Xbox One — Over Ear Headphones with Noise
> Cancelling Microphone, LED Light**
> → 0.962 · *Jeecoo J20 Stereo Gaming Headset for PS4, Pro, Xbox One S, Xbox One Controller,
> Noise Cancelling Over Ear Headphones with Mic*
> → 0.959 · *PS4 Gaming Headset | Xbox One Headset | VOTRON Over Ear Stereo Gaming
> Headphones with LED Light Noise Reduction*
>
> Nothing about "headset" was ever labelled; that structure is what the RQ-VAE compresses
> into three digits.
>
> `[COLAB: tutorial_02 — embed a 5K-item subset, train the RQ-VAE, look at the clusters]`

Typos in this section: "toward the7 codebooks" → "toward the codebooks"; "25⁶³ ≈ 16.7
million" → "256³ ≈ 16.7 million".

---

## 5. Generative retrieval — merge the two sections and describe one model

The draft has a bold "Generative Retrieval" mini-section ("Semantic IDs (the last step) gave
every item a short, meaningful code…") immediately followed by the H1 "Generative
retrieval: writing the next item instead of searching for it" that says the same thing, and
the "2 Generative Retrieval with Semantic IDs" figure appears twice (once in each). Keep only
the H1 section and one copy of the figure.

It also describes two different models: "twenty items become a stream of 80 tokens, plus
one token at the front that identifies the user" (the final model) and "2 layers, 4
attention heads, 64-dim, 375K parameters" (the first, smaller one). Use the final model
throughout.

**Replace Step 2 with:**

> **Step 2: train a small language model on it.** A decoder-only Transformer — the
> architecture behind GPT, hand-written here rather than imported: 4 layers, 4 attention
> heads, 128-dim, dropout 0.1, about 1.1M parameters — reads the stream and, at every
> position, predicts the next token from everything before it. Training maximises the
> probability of the token that actually came next, which is the same as minimising
> cross-entropy:

(equation unchanged)

**Replace the sentence in Step 4** "beam search keeps the 20 best partial IDs alive" with
"beam search keeps the 30 best partial IDs alive, drops anything the user already has, and
returns the top 20".

**After "Every recommendation this produces is guaranteed to be a real item; nothing needs
filtering afterward." add:**

> Training took 45 passes over the data on a Colab GPU, bringing the loss from 4.2 (random
> digits) to 2.2 per digit — and still going down slowly when I stopped, which matters
> later.
>
> `[COLAB: tutorial_03 — train on a 20K-user subset for five minutes, run constrained beam
> search, inspect a user's generated list]`

Keep the "One concrete case" Switch paragraph as is.

---

## 6. Ranking — replace "Training data" and "What the trees learned to look at"

**Old:** "**Training data.** 12,000 users, 100 ALS candidates each, labeled by what they
actually bought next. When the true item wasn't in ALS's top 100 we injected it, so the
ranker always has at least one positive to learn from: 1.2 million rows, 29,000 positives.
Training took under a minute.

**What the trees learned to look at.** The most-used features were retrieval's own score,
the user's history length, and item popularity; content similarity came fifth. Price barely
mattered."

**New:**

> **Training data.** 18,000 users, ALS's top-100 candidates for each, labelled by what they
> actually bought next. Only 2,347 of those users had their true next item anywhere in
> ALS's top 100 — that is retrieval's recall@100, and it is the ceiling a reranker works
> under. Those 2,347 users, 235K rows, 2.9K positives, are the training set. Training took
> under a minute.
>
> **The bug I shipped first.** My first version "fixed" the missing positives: when the true
> item wasn't in ALS's top 100, I injected it into the list with a placeholder retrieval
> score (the lowest one in the list). Since ALS misses the true item about 90% of the time,
> 93% of the ranker's positives were injected — and it learned exactly what I had taught it:
> *the lowest-scored, never-seen candidate is the one to promote.* Probing the trained model
> with every other feature held fixed, a retrieval score of 0.1 got +0.4 and a score of 0.5
> got −4.4; an item with zero training interactions got +6.5. On real candidates that is the
> opposite of useful, and the reranker duly broke even with plain ALS. A reranker can only
> reorder what retrieval gives it, so it has to be trained on what retrieval gives it, with
> the real feature values. That is the whole fix.
>
> **What the trees learned to look at**, after the fix: retrieval's own score first, then
> content similarity — the Semantic-ID embedding space, used continuously — then how
> recently the item has been trending, then popularity. Price and the user's history length
> mattered least.
>
> `[COLAB: tutorial_04 — build the features, reproduce the injection bug in one cell and fix
> it in the next]`

Typo: "Retrieval has one job: be cheap and wide" is fine; "how recently an item has been
selling" ok.

---

## 7. NEW SECTION — after Ranking: "What each piece actually bought"

> ## What each piece actually bought
>
> Every tier — popularity, ALS, the Semantic-ID Transformer, ALS plus the reranker — scored
> on the same 2,000 users per split under the time-based split, with the same rule for the
> users who have no training history at all (everyone falls back to the popularity list
> for them; that is 29–42% of the sample, and excluding them would be hiding the hardest
> part of the problem).
>
> | split | model | recall@10 | recall@20 | ndcg@10 | ndcg@20 |
> |---|---|---|---|---|---|
> | val | popularity | 0.0169 | 0.0187 | 0.0139 | 0.0146 |
> | val | ALS | 0.0140 | 0.0210 | 0.0111 | 0.0134 |
> | val | Semantic-ID Transformer | 0.0178 | 0.0208 | 0.0134 | 0.0146 |
> | val | **ALS + reranker** | **0.0229** | **0.0313** | **0.0185** | **0.0212** |
> | test | popularity | 0.0075 | 0.0083 | 0.0042 | 0.0045 |
> | test | ALS | 0.0074 | 0.0099 | 0.0046 | 0.0054 |
> | test | Semantic-ID Transformer | 0.0067 | 0.0090 | 0.0044 | 0.0051 |
> | test | **ALS + reranker** | **0.0114** | **0.0136** | **0.0084** | **0.0091** |
>
> Two things stand out, and one of them bothered me for a while.
>
> The reranker is the clear winner — 35–50% over the best retrieval-only tier on every
> metric, both splits. That is the expected shape of a two-stage system: retrieval is cheap
> and wide, ranking spends the compute where it counts.
>
> The other is that the three retrieval tiers are all at the same floor, and the generative
> model — the whole point of the post — does no better than counting what's popular. Is it
> broken? I spent a while on that question, and the answer turned out to be no. It's the
> split.

---

## 8. NEW SECTION — "Measuring it the way the papers do"

> ## Measuring it the way the papers do
>
> Look again at what the time split asks: from a median of *four* reviews written before
> November 2019, predict which games this user reviews in 2021–2023. The median gap between
> a user's last training interaction and their first test interaction is 1,397 days, and
> 72% of the test items never appear in training. No collaborative model can retrieve an
> item it has never seen, and no model of any kind is good at four-year-ahead prediction
> from four data points. Everything collapses to the same floor, and the floor is
> "recommend what's popular".
>
> TIGER and SASRec — and essentially every sequential-recommendation paper — measure
> something different: **leave-one-out next-item prediction.** For each user, the last item
> is the test target, the second-to-last is the validation target, and the context is
> everything before. The context ends right before the target and the target almost always
> exists in training (0.1% cold instead of 72%). Same data, different question, and the
> numbers are an order of magnitude apart.
>
> So I added that protocol as a second track, and with it the baseline the papers actually
> compare against: **SASRec** (Kang & McAuley 2018), which is the same causal Transformer
> over *raw item IDs* instead of Semantic IDs, scoring the next item by a dot product with
> the item table. It's the cleanest possible control for the question the post is really
> about — does writing items out of content-derived digits beat looking them up by ID?
>
> All 94,762 users, both splits, each user's own history filtered from every model's list:
>
> | split | model | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
> |---|---|---|---|---|---|
> | val | popularity | 0.0151 | 0.0268 | 0.0099 | 0.0136 |
> | val | ALS | 0.0537 | 0.0833 | 0.0350 | 0.0446 |
> | val | **SASRec** | **0.0630** | **0.0958** | **0.0415** | **0.0521** |
> | val | Semantic-ID Transformer | 0.0482 | 0.0765 | 0.0315 | 0.0406 |
> | test | popularity | 0.0137 | 0.0249 | 0.0091 | 0.0126 |
> | test | ALS | 0.0372 | 0.0569 | 0.0243 | 0.0307 |
> | test | **SASRec** | **0.0520** | **0.0792** | **0.0347** | **0.0434** |
> | test | Semantic-ID Transformer | 0.0443 | 0.0690 | 0.0292 | 0.0372 |
>
> First: the generative model works. Under the paper's protocol it reaches Recall@10 of
> 0.069–0.077, nearly three times the popularity floor and in the range TIGER reports on
> its own datasets (0.065 on Amazon Beauty). The earlier table was measuring a task nothing
> here can do well, not a broken model.
>
> Second: it does *not* beat SASRec on this dataset. SASRec is ahead by about 15% on every
> metric, where TIGER's paper reports the opposite. My list of suspects, cheapest to test
> first: the Transformer was still improving when I stopped (TIGER trains for ~200K steps;
> I did a few thousand); the RQ-VAE only uses 29% of its first-level codebook, so the
> Semantic IDs are a weaker partition than they should be; the model is decoder-only where
> TIGER uses an encoder–decoder; and Video Games has a heavy head of blockbuster titles and
> consoles that an ID-based model can simply memorise, while the content-based model has to
> reach them through shared digits. Each of those is a concrete experiment, and that's the
> point of having the table.
>
> `[COLAB: tutorial_05 — load the trained checkpoints and reproduce this table]`

---

## 9. NEW SECTION — "What I'd do next" + takeaways (closing)

> ## What I'd do next
>
> In order: train the Semantic-ID model to a loss plateau and re-run the table (the
> cheapest test of the SASRec gap); retrain the RQ-VAE on title + brand + category + price
> the way TIGER does, and check codebook utilisation before anything else; try a
> length-normalised beam score, since the current one favours frequent digit prefixes; and
> put the reranker on top of SASRec and the Transformer, not just ALS.
>
> ## Takeaways
>
> The evaluation protocol is part of the result. Same data, same models, and the ordering
> of the tiers flipped between "predict next year" and "predict the next item". Neither is
> wrong; they answer different questions, and a post that only shows one of them is hiding
> something.
>
> A reranker trained on the wrong candidates learns the wrong thing, confidently. The
> injection bug produced a model with sensible feature importances and a plausible score,
> and it was learning the exact inverse of what it should. Probe your model's response to
> each feature; it takes five minutes.
>
> Generative retrieval from content-derived IDs is real: on the standard protocol it lands
> where the papers say it should, and it can name items nobody has bought. On this dataset
> it still loses to a well-trained ID-based Transformer, and the gap has a shortlist of
> known causes rather than a mystery. That's a better place to end than a win I couldn't
> explain.
>
> Code, data pipeline and every table in this post:
> [github.com/juanmigutierrez/generative-recommendation-engine](https://github.com/juanmigutierrez/generative-recommendation-engine).
> Each section has an "Open in Colab" notebook that runs a small version of the same step
> in under ten minutes.

---

## 10. Sources footer (replace the existing one)

> *Sources: Rajput et al. 2023, TIGER — [arXiv:2305.05065](https://arxiv.org/abs/2305.05065)
> · Zhai et al. 2024, HSTU, "Actions Speak Louder than Words" —
> [arXiv:2402.17152](https://arxiv.org/abs/2402.17152) (the "retrieval as generation"
> framing; not implemented here) · Kang & McAuley 2018, SASRec —
> [arXiv:1808.09781](https://arxiv.org/abs/1808.09781) · van den Oord et al. 2017, VQ-VAE —
> [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) · Vaswani et al. 2017 —
> [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · Ni et al. 2021, Sentence-T5 —
> [arXiv:2108.08877](https://arxiv.org/abs/2108.08877) · Hu, Koren & Volinsky 2008 —
> [Collaborative Filtering for Implicit Feedback Datasets](https://www.semanticscholar.org/paper/Collaborative-Filtering-for-Implicit-Feedback-Hu-Koren/184b7281a87ee16228b24716ca02b29519d52eb5)
> · Ke et al. 2017, LightGBM —
> [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
> · Hou et al. 2024, Amazon Reviews 2023 — [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)
> · Libraries: implicit, sentence-transformers, JAX, LightGBM.*

---

## 11. Typo / wording sweep (search for these strings)

- "the7 codebooks" → "the codebooks"
- "minizamization" → "minimization"; "proyect" → "project"
- "25⁶³" → "256³"
- "Now if we fixed from the minizamization (consider as constant) the other expression"
  → "If we hold one side fixed, the derivative of the loss with respect to the other has a
  closed-form zero (derivation omitted):"
- "for recommend similar objects" — removed by edit 1
- "What YouTube, Netflix and Meta Are Moving To": make sure the Netflix link points at
  their foundation-model-for-recommendations post, not a generic page; YouTube → the
  Semantic-ID papers from Google; Meta → HSTU.
- The Recall@5 / NDCG@5 example figures use K = 5 while the text says K = 10 and 20 —
  fine, but say "the same idea at K = 5" in the caption so it doesn't look inconsistent.
