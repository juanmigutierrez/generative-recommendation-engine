"""
Generates the five tutorial notebooks (notebooks/tutorial_0X_*.ipynb) that accompany the
blog post. Each one is self-contained on Colab: it clones the repo, downloads the
pre-computed artifacts from the GitHub release, and runs a small, interactive version of
one section of the post in well under ten minutes on a free T4.

    python notebooks/_build_tutorials.py

Set TUTORIAL_QUICK=1 in the environment when executing the notebooks headlessly (CI /
local smoke test): every training loop and sample size shrinks to a few seconds.
"""
import json
import os

REPO_URL = "https://github.com/juanmigutierrez/generative-recommendation-engine"
RELEASE_TAG = "v1.0-artifacts"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NOTEBOOK_FILES = {
    1: "tutorial_01_data_and_baselines.ipynb",
    2: "tutorial_02_semantic_ids.ipynb",
    3: "tutorial_03_generative_retrieval.ipynb",
    4: "tutorial_04_ranking.ipynb",
    5: "tutorial_05_evaluation.ipynb",
}


IMG_BASE = "https://raw.githubusercontent.com/juanmigutierrez/generative-recommendation-engine/main/blog/Images/"


def img(name, caption=""):
    """Markdown for one of the explanatory figures from the post (served from the repo)."""
    cap = f"\n\n*{caption}*" if caption else ""
    return f"![{caption or name}]({IMG_BASE}{name}){cap}"


def badge(n):
    f = NOTEBOOK_FILES[n]
    url = f"https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/{f}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def setup_cell(files):
    files_str = ", ".join(f'"{f}"' for f in files)
    return f'''#@title Setup — clone the repo, install deps, download the pre-computed artifacts (~1 min)
import os, sys, urllib.request
REPO_URL = "{REPO_URL}"
RELEASE  = REPO_URL + "/releases/download/{RELEASE_TAG}"
QUICK    = os.environ.get("TUTORIAL_QUICK") == "1"   # tiny sizes for headless smoke tests

if not os.path.exists("backend"):
    if not os.path.exists("generative-recommendation-engine"):
        !git clone -q {{REPO_URL}}
    %cd generative-recommendation-engine
if "google.colab" in sys.modules:
    !pip install -q implicit lightgbm pyarrow ipywidgets 2>&1 | tail -1
os.makedirs("data/processed", exist_ok=True)

def fetch(*names):
    """Download an artifact from the GitHub release unless it is already on disk."""
    for n in names:
        p = os.path.join("data", "processed", n)
        if not os.path.exists(p):
            print("downloading", n, "...")
            urllib.request.urlretrieve(f"{{RELEASE}}/{{n}}", p)

fetch({files_str})
for p in ["backend", "backend/scripts"]:
    if p not in sys.path: sys.path.insert(0, p)

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
pd.set_option("display.max_colwidth", 90)
P = os.path.join("data", "processed")
print("ready")'''


def nb(cells, title):
    out = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                     "language_info": {"name": "python"},
                                     "colab": {"provenance": [], "name": title}},
           "nbformat": 4, "nbformat_minor": 0}
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            c["execution_count"] = None
            c["outputs"] = []
        out["cells"].append(c)
    return out


# ============================================================================ 1
def build_01():
    cells = [("markdown", f"""# Tutorial 1 — The data, the split, and the two baselines

{badge(1)}

Companion to the *Dataset*, *Baseline Popularity*, *ALS* and *Scoring the models* sections of
the post. In ~5 minutes on a free Colab runtime this notebook:

1. loads the Amazon Reviews 2023 **Video_Games 5-core** data the whole project uses,
2. lets you pick a user and read their history,
3. shows why the split is *by time* and what that does to cold-start,
4. lets you play with Recall@K and NDCG@K on toy lists until they make sense,
5. fits the popularity and ALS baselines and lets you compare their recommendations for any user against what that user actually did next.

Everything runs against the project's own code (`backend/models/*.py`), not a re-implementation."""),
    ("code", setup_cell(["train.parquet", "val.parquet", "test.parquet", "item_catalog.parquet",
                         "val_targets.parquet", "test_targets.parquet"])),
    ("markdown", "## 1. The data\n\nOne row per (user, item, timestamp). `description` is the product title — the only content signal used in the whole project."),
    ("code", '''train = pd.read_parquet(f"{P}/train.parquet")
val   = pd.read_parquet(f"{P}/val.parquet")
test  = pd.read_parquet(f"{P}/test.parquet")
items = pd.read_parquet(f"{P}/item_catalog.parquet").set_index("item_id")
title = items["description"]

n_users = pd.concat([train, val, test])["user_id"].nunique()
print(f"{len(train)+len(val)+len(test):,} interactions, {n_users:,} users, {len(items):,} items")
per_user = pd.concat([train, val, test]).groupby("user_id").size()
print(f"interactions per user: median {per_user.median():.0f}, mean {per_user.mean():.1f}, max {per_user.max()}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(per_user.clip(upper=40), bins=36, color="#2a78d6")
ax.set_xlabel("interactions per user (clipped at 40)"); ax.set_ylabel("users"); ax.set_title("Every user has ≥ 5 by construction (5-core), most have 5–8")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.show()'''),
    ("markdown", "### Pick a user and read their history\n\nThe dropdown holds 30 random users with a longer history so there is something to read. `split` tells you which side of the time cut each review falls on."),
    ("code", '''allrows = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
allrows["timestamp"] = pd.to_datetime(allrows["timestamp"])
rng = np.random.RandomState(0)
long_users = per_user[per_user.between(8, 25)].index
sample_users = sorted(rng.choice(long_users, 30, replace=False).tolist())

def show_history(user_id):
    h = allrows[allrows.user_id == user_id].sort_values("timestamp")
    out = pd.DataFrame({"when": h["timestamp"].dt.date.values, "split": h["split"].values,
                        "item_id": h["item_id"].values, "title": title.loc[h["item_id"]].str.slice(0, 80).values})
    display(out.reset_index(drop=True))

interact(show_history, user_id=widgets.Dropdown(options=sample_users, description="user"));'''),
    ("markdown", "### How one user is scored\n\n" + img("ranking_models.png", "The exam every model takes: history before the cutoff → top-K guesses → what actually happened → Recall@K and NDCG@K.")),
    ("markdown", """## 2. Splitting by time, not at random

A random split lets a model train on a 2022 review and be tested on one from 2005 — it has seen the future. So the cut is by date: the last 8% of the dataset's time span is test, the 8% before that is val, everything earlier is train. The price is that val/test contain users the model has never seen (cold users) and items that did not exist yet (cold items). This is the honest production question — and, as the post explains, a *much* harder one than the papers' leave-one-out protocol (Tutorial 5)."""),
    ("code", '''for name, df in [("train", train), ("val", val), ("test", test)]:
    t = pd.to_datetime(df["timestamp"])
    print(f"{name:<6} {t.min().date()} → {t.max().date()}   {len(df):>8,} rows   {df.user_id.nunique():>7,} users")

train_users, train_items = set(train.user_id), set(train.item_id)
for name, df in [("val", val), ("test", test)]:
    cold_u = (~df.user_id.isin(train_users)).groupby(df.user_id).first().mean()
    cold_i = (~df.item_id.isin(train_items)).mean()
    print(f"{name}: {cold_u:.0%} of users never appear in train; {cold_i:.0%} of rows are items never seen in train")'''),
    ("markdown", """## 3. The metrics, on toy lists

Both metrics take a ranked list and the set of items the user actually interacted with next.

**Recall@K — did we find it?** Of the items the user actually interacted with, what fraction appear anywhere in the top-K list? Order inside the list doesn't matter.

$$\\mathrm{Recall@}K = \\frac{|\\, \\mathrm{top}K \\cap \\mathrm{relevant} \\,|}{|\\, \\mathrm{relevant} \\,|}$$

**NDCG@K — did we find it early?** Each hit gets a weight that shrinks with its position, and the total is divided by the best score possible so the result lands in [0, 1]:

$$\\mathrm{DCG@}K = \\sum_{i=1}^{K} \\frac{\\mathbb{1}[\\text{item}_i \\in \\mathrm{relevant}]}{\\log_2(i+1)}, \\qquad \\mathrm{NDCG@}K = \\frac{\\mathrm{DCG@}K}{\\mathrm{IDCG@}K}$$

""" + img("eval_example_recall.png", "Recall@5 on three lists: only presence counts.") + "\n\n" + img("eval_example_ndcg.png", "NDCG@5: same hit, different position, different score.") + """

Edit the two lists below and re-run — the functions are the project's own (`backend/models/metrics.py`)."""),
    ("code", '''from models.metrics import recall_at_k, ndcg_at_k

relevant = {"E", "M"}                              #@param {type:"raw"}
ranked   = ["G", "E", "H", "J", "K", "M", "N"]     #@param {type:"raw"}

for k in (3, 5, 7):
    print(f"K={k}: recall@{k} = {recall_at_k(ranked, set(relevant), k):.2f}   ndcg@{k} = {ndcg_at_k(ranked, set(relevant), k):.2f}")
print("\\nposition weights used by NDCG (1/log2(pos+1)):", [round(float(1/np.log2(i+2)), 2) for i in range(7)])'''),
    ("code", '''@interact(hit_position=widgets.IntSlider(1, 1, 10, description="hit at #"))
def ndcg_vs_position(hit_position):
    ranked = [f"x{i}" for i in range(10)]; ranked[hit_position - 1] = "E"
    print(f"one relevant item at position {hit_position}:  recall@10 = {recall_at_k(ranked, {'E'}, 10):.2f}   ndcg@10 = {ndcg_at_k(ranked, {'E'}, 10):.2f}")'''),
    ("markdown", """## 4. Baseline 1 — popularity

Count how often each item appears in train, sort, cross off what the user already has, return the top-K. Everyone gets the same list.

$$\\mathrm{pop}(i) = \\sum_{u} \\mathbb{1}\\big[(u, i) \\in \\mathcal{D}\\big], \\qquad \\mathrm{rec}(u) = \\operatorname{top\\text{-}K}_{\\,i \\notin H_u} \\; \\mathrm{pop}(i)$$

""" + img("popularity_model_histogram_indigo.png", "Count → sort → remove what the user already has → return the top-K. The only personalisation is the crossing-off.")),
    ("code", '''from models.popularity import PopularityRecommender
pop = PopularityRecommender().fit(train)
print("the ten most popular items in train:")
for i in pop.ranked_items[:10]:
    print(f"  {title[i][:90]}")'''),
    ("markdown", """## 5. Baseline 2 — ALS

Put users in rows and items in columns; a 1 where we saw an interaction, a blank where we didn't. Call it $R$. Recommending is filling in the blanks. ALS says $R$ is (approximately) the product of two thin matrices — a row $\\mathbf{x}_u$ per user and a row $\\mathbf{y}_i$ per item, both of length $k$ — and the prediction for any cell is their dot product:

$$\\hat r_{ui} = \\mathbf{x}_u^{\\top} \\mathbf{y}_i$$

""" + img("als_matrices_R_X_Y.png", "R ≈ X · Yᵀ. The dashed cell (user 1, item 2) was never observed; the model predicts 0.45 for it.") + """

The numbers are chosen to make the predictions match the cells we know, with a penalty on their size so the model can't overfit by making them huge (regularisation, $\\lambda$). And because a blank means *not observed*, not *disliked*, each cell gets a confidence weight — trust the 1s a lot, the blanks only a little (Hu, Koren & Volinsky 2008):

$$c_{ui} = 1 + \\alpha\\, r_{ui}, \\qquad p_{ui} = \\mathbb{1}[r_{ui} > 0]$$

$$\\min_{X,Y} \\sum_{u,i} c_{ui}\\,\\big(p_{ui} - \\mathbf{x}_u^{\\top}\\mathbf{y}_i\\big)^2 + \\lambda\\Big(\\sum_u \\|\\mathbf{x}_u\\|^2 + \\sum_i \\|\\mathbf{y}_i\\|^2\\Big)$$

Solving for $X$ and $Y$ together is hard, so freeze one and the other has a closed-form least-squares solution; alternate until it stops improving — hence the name:

$$\\mathbf{x}_u = (Y^{\\top} C^u Y + \\lambda I)^{-1} Y^{\\top} C^u \\mathbf{p}_u, \\qquad \\mathbf{y}_i = (X^{\\top} C^i X + \\lambda I)^{-1} X^{\\top} C^i \\mathbf{p}_i$$

""" + img("als_full_breakdown_diagram.png", "The three terms of the loss on the toy matrix: confidence, error, regularisation.") + """

The `implicit` library implements exactly this. 64 factors, $\\alpha = 40$, 15 alternations — about 30 s."""),
    ("code", '''from models.als import ALSRecommender
import time
n_items = len(items); n_users_total = int(allrows.user_id.max()) + 1
t0 = time.time()
als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(train, n_users_total, n_items)
print(f"ALS fit in {time.time()-t0:.0f}s")'''),
    ("markdown", "### What ALS learned: nearest items in its latent space\n\nNo content was used — these neighbours come purely from who-bought-what."),
    ("code", '''item_vecs = als.model.item_factors
if hasattr(item_vecs, "to_numpy"): item_vecs = item_vecs.to_numpy()   # GPU build returns a wrapper
item_vecs = np.asarray(item_vecs); item_norm = item_vecs / (np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-9)
popular_examples = [i for i in pop.ranked_items[:300] if len(title[i]) < 70][:25]

def als_neighbours(item_id):
    sims = item_norm @ item_norm[item_id]
    top = np.argsort(-sims)[1:8]
    print("query:", title[item_id]); print()
    for j in top: print(f"  {sims[j]:.3f}  {title[j][:85]}")

interact(als_neighbours, item_id=widgets.Dropdown(options=[(title[i][:60], i) for i in popular_examples], description="item"));'''),
    ("markdown", """## 6. Compare the two for one user

For a user with training history, both models produce a top-10; the *val* items are what the user actually reviewed next. Hits are marked. Try several users — you will see how rarely either model hits, which is the point the post makes about this split."""),
    ("code", '''val_targets = {r.user_id: set(r.item_ids) for r in pd.read_parquet(f"{P}/val_targets.parquet").itertuples()}
warm_val_users = [u for u in val_targets if u in train_users and per_user[u] >= 8]
demo_users = sorted(rng.choice(warm_val_users, 40, replace=False).tolist())

def compare(user_id, K=10):
    truth = val_targets[user_id]
    print("actually reviewed next (val):")
    for i in truth: print("   •", title[i][:85])
    for name, fn in [("popularity", pop.recommend), ("ALS", als.recommend)]:
        recs = fn(user_id, K)
        hits = sum(i in truth for i in recs)
        print(f"\\n{name} top-{K}  —  recall@{K} = {hits/len(truth):.2f}, ndcg@{K} = {ndcg_at_k(recs, truth, K):.2f}")
        for r, i in enumerate(recs, 1):
            print(f"  {'✔' if i in truth else ' '} {r:>2}. {title[i][:80]}")

interact(compare, user_id=widgets.Dropdown(options=demo_users, description="user"), K=widgets.IntSlider(10, 5, 20, 5));'''),
    ("markdown", "## 7. Score both on a sample of val users\n\nThe post's baseline table, reproduced on a random 2,000-user sample (cold users fall back to the popularity list for ALS, exactly as in the project's `run_baselines.py`)."),
    ("code", '''from models.metrics import evaluate
n_eval = 300 if QUICK else 2000
eval_users = rng.choice(sorted(val_targets), n_eval, replace=False)
targets = {u: val_targets[u] for u in eval_users}
als_fn = lambda u, k: als.recommend(u, k) if not als.is_cold(u) else pop.recommend(u, k)
rows = {"popularity": evaluate(pop.recommend, targets), "als": evaluate(als_fn, targets)}
table = pd.DataFrame(rows).T[["recall@10", "recall@20", "ndcg@10", "ndcg@20"]].round(4)
display(table)
print("Low numbers everywhere — and ALS barely beats counting. Tutorial 5 shows the same models under the papers' protocol, where the picture changes completely.")'''),
    ("markdown", "**Next:** [Tutorial 2 — Semantic IDs](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_02_semantic_ids.ipynb): give every item a code built from its title."),
    ]
    return nb(cells, "Tutorial 1 — data & baselines")


# ============================================================================ 2
def build_02():
    cells = [("markdown", f"""# Tutorial 2 — Semantic IDs: an identity built from content

{badge(2)}

Companion to the *Semantic IDs* section. Here you will:

1. see what the Sentence-T5 embedding already knows about products (type any title fragment and get its neighbours),
2. see the anisotropy problem that silently collapsed the first RQ-VAE, and the fix,
3. **train the RQ-VAE yourself** (~1 min on GPU) with the project's JAX code and watch codebook utilisation and collisions,
4. look up any item's Semantic ID and browse everything that shares its first digit — the "category the model discovered on its own".

A GPU runtime is recommended (Runtime → Change runtime type → T4)."""),
    ("code", setup_cell(["item_catalog.parquet", "item_embeddings.npy", "item_embeddings_index.parquet",
                         "semantic_ids.parquet"])),
    ("markdown", img("genai_recommender.png", "The whole idea: each item is converted to a short sequence of discrete tokens (its Semantic ID); a generative model writes the next one; a lookup turns it back into an item.")),
    ("markdown", """## 1. Step 1 of the post: the title becomes 768 numbers

$$\\mathbf{x} = \\text{SentenceT5}(\\text{title}) \\in \\mathbb{R}^{768}$$

The project embedded all 25,612 titles with `sentence-t5-base` (see `backend/scripts/build_item_embeddings_local.py`). The matrix is downloaded above; the optional cell after it re-embeds a few titles live so you can check the two agree."""),
    ("code", '''items = pd.read_parquet(f"{P}/item_catalog.parquet").set_index("item_id")
title = items["description"]
emb = np.load(f"{P}/item_embeddings.npy").astype("float32")
idx = pd.read_parquet(f"{P}/item_embeddings_index.parquet")
assert (idx["item_id"].values == np.arange(len(emb))).all()
print("embedding matrix:", emb.shape)
unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)

def neighbours(query, n=6):
    """Find catalog titles containing `query`, take the first, list its nearest neighbours by cosine."""
    hits = title[title.str.contains(query, case=False, regex=False)]
    if len(hits) == 0: print("no title contains", repr(query)); return
    q = hits.index[0]
    sims = unit @ unit[q]; top = np.argsort(-sims)[1:n+1]
    print(f"query item: {title[q][:100]}\\n")
    for j in top: print(f"  {sims[j]:.3f}  {title[j][:95]}")

interact(neighbours, query=widgets.Text(value="Jeecoo V20", description="title has"), n=widgets.IntSlider(6, 3, 12));'''),
    ("code", '''#@title Optional — embed a few titles live with sentence-t5-base and compare (needs ~1 min to download the model)
RUN_LIVE_EMBED = False  #@param {type:"boolean"}
if RUN_LIVE_EMBED and not QUICK:
    !pip install -q sentence-transformers 2>&1 | tail -1
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")
    sample = [0, 3, 100, 2000, 9000]
    live = model.encode([title[i] for i in sample], convert_to_numpy=True)
    live /= np.linalg.norm(live, axis=1, keepdims=True)
    for i, v in zip(sample, live):
        print(f"cosine(live, precomputed) = {float(v @ unit[i]):.4f}   {title[i][:70]}")'''),
    ("markdown", """## 2. The trap: sentence embeddings are anisotropic

Every Sentence-T5 vector shares one big common direction. Two *random* products look 76% similar. Feed that straight into a VQ codebook initialised near the origin and every item snaps to the same code — the first RQ-VAE run in this project did exactly that. Standardising each dimension (subtract mean, divide by std) removes the shared direction; the project also seeds the codebooks with k-means on real encoder outputs."""),
    ("code", '''rng = np.random.RandomState(0)
a, b = rng.randint(0, len(emb), 5000), rng.randint(0, len(emb), 5000)
print(f"cosine similarity of RANDOM item pairs, raw embeddings:        {np.mean(np.sum(unit[a]*unit[b], axis=1)):.3f}")
emb_std = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
unit_std = emb_std / np.linalg.norm(emb_std, axis=1, keepdims=True)
print(f"cosine similarity of RANDOM item pairs, after standardising:   {np.mean(np.sum(unit_std[a]*unit_std[b], axis=1)):.3f}")'''),
    ("markdown", """## 3. Steps 2–4: train the RQ-VAE

**Step 2: compress it.** A small encoder $E$ squeezes 768 numbers down to 32:

$$\\mathbf{z} = E(\\mathbf{x}), \\qquad \\mathbf{x} \\in \\mathbb{R}^{768}, \\; \\mathbf{z} \\in \\mathbb{R}^{32}$$

**Step 3: snap it to a codebook.** Instead of keeping $\\mathbf{z}$ as 32 free numbers, snap it to the nearest of 256 learned reference vectors $C_1$; the first digit is the index of that vector. One snap is coarse, so take what's left over (the residual) and snap *that* to a second codebook, then a third:

$$c_1 = \\arg\\min_k \\|\\mathbf{z} - C_1[k]\\|^2, \\qquad \\mathbf{r}_1 = \\mathbf{z} - C_1[c_1]$$
$$c_2 = \\arg\\min_k \\|\\mathbf{r}_1 - C_2[k]\\|^2, \\qquad \\mathbf{r}_2 = \\mathbf{r}_1 - C_2[c_2], \\qquad c_3 = \\arg\\min_k \\|\\mathbf{r}_2 - C_3[k]\\|^2$$

The Semantic ID is $(c_1, c_2, c_3)$ — coarse, finer, finest — and the compressed vector is rebuilt by adding the three snapped vectors back together:

$$\\hat{\\mathbf{z}} = C_1[c_1] + C_2[c_2] + C_3[c_3]$$

Three codebooks of 256 give $256^3 \\approx 16.7$ million possible IDs from only 768 learned vectors.

**Step 4: train it.** A decoder $D$ tries to rebuild the original 768 numbers from $\\hat{\\mathbf{z}}$. The loss makes that rebuild accurate while pulling the codebook vectors toward the data and the encoder toward the codebooks; $\\mathrm{sg}[\\cdot]$ is *stop-gradient* — the middle term moves only the codebooks, the last one only the encoder:

$$L = \\|\\mathbf{x} - D(\\hat{\\mathbf{z}})\\|^2 + \\|\\mathrm{sg}[\\mathbf{z}] - \\hat{\\mathbf{z}}\\|^2 + \\beta\\,\\|\\mathbf{z} - \\mathrm{sg}[\\hat{\\mathbf{z}}]\\|^2, \\qquad \\beta = 0.25$$

""" + img("RQ_VAE.png", "RQ-VAE: encode, quantise level by level on the residual, decode.") + """

That is what `backend/models/rqvae.py` implements: encoder 768 → 256 → 32, three codebooks of 256, decoder back to 768, straight-through estimator. The post's run used 400 epochs; 150 is enough to see the behaviour. On a T4 this is about a minute."""),
    ("code", '''import jax, jax.numpy as jnp, time
from models import rqvae
EPOCHS = 4 if QUICK else 150   #@param {type:"integer"}
HIDDEN, LATENT, LEVELS, CODES, BATCH, LR, BETA = 256, 32, 3, 256, 1024, 2e-3, 0.25

x = jnp.array(emb_std)
key = jax.random.PRNGKey(42); k_init, k_cb = jax.random.split(key)
params = rqvae.init_params(k_init, emb.shape[1], HIDDEN, LATENT, LEVELS, CODES)
params["codebooks"] = rqvae.kmeans_init_codebooks(k_cb, rqvae.encode(params, x), LEVELS, CODES)   # the k-means fix
opt_init, opt_update = rqvae.make_adam(lr=LR); opt_state = opt_init(params)
step = jax.jit(jax.value_and_grad(lambda p, b: rqvae.forward(p, b, beta=BETA)[0]))

n = len(emb); nb_ = n // BATCH; hist = []; t0 = time.time(); prng = np.random.RandomState(0)
for ep in range(EPOCHS):
    perm = prng.permutation(n); tot = 0.0
    for b in range(nb_):
        loss, g = step(params, x[perm[b*BATCH:(b+1)*BATCH]]); params, opt_state = opt_update(g, opt_state, params); tot += float(loss)
    hist.append(tot / nb_)
    if ep % 25 == 0 or ep == EPOCHS - 1: print(f"epoch {ep:3d}  loss {hist[-1]:.4f}  ({time.time()-t0:.0f}s)")

fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(hist, color="#2a78d6"); ax.set_xlabel("epoch"); ax.set_ylabel("recon + codebook + β·commitment")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.show()'''),
    ("markdown", "### What came out: utilisation and collisions\n\nHow many of the 256 codes does each level actually use, and how many items end up with an identical 3-digit code (and therefore need the 4th, tie-break digit)?"),
    ("code", '''from collections import Counter
codes = np.array(rqvae.get_codes(params, x))
for l in range(LEVELS):
    print(f"level {l+1}: {len(set(codes[:, l]))}/{CODES} codes used")
cnt = Counter(map(tuple, codes)); colliding = sum(c for c in cnt.values() if c > 1)
print(f"{len(cnt):,} unique 3-digit codes for {n:,} items; {colliding/n:.0%} of items share a code with another item → get a tie-break digit")

ref = pd.read_parquet(f"{P}/semantic_ids.parquet").set_index("item_id")
print(f"\\n(the project's 400-epoch run: level-1 used {ref.sid_level_1.nunique()}/256, {(ref.groupby(['sid_level_1','sid_level_2','sid_level_3']).size()>1).sum():,} colliding codes)")'''),
    ("markdown", """## 4. Browse the Semantic IDs

Type part of a title: you get the item's code from the model you just trained, and the other items that share its **first digit** — a family the model was never told about. Then narrow to items sharing the first *two* digits."""),
    ("code", '''def browse(query, shared_digits=1, n=12):
    hits = title[title.str.contains(query, case=False, regex=False)]
    if len(hits) == 0: print("no title contains", repr(query)); return
    q = hits.index[0]; c = codes[q]
    print(f"{title[q][:90]}\\n→ Semantic ID (this run): {tuple(int(v) for v in c)}\\n")
    mask = np.all(codes[:, :shared_digits] == c[:shared_digits], axis=1)
    fam = np.where(mask)[0]
    print(f"{len(fam):,} items share the first {shared_digits} digit(s). A sample:")
    for j in rng.choice(fam, min(n, len(fam)), replace=False): print(f"  {tuple(int(v) for v in codes[j])}  {title[j][:80]}")

interact(browse, query=widgets.Text(value="gaming keyboard", description="title has"),
         shared_digits=widgets.IntSlider(1, 1, 3, description="shared digits"), n=widgets.IntSlider(12, 5, 25));'''),
    ("markdown", """## 5. The 2-D toy example from the post, in code

`z = (0.72, 0.31)`, codebook 1 = {(0.2, 0.8), (0.8, 0.2), (0.5, 0.5)}, codebook 2 = {(−0.1, 0.1), (0.1, −0.1), (0, 0)}."""),
    ("code", '''z = np.array([0.72, 0.31]); C1 = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]]); C2 = np.array([[-0.1, 0.1], [0.1, -0.1], [0.0, 0.0]])
c1 = int(np.argmin(((C1 - z) ** 2).sum(1))); r1 = z - C1[c1]
c2 = int(np.argmin(((C2 - r1) ** 2).sum(1))); z_hat = C1[c1] + C2[c2]
print(f"c1 = {c1} (nearest {C1[c1]}), residual {r1.round(2)}, c2 = {c2}, reconstruction {z_hat.round(2)} vs z {z} → error {(z - z_hat).round(2)}")'''),
    ("markdown", "**Next:** [Tutorial 3 — Generative retrieval](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_03_generative_retrieval.ipynb): train a Transformer to *write* the next Semantic ID."),
    ]
    return nb(cells, "Tutorial 2 — Semantic IDs")


# ============================================================================ 3
def build_03():
    cells = [("markdown", f"""# Tutorial 3 — Generative retrieval: writing the next item

{badge(3)}

Companion to the *Generative retrieval* section. Here you will:

1. turn a real user's history into the token stream the model reads,
2. **train the Semantic-ID Transformer on a subset** for a few minutes (or load the fully trained checkpoint — both paths are here),
3. run constrained beam search step by step and watch the model pick a *family* of items with the first digit before pinning one down,
4. compare its top-10 against the popularity list for any user, with the true next item marked.

Use a GPU runtime. The project's own JAX code is used throughout (`backend/models/transformer.py`, `backend/scripts/evaluate_retrieval.py`)."""),
    ("code", setup_cell(["item_catalog.parquet", "item_tokens.parquet", "loo_train_sequences.parquet",
                         "loo_val_targets.parquet", "transformer_train_sequences_loo.npz",
                         "transformer_vocab_meta_loo.json", "transformer_checkpoint_loo.pkl"])),
    ("markdown", img("generative_retrieval.png", "Items → Semantic IDs → a Transformer that generates the next ID → lookup back to an item.")),
    ("markdown", """## 1. Step 1: the history becomes a sentence

Take a user's purchases in time order and replace each item by its four digits (three Semantic ID levels plus the tie-break digit), plus one token at the front that identifies the user:

$$(i_1, i_2, \\ldots, i_T) \\longrightarrow \\big(u,\\; c^{(1)}_1, c^{(1)}_2, c^{(1)}_3, c^{(1)}_4,\\; c^{(2)}_1, \\ldots, c^{(T)}_4\\big)$$

Nothing in that stream says "product" or "user". It's just tokens, and tokens are what language models eat. This is the leave-one-out training data (`build_semantic_sequences.py --loo`): the last two items of every user are held out for evaluation.

""" + img("step5_step6_flow.png", "Step 5 builds the dictionary (Semantic IDs); Step 6 writes sentences in it.")),
    ("code", '''import json, pickle, jax, jax.numpy as jnp, time
from models import transformer as tx
import evaluate_retrieval as ev

items = pd.read_parquet(f"{P}/item_catalog.parquet").set_index("item_id"); title = items["description"]
item_tokens = pd.read_parquet(f"{P}/item_tokens.parquet"); item_to_tok = {r.item_id: [r.tok1, r.tok2, r.tok3, r.tok4] for r in item_tokens.itertuples(index=False)}
meta = json.load(open(f"{P}/transformer_vocab_meta_loo.json"))
seqs = {r.user_id: list(r.item_ids) for r in pd.read_parquet(f"{P}/loo_train_sequences.parquet").itertuples()}
val_target = {r.user_id: r.item_ids[0] for r in pd.read_parquet(f"{P}/loo_val_targets.parquet").itertuples()}
data = np.load(f"{P}/transformer_train_sequences_loo.npz"); tokens_all, user_ids_all = data["tokens"], data["user_ids"]
print("vocab", meta["vocab_size"], "| pad", meta["pad_token"], "| max tokens", meta["max_len"], "| users", tokens_all.shape[0])

def user_token(u): return meta["item_vocab"] + int(u) % meta["user_buckets"]
def context_tokens(u, max_items=meta["max_items"] - 1):
    return [user_token(u)] + sum((item_to_tok[i] for i in seqs[u][-max_items:]), [])

rng = np.random.RandomState(1)
demo_users = sorted(rng.choice([u for u, s in seqs.items() if 4 <= len(s) <= 8], 40, replace=False).tolist())

def show_stream(user_id):
    print(f"user {user_id} — history ({len(seqs[user_id])} items):")
    for i in seqs[user_id]: print(f"   {tuple(item_to_tok[i])}  {title[i][:75]}")
    print("\\ntoken stream the model reads:", context_tokens(user_id))
    print("\\nheld-out next item (val target):", title[val_target[user_id]][:80], tuple(item_to_tok[val_target[user_id]]))

interact(show_stream, user_id=widgets.Dropdown(options=demo_users, description="user"));'''),
    ("markdown", """## 2. Step 2: train it — or load it

A decoder-only Transformer reads the stream and, at every position, predicts the next token from everything before it. Training maximises the probability of the token that actually came next — i.e. minimises cross-entropy:

$$\\min_{\\theta} \\; -\\sum_{n} \\log P_{\\theta}(t_{n+1} \\mid t_1, \\ldots, t_n)$$

That's the whole objective. No "similarity", no user–item matrix: the model learns which digits tend to follow which.

**Option A** trains a small model on a subset of users right here (a few minutes on a T4; the loss will still be high). **Option B** loads the checkpoint from the post (4 layers, d=128, 45 epochs on the full data). The rest of the notebook works with whichever you ran last."""),
    ("code", '''#@title Option A — train a small model on a subset (GPU: ~3 min)
TRAIN_HERE  = True      #@param {type:"boolean"}
N_USERS     = 20000     #@param {type:"integer"}
EPOCHS      = 6         #@param {type:"integer"}
D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH, LR, DROPOUT = 64, 4, 2, 256, 256, 1e-3, 0.1
if QUICK: N_USERS, EPOCHS = 2000, 1

if TRAIN_HERE:
    x = jnp.array(tokens_all[:N_USERS]); n = x.shape[0]
    params = tx.init_params(jax.random.PRNGKey(0), meta["vocab_size"], meta["max_len"], D_MODEL, N_HEADS, N_LAYERS, D_FF)
    opt_init, opt_update = tx.make_adam(lr=LR); opt_state = opt_init(params)
    step = jax.jit(jax.value_and_grad(lambda p, b, k: tx.loss_fn(p, b, meta["pad_token"], N_HEADS, DROPOUT, k)))
    key = jax.random.PRNGKey(1); prng = np.random.RandomState(0); hist = []; t0 = time.time()
    for ep in range(EPOCHS):
        perm = prng.permutation(n); tot = 0.0; nb_ = n // BATCH
        for b in range(nb_):
            key, sub = jax.random.split(key)
            loss, g = step(params, x[perm[b*BATCH:(b+1)*BATCH]], sub); params, opt_state = opt_update(g, opt_state, params); tot += float(loss)
        hist.append(tot / nb_); print(f"epoch {ep+1}  loss/token {hist[-1]:.3f}  ({time.time()-t0:.0f}s)")
    model_params, model_heads, model_name = params, N_HEADS, f"small model trained here ({N_USERS:,} users, {EPOCHS} epochs)"
    print("random-guess loss would be ln(vocab) =", round(float(np.log(meta["vocab_size"])), 2))'''),
    ("code", '''#@title Option B — load the fully trained checkpoint from the post
LOAD_CHECKPOINT = True  #@param {type:"boolean"}
if LOAD_CHECKPOINT:
    raw = pickle.load(open(f"{P}/transformer_checkpoint_loo.pkl", "rb"))
    model_params = jax.tree_util.tree_map(jnp.array, raw["params"]); model_heads = raw["config"]["n_heads"]
    model_name = f"checkpoint from the post ({raw['config']['n_layers']} layers, d={raw['config']['d_model']}, epoch {raw['epoch']}, loss {raw['history'][-1]:.2f})"
    fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(raw["history"], color="#2a78d6"); ax.set_xlabel("epoch"); ax.set_ylabel("loss per digit"); ax.set_title("training curve of the post's model")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.show()
print("active model:", model_name)'''),
    ("markdown", """## 3. Steps 3–4: generate, constrained to real items

At inference, feed the history and let the model write four more tokens. The probability of a full item is the product of its four digit probabilities, each conditioned on the digits written so far:

$$P(i \\mid h) = P(c_1 \\mid h)\\; P(c_2 \\mid h, c_1)\\; P(c_3 \\mid h, c_1, c_2)\\; P(c_4 \\mid h, c_1, c_2, c_3)$$

The first digit picks the broad family, the second narrows it, the third pins the item down, the fourth breaks ties. Not every 4-digit combination is a product, so generation is constrained to digits that lead to a real item, and beam search keeps the best partial IDs alive instead of greedily taking one digit at a time. The score of a finished candidate is its log-probability; the top-K by score (minus what the user already has) are the recommendations:

$$\\mathrm{score}(i) = \\sum_{j=1}^{4} \\log P(c_j \\mid h, c_{<j}), \\qquad \\mathrm{rec}(u) = \\operatorname{top\\text{-}K}_{\\,i \\in \\text{catalog},\\, i \\notin H_u} \\mathrm{score}(i)$$

The trie below is built from every catalog item's 4 digits. At each step the model only scores digits that can still complete a real item. Pick a user: you see the *families* the first digit points at, then the finished top-10 with the true next item marked."""),
    ("code", '''trie = ev.build_trie(item_tokens); ev.N_HEADS = model_heads; ev.BEAM_WIDTH = 30
level1_members = item_tokens.groupby("tok1")["item_id"].apply(list).to_dict()

def generate(user_id, K=10):
    ctx = context_tokens(user_id); seen = set(seqs[user_id]); truth = val_target[user_id]
    # digit 1: what families does the model reach for?
    logits = np.array(ev.get_batch_logits(model_params, np.array([ctx + [meta["pad_token"]] * (meta["max_len"] - len(ctx))], dtype=np.int32), meta["pad_token"]))[0]
    lp = logits - np.logaddexp.reduce(logits); valid = np.array(sorted(trie)); top1 = valid[np.argsort(-lp[valid])[:3]]
    print(f"history: {' | '.join(title[i][:35] for i in seqs[user_id][-4:])}\\n")
    print("first digit — the three most likely families and a sample of what lives in each:")
    for t in top1:
        members = level1_members[int(t)]
        print(f"  digit {int(t):3d}  p={np.exp(lp[t]):.2f}  ({len(members):,} items)  e.g. " + " / ".join(title[i][:28] for i in rng.choice(members, 3)))
    # full beam search
    ranked = ev.beam_search_batch(model_params, [ctx], meta, trie)[0]
    recs = [(i, s) for i, s in ranked if i not in seen][:K]
    print(f"\\ntop-{K} after constrained beam search (✔ = the item the user actually reviewed next):")
    for r, (i, s) in enumerate(recs, 1):
        print(f"  {'✔' if i == truth else ' '} {r:>2}. p={np.exp(s):.4f}  {title[i][:75]}")
    if truth not in [i for i, _ in recs]: print(f"\\n  (true next item: {title[truth][:75]})")

interact(generate, user_id=widgets.Dropdown(options=demo_users, description="user"), K=widgets.IntSlider(10, 5, 20, 5));'''),
    ("markdown", "### Find the Nintendo Switch user from the post\n\nAny user whose recent history is all one platform makes the point: the model is never told the platform."),
    ("code", '''KEYWORD = "Switch"  #@param {type:"string"}
def platform_users(word, min_items=4):
    out = []
    for u, s in seqs.items():
        last = s[-min_items:]
        if len(last) == min_items and all(word.lower() in title[i].lower() for i in last): out.append(u)
    return out
pu = platform_users(KEYWORD)
print(f"{len(pu)} users whose last 4 items all mention '{KEYWORD}'")
if pu: generate(pu[0])'''),
    ("markdown", "## 4. How much better than popularity? (small sample)\n\nRecall@10 on a few hundred held-out val targets, versus the popularity list. Tutorial 5 does this properly on all 94,762 users with SASRec in the mix."),
    ("code", '''from models.metrics import recall_at_k, ndcg_at_k
N_EVAL = 100 if QUICK else 500
train_rows = pd.DataFrame({"user_id": np.repeat(list(seqs), [len(s) for s in seqs.values()]), "item_id": np.concatenate(list(seqs.values()))})
pop_rank = train_rows.groupby("item_id").size().sort_values(ascending=False).index.tolist()
eval_users = rng.choice(sorted(seqs), N_EVAL, replace=False)
ctxs = [context_tokens(u) for u in eval_users]; r_tf, r_pop, n_tf = [], [], []
for s in range(0, N_EVAL, 50):
    for u, ranked in zip(eval_users[s:s+50], ev.beam_search_batch(model_params, ctxs[s:s+50], meta, trie)):
        seen = set(seqs[u]); recs = [i for i, _ in ranked if i not in seen][:10]; truth = {val_target[u]}
        r_tf.append(recall_at_k(recs, truth, 10)); n_tf.append(ndcg_at_k(recs, truth, 10))
        r_pop.append(recall_at_k([i for i in pop_rank if i not in seen][:10], truth, 10))
print(f"{model_name}\\n  recall@10  transformer {np.mean(r_tf):.3f}   popularity {np.mean(r_pop):.3f}   (ndcg@10 transformer {np.mean(n_tf):.3f}; {N_EVAL} users)")'''),
    ("markdown", "**Next:** [Tutorial 4 — Ranking](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_04_ranking.ipynb): a second opinion on the shortlist — including the bug that made the first reranker learn the opposite of what it should."),
    ]
    return nb(cells, "Tutorial 3 — generative retrieval")


# ============================================================================ 4
def build_04():
    cells = [("markdown", f"""# Tutorial 4 — Ranking: a second opinion on the shortlist

{badge(4)}

Companion to the *Ranking* section. Here you will:

1. take ALS's top-100 candidates for a set of users and build the seven features the post lists,
2. train the LambdaMART reranker **twice**: once with the positive-injection bug from the first version of the project, once fixed,
3. probe both models to see what each actually learned (the buggy one promotes the lowest-scored, least popular candidate),
4. evaluate both on the *identical* candidate sets for held-out users, and inspect single users — including the "position 98 → position 1" kind of case.

CPU is fine for this one (~3 min)."""),
    ("code", setup_cell(["train.parquet", "item_catalog.parquet", "train_sequences.parquet",
                         "val_targets.parquet", "item_embeddings.npy"])),
    ("markdown", img("re-ranking-modeling.png", "Retrieve wide and cheap, then re-rank a small set of candidates with richer signals.") + "\n\n## 1. Retrieval gives 100 candidates per user\n\nALS is the retrieval stage here (the same reasoning as in the project: it scores the whole catalog in one matrix multiply)."),
    ("code", '''import time, lightgbm as lgb
from models.als import ALSRecommender
from models.ranking_features import FEATURE_COLUMNS, build_item_features, build_user_profile_embeddings, build_user_interaction_counts, assemble_features
from models.metrics import recall_at_k, ndcg_at_k

train = pd.read_parquet(f"{P}/train.parquet"); items = pd.read_parquet(f"{P}/item_catalog.parquet").set_index("item_id"); title = items["description"]
train_seq = pd.read_parquet(f"{P}/train_sequences.parquet"); emb = np.load(f"{P}/item_embeddings.npy")
val_targets = {r.user_id: set(r.item_ids) for r in pd.read_parquet(f"{P}/val_targets.parquet").itertuples()}
n_items = len(items); n_users = int(max(train.user_id.max(), max(val_targets)) + 1)
t0 = time.time(); als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(train, n_users, n_items); print(f"ALS fit {time.time()-t0:.0f}s")

item_features = build_item_features(train, n_items); user_profiles = build_user_profile_embeddings(train_seq, emb); user_n = build_user_interaction_counts(train_seq)
eligible = sorted(set(train_seq.user_id) & set(val_targets)); rng = np.random.RandomState(42)
N_TRAIN, N_EVAL = (1500, 400) if QUICK else (12000, 3000)
chosen = rng.choice(eligible, N_TRAIN + N_EVAL, replace=False); train_users, eval_users = np.sort(chosen[:N_TRAIN]), np.sort(chosen[N_TRAIN:])

def candidates(users, N=100):
    ids, scores = als.model.recommend(users, als.user_items[users], N=N, filter_already_liked_items=True)
    return [list(r) for r in ids], [list(r) for r in scores]
print(f"{N_TRAIN:,} ranker-training users, {N_EVAL:,} held-out users; 100 ALS candidates each")'''),
    ("markdown", """## 2. Build the feature table — with and without the bug

Seven features per (user, candidate): ALS score, log popularity, price (+ has_price flag), recency, content similarity to the user's profile embedding, log history length. The label is 1 if the candidate is one of the user's val items.

The model is a gradient-boosted tree ensemble (LightGBM) that maps those seven numbers to a score, $s_{ui} = f(\\phi_{ui})$, $\\phi_{ui} \\in \\mathbb{R}^7$. What makes it a *ranker* rather than a classifier is the objective: LambdaMART looks at pairs. For every pair where item $i$ was the true next purchase and item $j$ was not, it pushes $s_{ui}$ above $s_{uj}$, and pushes harder when swapping the two would change NDCG more:

$$L = \\sum_{u} \\sum_{i \\succ j} |\\Delta\\mathrm{NDCG}_{ij}| \\cdot \\log\\big(1 + e^{-(s_{ui} - s_{uj})}\\big), \\qquad \\mathrm{rec}(u) = \\operatorname{sort}_{\\,i \\in \\mathrm{candidates}(u)} s_{ui}$$

**The bug:** ALS's top-100 contains the true item for only ~13% of users. The first version "fixed" that by *injecting* the missing true items with a placeholder ALS score (the list minimum). Toggle it below."""),
    ("code", '''def build_table(users, inject):
    ids, scores = candidates(users); n_inj = 0
    for k, u in enumerate(users):
        missing = val_targets[u] - set(ids[k])
        if inject and missing:
            n_inj += len(missing); ids[k] += list(missing); scores[k] += [min(scores[k])] * len(missing)
    df = assemble_features(users, ids, scores, item_features, user_profiles, user_n, emb)
    pos = {(u, i) for u in users for i in val_targets[u]}
    df["label"] = [int((u, i) in pos) for u, i in zip(df.user_id, df.item_id)]
    if not inject:   # groups with no positive carry no LambdaRank gradient — drop them
        keep = df.groupby("user_id")["label"].transform("max") == 1; df = df[keep]
    return df.sort_values("user_id").reset_index(drop=True), n_inj

def fit_ranker(df):
    r = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20, random_state=42, verbosity=-1)
    r.fit(df[FEATURE_COLUMNS].values, df["label"].values, group=df.groupby("user_id").size().values); return r

df_bug, n_inj = build_table(train_users, inject=True)
print(f"WITH injection: {len(df_bug):,} rows, {df_bug.label.sum():,} positives of which {n_inj:,} injected ({n_inj/df_bug.label.sum():.0%})")
df_fix, _ = build_table(train_users, inject=False)
print(f"FIXED:          {len(df_fix):,} rows, {df_fix.label.sum():,} positives, {df_fix.user_id.nunique():,} users kept (the ones whose true item ALS actually retrieved)")
t0 = time.time(); ranker_bug = fit_ranker(df_bug); ranker_fix = fit_ranker(df_fix); print(f"both rankers trained in {time.time()-t0:.0f}s")'''),
    ("markdown", "## 3. What did each one learn?\n\nHold every feature at a typical value and sweep one. The buggy ranker's response to the ALS score and to popularity is *inverted*: it learned that the injected item — lowest ALS score, zero training popularity — is the positive."),
    ("code", '''base = dict(als_score=0.5, item_popularity_log=np.log1p(50), item_price_imputed=30.0, has_price=1.0, item_recency_norm=0.8, content_sim=0.7, user_n_interactions_log=np.log1p(4))
def sweep(model, feat, values):
    rows = []
    for v in values:
        d = dict(base); d[feat] = v; rows.append(float(model.booster_.predict(np.array([[d[c] for c in FEATURE_COLUMNS]]))[0]))
    return rows

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, (feat, xs, lab) in zip(axes, [("als_score", np.linspace(0, 1.3, 14), "ALS score"), ("item_popularity_log", np.log1p([0, 1, 5, 20, 100, 500, 2000]), "log(1 + train count)"), ("content_sim", np.linspace(0, 1, 11), "content similarity")]):
    ax.plot(xs, sweep(ranker_bug, feat, xs), color="#eb6834", lw=2, label="with injection (bug)"); ax.plot(xs, sweep(ranker_fix, feat, xs), color="#2a78d6", lw=2, label="fixed")
    ax.set_xlabel(lab); ax.set_ylabel("ranker score"); [ax.spines[s].set_visible(False) for s in ["top", "right"]]
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()'''),
    ("markdown", "## 4. Same candidates, three orders\n\nFor the held-out users: ALS's own order, the buggy reranker, the fixed reranker — all on the identical top-100."),
    ("code", '''ids_e, scores_e = candidates(eval_users)
df_e = assemble_features(eval_users, ids_e, scores_e, item_features, user_profiles, user_n, emb)
df_e["bug"] = ranker_bug.booster_.predict(df_e[FEATURE_COLUMNS].values); df_e["fix"] = ranker_fix.booster_.predict(df_e[FEATURE_COLUMNS].values)
orders = {"ALS order": {u: ids_e[k] for k, u in enumerate(eval_users)}}
for col in ["bug", "fix"]:
    orders[{"bug": "reranked (bug)", "fix": "reranked (fixed)"}[col]] = {u: g.sort_values(col, ascending=False).item_id.tolist() for u, g in df_e.groupby("user_id")}
res = {name: {f"recall@{k}": np.mean([recall_at_k(lst[u], val_targets[u], k) for u in eval_users]) for k in (10, 20)} | {f"ndcg@{k}": np.mean([ndcg_at_k(lst[u], val_targets[u], k) for u in eval_users]) for k in (10, 20)} for name, lst in orders.items()}
display(pd.DataFrame(res).T.round(4))'''),
    ("markdown", "### Inspect one user\n\nUsers where the fixed reranker moved a true item into the top 10 from deep in ALS's list. The feature values show *why* — usually content similarity."),
    ("code", '''moved = []
for u in eval_users:
    a, f = orders["ALS order"][u], orders["reranked (fixed)"][u]
    for t in val_targets[u]:
        if t in a and a.index(t) >= 20 and f.index(t) < 10: moved.append((u, t, a.index(t) + 1, f.index(t) + 1))
print(f"{len(moved)} (user, item) cases moved from ≥ position 20 into the top 10")

def inspect(case):
    u, t, pa, pf = case
    print("history:", " | ".join(title[i][:30] for i in train_seq.set_index('user_id').loc[u, 'item_ids'][-5:]))
    print(f"\\ntrue next item: {title[t][:80]}\\n  ALS position {pa} → reranked position {pf}")
    row = df_e[(df_e.user_id == u) & (df_e.item_id == t)].iloc[0]
    print("  features:", {c: round(float(row[c]), 3) for c in FEATURE_COLUMNS})
    print("\\nreranked top-5:"); [print(f"  {'✔' if i in val_targets[u] else ' '} {r}. {title[i][:70]}") for r, i in enumerate(orders['reranked (fixed)'][u][:5], 1)]

if moved: interact(inspect, case=widgets.Dropdown(options=[(f"user {u}: {pa} → {pf}", c) for c in moved[:40] for (u, t, pa, pf) in [c]], description="case"))'''),
    ("markdown", "**Next:** [Tutorial 5 — Evaluation](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_05_evaluation.ipynb): every model on the papers' protocol, with SASRec."),
    ]
    return nb(cells, "Tutorial 4 — ranking")


# ============================================================================ 5
def build_05():
    cells = [("markdown", f"""# Tutorial 5 — Measuring it the way the papers do

{badge(5)}

Companion to the two results sections. Here you will:

1. see the two protocols side by side for one user — the same history, the "predict next year" target versus the "predict the next item" target,
2. load the trained SASRec and Semantic-ID checkpoints and reproduce the leave-one-out table on a sample of users (all 94,762 takes ~30 min on a T4; the default is 3,000),
3. compare the four models' lists for any user, with the true next item marked.

Use a GPU runtime; beam search is the slow part."""),
    ("code", setup_cell(["item_catalog.parquet", "user_catalog.parquet", "item_tokens.parquet", "loo_train.parquet",
                         "loo_train_sequences.parquet", "loo_val_sequences.parquet", "loo_val_targets.parquet",
                         "loo_test_targets.parquet", "transformer_vocab_meta_loo.json", "transformer_checkpoint_loo.pkl",
                         "sasrec_checkpoint.pkl", "train.parquet", "val_targets.parquet", "loo_evaluation_results.json"])),
    ("markdown", img("eval_flow_example.png", "Leave-one-out: hide the last item, rank, score with Recall@K and NDCG@K.")),
    ("markdown", """## 1. One user, two questions

The time split asks: given everything before Nov 2019, what will this user review in 2021–2023? Leave-one-out asks: given everything but the last item, what is the last item? Same person, very different problems."""),
    ("code", '''import json, pickle, jax, jax.numpy as jnp, time
items = pd.read_parquet(f"{P}/item_catalog.parquet").set_index("item_id"); title = items["description"]
loo_train_seq = {r.user_id: list(r.item_ids) for r in pd.read_parquet(f"{P}/loo_train_sequences.parquet").itertuples()}
loo_val_seq   = {r.user_id: list(r.item_ids) for r in pd.read_parquet(f"{P}/loo_val_sequences.parquet").itertuples()}
loo_val_t  = {r.user_id: set(r.item_ids) for r in pd.read_parquet(f"{P}/loo_val_targets.parquet").itertuples()}
loo_test_t = {r.user_id: set(r.item_ids) for r in pd.read_parquet(f"{P}/loo_test_targets.parquet").itertuples()}
train = pd.read_parquet(f"{P}/train.parquet"); ts_val_t = {r.user_id: set(r.item_ids) for r in pd.read_parquet(f"{P}/val_targets.parquet").itertuples()}
train["timestamp"] = pd.to_datetime(train["timestamp"]); ts_hist = train.sort_values("timestamp").groupby("user_id")["item_id"].apply(list).to_dict()
train_items = set(train.item_id)
rng = np.random.RandomState(3)
both = [u for u in ts_val_t if u in ts_hist and len(ts_hist[u]) >= 3 and len(loo_train_seq[u]) >= 3]
demo_users = sorted(rng.choice(both, 40, replace=False).tolist())

def two_questions(user_id):
    print("TIME SPLIT — context (reviews before Nov 2019):"); [print("   ", title[i][:70]) for i in ts_hist[user_id][-5:]]
    print("  targets (reviews Nov 2019 – Oct 2021):"); [print(f"   {'(never seen in training) ' if i not in train_items else ''}{title[i][:60]}") for i in ts_val_t[user_id]]
    print("\\nLEAVE-ONE-OUT — context (all but the last two):"); [print("   ", title[i][:70]) for i in loo_train_seq[user_id][-5:]]
    print("  val target (second-to-last item):"); [print("   ", title[i][:70]) for i in loo_val_t[user_id]]

interact(two_questions, user_id=widgets.Dropdown(options=demo_users, description="user"));'''),
    ("markdown", "## 2. Load the four models\n\nPopularity and ALS are fit on the leave-one-out training rows in a few seconds; SASRec and the Semantic-ID Transformer come from the checkpoints trained for the post."),
    ("code", '''from models.popularity import PopularityRecommender
from models.als import ALSRecommender
from models import sasrec, transformer as tx
import evaluate_retrieval as ev
from models.metrics import recall_at_k, ndcg_at_k

loo_train = pd.read_parquet(f"{P}/loo_train.parquet"); n_items = len(items); n_users = len(pd.read_parquet(f"{P}/user_catalog.parquet"))
pop = PopularityRecommender().fit(loo_train)
t0 = time.time(); als = ALSRecommender(factors=64, regularization=0.05, iterations=15, alpha=40.0).fit(loo_train, n_users, n_items); print(f"ALS fit {time.time()-t0:.0f}s")

sas_raw = pickle.load(open(f"{P}/sasrec_checkpoint.pkl", "rb")); sas_p = jax.tree_util.tree_map(jnp.array, sas_raw["params"]); sas_cfg = sas_raw["config"]
sas_score = jax.jit(lambda t: sasrec.score_batch(sas_p, t, n_items, sas_cfg["n_heads"]))
print(f"SASRec: epoch {sas_raw['epoch']}, {sas_cfg['n_layers']} layers d={sas_cfg['d_model']}, loss {sas_raw['history'][-1]:.2f}")

tf_raw = pickle.load(open(f"{P}/transformer_checkpoint_loo.pkl", "rb")); tf_p = jax.tree_util.tree_map(jnp.array, tf_raw["params"]); tf_cfg = tf_raw["config"]
meta = json.load(open(f"{P}/transformer_vocab_meta_loo.json")); item_tokens = pd.read_parquet(f"{P}/item_tokens.parquet"); trie = ev.build_trie(item_tokens)
item_to_tok = {r.item_id: [r.tok1, r.tok2, r.tok3, r.tok4] for r in item_tokens.itertuples(index=False)}
ev.N_HEADS = tf_cfg["n_heads"]; ev.BEAM_WIDTH = 30
print(f"Semantic-ID Transformer: epoch {tf_raw['epoch']}, {tf_cfg['n_layers']} layers d={tf_cfg['d_model']}, loss {tf_raw['history'][-1]:.2f}")

max_ctx = (meta["max_len"] - 1) // 4 - 1
def tf_context(hist, u): return [meta["item_vocab"] + int(u) % meta["user_buckets"]] + sum((item_to_tok[i] for i in hist[-max_ctx:]), [])

def recommend_all(users, contexts, K=20):
    """contexts: dict user -> item history. Returns {model: {user: ranked list}} with the history filtered out."""
    out = {"popularity": {}, "als": {}, "sasrec": {}, "transformer": {}}
    for u in users:
        seen = set(contexts[u])
        out["popularity"][u] = [i for i in pop.recommend(u, K + 50) if i not in seen][:K]
        out["als"][u] = [i for i in (als.recommend(u, K + 50) if not als.is_cold(u) else pop.recommend(u, K + 50)) if i not in seen][:K]
    for s in range(0, len(users), 256):
        bu = users[s:s+256]; toks = np.full((len(bu), sas_cfg["max_items"]), n_items, dtype=np.int32)
        for k, u in enumerate(bu):
            h = contexts[u][-sas_cfg["max_items"]:]; toks[k, :len(h)] = h
        sc = np.array(sas_score(jnp.array(toks)))
        for k, u in enumerate(bu):
            sc[k, list(contexts[u])] = -np.inf; top = np.argpartition(-sc[k], K)[:K]; out["sasrec"][u] = top[np.argsort(-sc[k][top])].tolist()
    for s in range(0, len(users), 50):
        bu = users[s:s+50]
        for u, ranked in zip(bu, ev.beam_search_batch(tf_p, [tf_context(contexts[u], u) for u in bu], meta, trie)):
            out["transformer"][u] = [i for i, _ in ranked if i not in set(contexts[u])][:K]
    return out'''),
    ("markdown", "## 3. Four lists for one user\n\nLeave-one-out val: context = all but the last two items, target = the second-to-last."),
    ("code", '''def four_lists(user_id, K=10):
    recs = recommend_all([user_id], {user_id: loo_train_seq[user_id]}, K); truth = loo_val_t[user_id]
    print("history:", " | ".join(title[i][:28] for i in loo_train_seq[user_id][-5:])); print("true next item:", title[list(truth)[0]][:80], "\\n")
    for name in recs:
        lst = recs[name][user_id]; hit = [r for r, i in enumerate(lst, 1) if i in truth]
        print(f"{name:<12} {'✔ hit at #' + str(hit[0]) if hit else 'miss'}"); [print(f"   {'✔' if i in truth else ' '} {r}. {title[i][:65]}") for r, i in enumerate(lst[:5], 1)]

interact(four_lists, user_id=widgets.Dropdown(options=demo_users, description="user"), K=widgets.IntSlider(10, 5, 20, 5));'''),
    ("markdown", "## 4. The table, on a sample of users\n\nRecall@K and NDCG@K for K = 5, 10, 20 — the papers' numbers. `N_USERS = 0` evaluates all 94,762 (about 30 min on a T4, mostly beam search). The full-population result from the post is loaded underneath for comparison."),
    ("code", '''N_USERS = 3000   #@param {type:"integer"}
SPLIT = "val"    #@param ["val", "test"]
if QUICK: N_USERS = 150
targets_full = loo_val_t if SPLIT == "val" else loo_test_t; contexts = loo_train_seq if SPLIT == "val" else loo_val_seq
users = sorted(targets_full) if N_USERS == 0 else sorted(np.random.RandomState(7).choice(sorted(targets_full), N_USERS, replace=False).tolist())
t0 = time.time(); recs = recommend_all(users, contexts); print(f"{len(users):,} users scored in {time.time()-t0:.0f}s")
rows = {m: {**{f"recall@{k}": np.mean([recall_at_k(recs[m][u], targets_full[u], k) for u in users]) for k in (5, 10, 20)},
            **{f"ndcg@{k}": np.mean([ndcg_at_k(recs[m][u], targets_full[u], k) for u in users]) for k in (5, 10, 20)}} for m in recs}
print(f"\\nthis run — {SPLIT}, {len(users):,} users:"); display(pd.DataFrame(rows).T.round(4))
full = json.load(open(f"{P}/loo_evaluation_results.json"))["results"][SPLIT]
print(f"the post — {SPLIT}, all 94,762 users:"); display(pd.DataFrame(full).T.round(4))'''),
    ("code", '''fig, ax = plt.subplots(figsize=(7, 3.2)); names = list(rows); vals = [rows[m]["recall@10"] for m in names]
ax.bar(names, vals, color="#2a78d6", width=0.55); [ax.text(i, v + max(vals) * 0.02, f"{v:.4f}", ha="center", fontsize=9) for i, v in enumerate(vals)]
ax.set_ylabel("recall@10"); ax.set_title(f"leave-one-out {SPLIT}, {len(users):,} users", loc="left"); [ax.spines[s].set_visible(False) for s in ["top", "right"]]
plt.show()
print("Under the papers' protocol the generative model is ~3x popularity and in TIGER's published range — but SASRec, the ID-based Transformer with the same backbone, is still ahead on this dataset. The post's closing section lists the suspects.")'''),
    ("markdown", img("recall_two_protocols.png", "The post's one-figure summary: the same models under both protocols.")),
    ("markdown", "That is the end of the tutorial series. The full pipeline scripts, docs for each step and the audit that led to the second protocol are in the [repository](https://github.com/juanmigutierrez/generative-recommendation-engine)."),
    ]
    return nb(cells, "Tutorial 5 — evaluation")


if __name__ == "__main__":
    for n, builder in [(1, build_01), (2, build_02), (3, build_03), (4, build_04), (5, build_05)]:
        path = os.path.join(OUT_DIR, NOTEBOOK_FILES[n])
        with open(path, "w") as f:
            json.dump(builder(), f, indent=1)
        print("wrote", path)
