"""
Diagnostics for the Step 6 transformer -- see docs/09_audit_sources_and_results.md.

    python backend/scripts/diagnose_retrieval.py split 400    # project's time-split protocol, warm val users
    python backend/scripts/diagnose_retrieval.py loo 1000     # TIGER-style leave-one-out on train sequences
    python backend/scripts/diagnose_retrieval.py loo_shuf 1000  # same, but every user gets another user's history
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import evaluate_retrieval as ev
from models.metrics import recall_at_k, ndcg_at_k
from models.popularity import PopularityRecommender

P = os.path.join("data", "processed")
mode = sys.argv[1] if len(sys.argv) > 1 else "split"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400

params, meta, ep, trie, get_context, val_t, test_t = ev.setup()
train = pd.read_parquet(os.path.join(P, "train.parquet")); pop = PopularityRecommender().fit(train)
train_items = set(train.item_id); seen = train.groupby("user_id").item_id.apply(set).to_dict()
seqs = pd.read_parquet(os.path.join(P, "train_sequences.parquet"))
item_tok = pd.read_parquet(os.path.join(P, "item_tokens.parquet")).set_index("item_id")
tok = lambda it: [int(x) for x in item_tok.loc[it, ["tok1", "tok2", "tok3", "tok4"]].values]
rng = np.random.RandomState(0)
if mode == "split":
    users = [u for u in val_t if get_context(u) is not None]; users = list(rng.choice(users, N, replace=False))
    ctx = [get_context(u) for u in users]; targets = {u: val_t[u] for u in users}
else:
    s = seqs[seqs.item_ids.apply(len) >= 4].sample(N, random_state=0)
    users = list(s.user_id); ctx = []; targets = {}
    for u, items in zip(s.user_id, s.item_ids):
        items = list(items); ctx.append(sum((tok(i) for i in items[-10:-1]), [])); targets[u] = {items[-1]}
    if mode == "loo_shuf":
        ctx = [ctx[(i + 1) % N] for i in range(N)]
t0 = time.time(); recs = {}
for st in range(0, N, 50):
    for u, r in zip(users[st:st + 50], ev.beam_search_batch(params, ctx[st:st + 50], meta, trie)):
        recs[u] = [i for i, _ in r]
print(f"beam search {N} users: {time.time() - t0:.0f}s\n== mode={mode}")

def score(fn, name):
    r10 = np.nanmean([recall_at_k(fn(u), targets[u], 10) for u in users])
    r20 = np.nanmean([recall_at_k(fn(u), targets[u], 20) for u in users])
    n10 = np.nanmean([ndcg_at_k(fn(u), targets[u], 10) for u in users])
    print(f"  {name:<24} recall@10 {r10:.4f}  recall@20 {r20:.4f}  ndcg@10 {n10:.4f}")

score(lambda u: recs[u], "transformer (beam 20)")
if mode == "split":
    score(lambda u: pop.recommend(u, 20), "popularity")
else:
    score(lambda u: [i for i in pop.ranked_items if i not in (seen.get(u, set()) - targets[u])][:20], "popularity")
allr = [i for u in users for i in recs[u]]
print("  recs that are cold items: %.1f%%" % (100 * np.mean([i not in train_items for i in allr])))
print("  recs already in user's train history: %.1f%%" % (100 * np.mean([i in seen.get(u, set()) for u in users for i in recs[u]])))
print("  distinct items across all recs: %d of %d slots" % (len(set(allr)), len(allr)))
print("  targets that are cold items: %.1f%%" % (100 * np.mean([i not in train_items for u in users for i in targets[u]])))
