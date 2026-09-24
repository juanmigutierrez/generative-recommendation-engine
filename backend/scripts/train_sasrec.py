"""
Step 10b: train the SASRec reference baseline (models/sasrec.py) on the leave-one-out
split, so the Semantic ID transformer has the same control TIGER's paper reports against.

Same resumable one-checkpoint-per-run pattern as train_transformer.py. Default size matches
SASRec's paper defaults on Amazon (d=50 in the original; 64 here, 2 layers, max 20 items, TIGER's context length;
TIGER's baseline table uses these) -- pass flags to change.

Run (repeat until the loss flattens; prints seconds/epoch so you can budget):
    python backend/scripts/train_sasrec.py --epochs 5
    python backend/scripts/train_sasrec.py --status
"""
import argparse
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import transformer as tx
from models import sasrec

PROCESSED_DIR = os.path.join("data", "processed")
CKPT_PATH = os.path.join(PROCESSED_DIR, "sasrec_checkpoint.pkl")
SEED = 42
DEFAULTS = {"d_model": 64, "n_heads": 2, "n_layers": 2, "d_ff": 256, "dropout": 0.2,
            "lr": 1e-3, "batch_size": 128, "max_items": 20}


def run_epochs(n_epochs, config, limit_users=None):
    seqs = pd.read_parquet(os.path.join(PROCESSED_DIR, "loo_train_sequences.parquet"))
    n_items = len(pd.read_parquet(os.path.join(PROCESSED_DIR, "item_catalog.parquet")))
    pad = n_items

    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "rb") as f:
            raw = pickle.load(f)
        state = {k: raw[k] for k in ("epoch", "history", "rng_state", "config")}
        state["params"] = jax.tree_util.tree_map(jnp.array, raw["params"])
        state["opt_state"] = jax.tree_util.tree_map(jnp.array, raw["opt_state"])
        print(f"resuming from epoch {state['epoch']} (config: {state['config']})")
    else:
        cfg = dict(config)
        params = tx.init_params(jax.random.PRNGKey(SEED), n_items + 1, cfg["max_items"],
                                cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["d_ff"])
        opt_init, _ = tx.make_adam(lr=cfg["lr"])
        state = {"params": params, "opt_state": opt_init(params), "epoch": 0, "history": [],
                 "rng_state": np.random.RandomState(SEED).get_state(), "config": cfg}
        print(f"fresh start (config: {cfg})")
    cfg = state["config"]

    tokens, _, _ = sasrec.build_sequences(seqs, n_items, cfg["max_items"])
    if limit_users:
        tokens = tokens[:limit_users]
    x = jnp.array(tokens)
    n_users, bs = tokens.shape[0], cfg["batch_size"]

    _, opt_update = tx.make_adam(lr=cfg["lr"])
    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, b, k: sasrec.loss_fn(p, b, pad, cfg["n_heads"], cfg["dropout"], k)))
    rng = np.random.RandomState(SEED); rng.set_state(state["rng_state"])
    drop_key = jax.random.PRNGKey(SEED + state["epoch"])
    params, opt_state = state["params"], state["opt_state"]
    n_batches = n_users // bs
    for _ in range(n_epochs):
        t0 = time.time(); perm = rng.permutation(n_users); epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            drop_key, sub = jax.random.split(drop_key)
            loss, grads = loss_and_grad(params, x[idx], sub)
            params, opt_state = opt_update(grads, opt_state, params)
            epoch_loss += float(loss)
        state["epoch"] += 1; state["history"].append(epoch_loss / n_batches)
        print(f"epoch {state['epoch']:3d}  loss {epoch_loss / n_batches:.4f}  ({time.time() - t0:.0f}s)")

    state["params"], state["opt_state"], state["rng_state"] = params, opt_state, rng.get_state()
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.array, params),
                     "opt_state": jax.tree_util.tree_map(np.array, opt_state),
                     "epoch": state["epoch"], "history": state["history"],
                     "rng_state": state["rng_state"], "config": cfg}, f)
    print(f"checkpoint saved at epoch {state['epoch']} -> {CKPT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--limit-users", type=int, default=None)
    args = ap.parse_args()
    if args.status:
        if not os.path.exists(CKPT_PATH):
            print("no checkpoint yet")
        else:
            with open(CKPT_PATH, "rb") as f:
                raw = pickle.load(f)
            print(f"epoch: {raw['epoch']}  config: {raw['config']}")
            print(f"loss history: {[round(h, 4) for h in raw['history']]}")
    else:
        run_epochs(args.epochs, {k: getattr(args, k) for k in DEFAULTS}, args.limit_users)
