"""
Step 6b: train the decoder-only Transformer on user Semantic ID sequences.

Resumable/chunked by design: one epoch over all 85,757 users takes ~110s on this sandbox's
CPU, close to the ~170s per-command budget once data loading and JIT compilation overhead is
included, and this project's build process runs as a series of separate shell calls rather
than one long-lived process. So this script does ONE epoch per invocation, checkpointing
before and after, and needs to be re-run repeatedly to reach a full training budget --
exactly the same reasoning and pattern used for the RQ-VAE retrain in Step 5.

Run (repeatedly, once per epoch):
    python backend/scripts/train_transformer.py --epochs 1
Check progress:
    python backend/scripts/train_transformer.py --status

Leave-one-out / paper-comparable variant (docs/09_audit_sources_and_results.md), trained on
transformer_train_sequences_loo.npz from `build_semantic_sequences.py --loo`, with a
separate checkpoint (transformer_checkpoint_loo.pkl) and TIGER-like size + dropout:
    python backend/scripts/train_transformer.py --loo --epochs 5 --d-model 128 --layers 4 --d-ff 512 --dropout 0.1
The architecture flags are stored in the checkpoint on the first run and reused afterwards.
"""
import argparse
import json
import os
import pickle
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import transformer as tx

PROCESSED_DIR = os.path.join("data", "processed")
CKPT_PATH = os.path.join(PROCESSED_DIR, "transformer_checkpoint.pkl")  # overridden by --loo

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
BATCH_SIZE = 512
LR = 3e-4
SEED = 42


DEFAULT_CONFIG = {"d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF,
                  "dropout": 0.0, "lr": LR, "batch_size": BATCH_SIZE}


def paths(loo: bool):
    suffix = "_loo" if loo else ""
    return (os.path.join(PROCESSED_DIR, f"transformer_checkpoint{suffix}.pkl"),
            os.path.join(PROCESSED_DIR, f"transformer_train_sequences{suffix}.npz"),
            os.path.join(PROCESSED_DIR, f"transformer_vocab_meta{suffix}.json"),
            os.path.join(PROCESSED_DIR, f"transformer_training_curve{suffix}.png"))


def load_data(loo=False):
    _, seq_path, meta_path, _ = paths(loo)
    data = np.load(seq_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return data["tokens"], meta


def fresh_state(tokens, meta, config):
    key = jax.random.PRNGKey(SEED)
    params = tx.init_params(key, meta["vocab_size"], meta["max_len"], config["d_model"],
                            config["n_heads"], config["n_layers"], config["d_ff"])
    opt_init, _ = tx.make_adam(lr=config["lr"])
    opt_state = opt_init(params)
    return {
        "params": params,
        "opt_state": opt_state,
        "epoch": 0,
        "history": [],
        "rng_state": np.random.RandomState(SEED).get_state(),
        "config": dict(config),
    }


def save_checkpoint(state, ckpt_path):
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "params": jax.tree_util.tree_map(np.array, state["params"]),
            "opt_state": jax.tree_util.tree_map(np.array, state["opt_state"]),
            "epoch": state["epoch"],
            "history": state["history"],
            "rng_state": state["rng_state"],
            "config": state.get("config", DEFAULT_CONFIG),
        }, f)


def load_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        raw = pickle.load(f)
    return {
        "params": jax.tree_util.tree_map(jnp.array, raw["params"]),
        "opt_state": jax.tree_util.tree_map(jnp.array, raw["opt_state"]),
        "epoch": raw["epoch"],
        "history": raw["history"],
        "rng_state": raw["rng_state"],
        "config": raw.get("config", DEFAULT_CONFIG),  # old checkpoints predate the config field
    }


def run_epochs(n_epochs, loo=False, config=None, limit_users=None):
    import time
    ckpt_path, _, _, curve_path = paths(loo)
    tokens, meta = load_data(loo)
    if limit_users:
        tokens = tokens[:limit_users]  # smoke-test mode only
    n_users = tokens.shape[0]
    x = jnp.array(tokens)

    if os.path.exists(ckpt_path):
        state = load_checkpoint(ckpt_path)
        print(f"resuming from epoch {state['epoch']} (config: {state['config']})")
    else:
        state = fresh_state(tokens, meta, config or DEFAULT_CONFIG)
        print(f"fresh start (config: {state['config']})")
    cfg = state["config"]
    n_heads, dropout, batch_size = cfg["n_heads"], cfg["dropout"], cfg["batch_size"]

    _, opt_update = tx.make_adam(lr=cfg["lr"])
    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, b, k: tx.loss_fn(p, b, meta["pad_token"], n_heads, dropout, k)))

    rng = np.random.RandomState(SEED)
    rng.set_state(state["rng_state"])
    n_batches = n_users // batch_size
    drop_key = jax.random.PRNGKey(SEED + state["epoch"])

    params, opt_state = state["params"], state["opt_state"]
    for _ in range(n_epochs):
        t0 = time.time()
        perm = rng.permutation(n_users)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            drop_key, sub = jax.random.split(drop_key)
            loss, grads = loss_and_grad(params, x[idx], sub)
            params, opt_state = opt_update(grads, opt_state, params)
            epoch_loss += float(loss)
        epoch_loss /= n_batches
        state["epoch"] += 1
        state["history"].append(epoch_loss)
        print(f"epoch {state['epoch']:3d}  loss {epoch_loss:.4f}  ({time.time() - t0:.0f}s)")

    state["params"], state["opt_state"] = params, opt_state
    state["rng_state"] = rng.get_state()
    save_checkpoint(state, ckpt_path)
    print(f"checkpoint saved at epoch {state['epoch']} -> {ckpt_path}")

    # keep the training-curve plot fresh after every chunk, cheap to redo
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(state["history"], color="#4F46E5")
    ax.set_title("Transformer training loss (next-token, Semantic ID sequences)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=110)
    plt.close(fig)


def print_status(loo=False):
    ckpt_path = paths(loo)[0]
    if not os.path.exists(ckpt_path):
        print("no checkpoint yet")
        return
    state = load_checkpoint(ckpt_path)
    print(f"epoch: {state['epoch']}  config: {state['config']}")
    print(f"loss history: {[round(h, 4) for h in state['history']]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--loo", action="store_true", help="train the leave-one-out model (separate data + checkpoint)")
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--heads", type=int, default=N_HEADS)
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--d-ff", type=int, default=D_FF)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit-users", type=int, default=None, help="smoke test on the first N users only")
    args = ap.parse_args()

    if args.status:
        print_status(args.loo)
    else:
        config = {"d_model": args.d_model, "n_heads": args.heads, "n_layers": args.layers, "d_ff": args.d_ff,
                  "dropout": args.dropout, "lr": args.lr, "batch_size": args.batch_size}
        run_epochs(args.epochs, loo=args.loo, config=config, limit_users=args.limit_users)
