"""
build_dataset.py -- generate the multi-SNR PQ dataset and extract features.

Data structure
--------------
A *base waveform* is one random realisation of one class, generated clean.
Each base waveform is then corrupted at all 5 SNR levels, producing 5 rows that
share the same `group` id. Splits are made over `group`, never over rows, so
all 5 SNR copies of a waveform always land in the same partition. This is what
makes the evaluation a test of generalisation to *new disturbances* rather than
to new noise draws.

    total rows = 29 classes x N_BASE base waveforms x 5 SNR levels

Execution model
---------------
The build is split into one resumable step per SNR level. Each step regenerates
the clean base waveforms from the same seed (cheap, ~1 s), adds noise at its
own level, extracts features in small batches and writes a shard. Peak memory
is a few tens of MB and any step can be re-run independently.

    python scripts/build_dataset.py --step 0 ... --step 4     # one shard per SNR level
    python scripts/build_dataset.py --merge                   # combine into dataset.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.features import PQFeatureExtractor
from src.pqmodel import pqmodel, add_awgn

# ---- generator parameters (confirmed) -------------------------------------
FS = 6400.0        # sampling frequency [Hz]   -> 128 samples/cycle
F0 = 50.0          # fundamental frequency [Hz]
NCYC = 10          # cycles per sample         -> 1280 points, 0.2 s
AMP = 1.0          # nominal amplitude [pu]
F_MAX = 1600.0     # S-transform upper band edge [Hz]
SNRS = [40, 30, 20, 10, 0]
CLEAN = 999          # sentinel meaning "no noise added" (infinite SNR)
LEVELS = SNRS + [CLEAN]      # step 0..4 = noisy levels, step 5 = clean
SEED = 20260807
BATCH = 250


def level_name(s):
    return "clean" if int(s) == CLEAN else f"{int(s)} dB"        # signals held in memory at once


def clean_set(n_base, seed):
    """Regenerate the clean base waveforms deterministically."""
    n_pts = int(round(FS / F0 * NCYC))
    raw = pqmodel(ns=n_base, fs=FS, f=F0, n=NCYC, A=AMP, seed=seed)
    clean = raw.transpose(2, 0, 1).reshape(29 * n_base, n_pts).astype(np.float32)
    del raw
    y = np.repeat(np.arange(1, 30, dtype=np.int16), n_base)
    g = np.arange(29 * n_base, dtype=np.int32)
    return clean, y, g


def step(k, n_base, shard_dir, seed=SEED):
    """
    Build the feature shard for level index k.

    k = 0..4 -> AWGN at 40/30/20/10/0 dB
    k = 5    -> clean, no noise added at all (the noise-free upper bound)
    """
    s = LEVELS[k]
    is_clean = (s == CLEAN)
    clean, y, g = clean_set(n_base, seed)
    rng = np.random.default_rng(seed + 1000 + k)     # independent per level

    fx = PQFeatureExtractor(FS, F0, NCYC, F_MAX)
    names = fx.feature_names()
    n, d = clean.shape[0], len(names)
    X = np.empty((n, d), dtype=np.float32)

    t0 = time.perf_counter()
    ach = []
    for a in range(0, n, BATCH):
        b = min(a + BATCH, n)
        c = clean[a:b].astype(np.float64)
        if is_clean:
            sig = c                       # the waveform exactly as generated
        else:
            sig = add_awgn(c, s, rng)
            ach.append(10 * np.log10(np.mean(c ** 2)
                                     / np.mean((sig - c) ** 2)))
        X[a:b] = fx.transform(sig)
        if (a // BATCH) % 8 == 0:
            el = time.perf_counter() - t0
            print(f"  {b:>5}/{n}  [{el:.0f}s elapsed, "
                  f"{el/max(b,1)*(n-b):.0f}s left]", flush=True)
        del sig, c

    os.makedirs(shard_dir, exist_ok=True)
    p = os.path.join(shard_dir, f"shard_{k}.npz")
    np.savez_compressed(p, X=X, y=y, group=g,
                        snr=np.full(n, s, dtype=np.int16),
                        names=np.array(names))
    tag = ("no noise added" if is_clean
           else f"achieved {np.mean(ach):.2f} dB")
    print(f"{level_name(s)} ({tag}): "
          f"{X.shape} in {(time.perf_counter()-t0)/60:.1f} min -> {p}")


def merge(shard_dir, out, steps=None):
    """
    Combine shards into one dataset.

    steps=None      -> every shard present in the folder
    steps=[0,1,2,3,4]  -> the 5 noisy levels only (the original dataset)
    steps=[5]       -> clean only (noise-free upper bound)
    steps=[0..5]    -> all six levels
    """
    if steps is None:
        steps = [k for k in range(len(LEVELS))
                 if os.path.exists(os.path.join(shard_dir, f"shard_{k}.npz"))]
    if not steps:
        raise SystemExit(f"no shards found in {shard_dir}")

    Xs, ys, gs, ss = [], [], [], []
    for k in steps:
        p = os.path.join(shard_dir, f"shard_{k}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run:  python scripts/build_dataset.py "
                             f"--step {k} --n-base 200 --shard-dir {shard_dir}")
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"]); gs.append(d["group"]); ss.append(d["snr"])
        names = list(d["names"])
    X = np.vstack(Xs); y = np.concatenate(ys)
    group = np.concatenate(gs); snr = np.concatenate(ss)

    bad = ~np.isfinite(X)
    if bad.any():
        for c in np.unique(np.where(bad)[1]):
            col = X[:, c]
            X[~np.isfinite(col), c] = np.median(col[np.isfinite(col)])
        print(f"repaired {bad.sum()} non-finite values")
    X = np.clip(X, -1e12, 1e12).astype(np.float32)

    np.savez_compressed(out, X=X, y=y, group=group, snr=snr,
                        names=np.array(names), fs=FS, f0=F0, ncyc=NCYC)
    lv = sorted(set(snr.tolist()), reverse=True)
    print(f"merged -> {out}   X={X.shape}  groups={len(np.unique(group))}  "
          f"classes={len(np.unique(y))}  "
          f"levels=[{', '.join(level_name(s) for s in lv)}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=200)
    ap.add_argument("--step", type=int, default=None,
                    help="level index 0..4 = 40/30/20/10/0 dB, 5 = clean")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="which shards to merge (default: all present)")
    ap.add_argument("--shard-dir", default="data/shards")
    ap.add_argument("--out", default="data/dataset.npz")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.merge:
        merge(a.shard_dir, a.out, a.steps)
    else:
        step(a.step, a.n_base, a.shard_dir, a.seed)
