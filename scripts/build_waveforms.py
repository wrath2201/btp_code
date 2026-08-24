"""
build_waveforms.py -- generate the raw-waveform dataset for DASNet.

The feature pipeline (build_dataset.py) never stores waveforms, only 191
features. DASNet consumes raw waveforms, so this script regenerates them with
BIT-IDENTICAL noise realizations to the feature dataset:

  * same clean base waveforms   (pqmodel, seed 20260807)
  * same per-level RNG          (default_rng(seed + 1000 + k))
  * same batch order            (BATCH = 250, float64 conversion before AWGN)

so every row here corresponds 1:1 to a row of data/dataset.npz -- same group
ids, same labels, same noise draw. Any accuracy difference between DASNet and
the ensemble is therefore attributable to the model, not the data.

Output: data/waveforms.npz
    W     (n, 1280) float32   raw waveforms
    y     (n,)      int16     class labels 1..29
    group (n,)      int32     base-waveform id (split unit)
    snr   (n,)      int16     40/30/20/10/0 dB, 999 = clean
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.build_dataset import (BATCH, CLEAN, FS, F0, LEVELS, NCYC, SEED,
                                   clean_set, level_name)
from src.pqmodel import add_awgn


def build(n_base: int, out: str, seed: int = SEED) -> None:
    n_pts = int(round(FS / F0 * NCYC))
    Ws, ys, gs, ss = [], [], [], []

    for k, s in enumerate(LEVELS):
        t0 = time.perf_counter()
        clean, y, g = clean_set(n_base, seed)          # deterministic, cheap
        n = clean.shape[0]
        W = np.empty((n, n_pts), dtype=np.float32)

        if s == CLEAN:
            W[:] = clean
        else:
            # identical RNG stream + batch order as build_dataset.step()
            rng = np.random.default_rng(seed + 1000 + k)
            for a in range(0, n, BATCH):
                b = min(a + BATCH, n)
                c = clean[a:b].astype(np.float64)
                W[a:b] = add_awgn(c, s, rng).astype(np.float32)

        Ws.append(W)
        ys.append(y)
        gs.append(g)
        ss.append(np.full(n, s, dtype=np.int16))
        print(f"  {level_name(s):>6}: {W.shape} "
              f"[{time.perf_counter() - t0:.1f}s]", flush=True)

    W = np.vstack(Ws)
    y = np.concatenate(ys)
    group = np.concatenate(gs)
    snr = np.concatenate(ss)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, W=W, y=y, group=group, snr=snr,
                        fs=FS, f0=F0, ncyc=NCYC)
    print(f"saved -> {out}   W={W.shape}  "
          f"groups={len(np.unique(group))}  classes={len(np.unique(y))}  "
          f"levels={sorted(set(snr.tolist()), reverse=True)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=200)
    ap.add_argument("--out", default="data/waveforms.npz")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    build(a.n_base, a.out, a.seed)
