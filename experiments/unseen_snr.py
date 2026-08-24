"""
Leave-one-SNR-out stress test.

The main protocol trains on all 5 SNR levels, so its per-SNR test metrics
measure "how hard is this noise level", not "can the model handle a noise level
it has never seen". This script answers the second question: for each level s,
train on the other four and test on held-out waveforms at s only.

The waveform-level group split is identical to the main run, so the only thing
that changes is which SNR levels were available in training. The reference
column is the main run's own per-SNR test score, i.e. the same waveforms scored
by a model that DID see level s in training. The gap between them is the cost
of SNR extrapolation.

40 dB and 0 dB are extrapolation (outside the training range); 30/20/10 dB are
interpolation (inside it). Those two regimes should behave very differently.

Run one level per invocation (--only k), then --merge.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.pipeline import SNRS, grouped_stratified_split, make_models, predict_proba

MODELS = ["rf", "lgbm", "svm", "mlp"]


def one(k, data, shard_dir, seed=0, n_jobs=2):
    s = SNRS[k]
    d = np.load(data, allow_pickle=True)
    X, y, group, snr = d["X"], d["y"].astype(int), d["group"], d["snr"].astype(int)

    (i_tr, i_va, i_te), _ = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), seed)
    # train+val used together: no early stopping or ensemble selection happens
    # in this study, so the validation partition carries no special role here
    i_dev = np.concatenate([i_tr, i_va])

    tr = i_dev[snr[i_dev] != s]           # the other four levels
    te = i_te[snr[i_te] == s]             # held-out level AND held-out waveforms
    assert len(set(group[tr]) & set(group[te])) == 0, "group leakage"
    assert set(snr[tr].tolist()) == set(SNRS) - {s}, "SNR leakage"

    row, P = {}, []
    for m in MODELS:
        mdl = make_models(seed, n_jobs, fast=True)[m]
        mdl.fit(X[tr], y[tr])
        p = predict_proba(m, mdl, X[te])
        P.append(p)
        row[m] = float(f1_score(y[te], p.argmax(1) + 1, average="macro"))
        print(f"  {m:<5} {row[m]:.4f}", flush=True)
        del mdl

    ens = np.mean(P, 0).argmax(1) + 1
    row["vote"] = float(f1_score(y[te], ens, average="macro"))
    row["balanced_acc"] = float(balanced_accuracy_score(y[te], ens))
    row["n_train"] = int(len(tr))
    row["n_test"] = int(len(te))

    os.makedirs(shard_dir, exist_ok=True)
    with open(os.path.join(shard_dir, f"unseen_{k}.json"), "w") as fh:
        json.dump(row, fh)
    print(f"SNR {s} dB held out -> vote macro-F1 {row['vote']:.4f}")


def merge(shard_dir, out, results="results.json"):
    R = json.load(open(results))
    sel = R["selected_ensemble"]
    res = {}
    print(f"{'held out':<10}{'regime':<16}" + "".join(f"{m:>9}" for m in MODELS)
          + f"{'vote':>9}{'ref':>9}{'gap':>9}")
    print("-" * 96)
    for k, s in enumerate(SNRS):
        p = os.path.join(shard_dir, f"unseen_{k}.json")
        if not os.path.exists(p):
            continue
        row = json.load(open(p))
        row["reference"] = R[sel]["test_per_snr"][str(s)]["macro_f1"]
        row["gap"] = row["reference"] - row["vote"]
        regime = "extrapolation" if s in (max(SNRS), min(SNRS)) else "interpolation"
        res[int(s)] = row
        print(f"{s:>7} dB  {regime:<16}"
              + "".join(f"{row[m]:>9.4f}" for m in MODELS)
              + f"{row['vote']:>9.4f}{row['reference']:>9.4f}{row['gap']:>+9.4f}")
    print("\ngap = (trained on all 5 levels) - (that level held out); "
          "larger = more SNR-specific overfitting")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--only", type=int, default=None, help="SNR index 0..4")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard-dir", default="data/unseen")
    ap.add_argument("--out", default="unseen_snr.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    if a.merge:
        merge(a.shard_dir, a.out)
    else:
        one(a.only, a.data, a.shard_dir, a.seed, a.n_jobs)
