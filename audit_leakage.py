"""
audit_leakage.py -- quantify how much a naive split inflates the score.

This is the single most important audit for this problem, because the
inflation it measures is the most likely explanation for optimistic 29-class
numbers reported on this generator.

The dataset contains 5 rows per base waveform, one per SNR level. Those 5 rows
are the SAME disturbance with different noise draws, so their feature vectors
are strongly correlated. Two ways to split it:

  GROUP split (correct)   whole base waveforms go to train/val/test, so all 5
                          SNR copies stay together. A test waveform has never
                          been seen in any form.

  ROW split (naive)       rows are shuffled independently. A waveform's 40 dB
                          copy can sit in training while its 10 dB copy is in
                          test. The model can recognise the specific waveform
                          instead of the disturbance class.

Both use identical proportions, identical models and identical seeds. The only
difference is the unit of splitting, so the gap between them is pure leakage.

A third variant is included for completeness:

  ROW split, single SNR   what you get if you never augment at all -- useful
                          for comparing against papers that report one SNR.

Usage:  python audit_leakage.py --data data/dataset.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from pipeline import (grouped_stratified_split, level_name, levels_of, make_models,
                      predict_proba, scores)

MODELS = ["rf", "lgbm"]          # two strongest; enough to show the effect


def row_stratified_split(y, fracs=(0.70, 0.15, 0.15), seed=0):
    """Naive split: shuffle ROWS, stratify by class, ignore the group id."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fracs[0], stratify=y,
                                random_state=seed)
    va, te = train_test_split(rest, train_size=fracs[1] / (fracs[1] + fracs[2]),
                              stratify=y[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def evaluate(X, y, snr, i_tr, i_te, seed, n_jobs, tag):
    out = {}
    P = []
    for m in MODELS:
        mdl = make_models(seed, n_jobs, fast=True)[m]
        mdl.fit(X[i_tr], y[i_tr])
        p = predict_proba(m, mdl, X[i_te])
        P.append(p)
        out[m] = float(f1_score(y[i_te], p.argmax(1) + 1, average="macro"))
        del mdl
    yp = np.mean(P, 0).argmax(1) + 1
    out["vote"] = float(f1_score(y[i_te], yp, average="macro"))
    out["per_snr"] = {int(s): float(f1_score(y[i_te][snr[i_te] == s],
                                             yp[snr[i_te] == s],
                                             average="macro"))
                      for s in levels_of(snr) if (snr[i_te] == s).any()}
    out["n_train"], out["n_test"] = int(len(i_tr)), int(len(i_te))
    print(f"  {tag:<34} " + "  ".join(f"{m}={out[m]:.4f}" for m in MODELS)
          + f"   vote={out['vote']:.4f}", flush=True)
    return out


def main(data, seed=0, n_jobs=2, out_path="leakage_audit.json"):
    d = np.load(data, allow_pickle=True)
    X, y, group, snr = d["X"], d["y"].astype(int), d["group"], d["snr"].astype(int)
    res = {}

    print(f"\ndataset: {X.shape[0]} rows, {len(np.unique(group))} base waveforms, "
          f"{len(levels_of(snr))} noise levels\n")
    print("test macro-F1 (fast model configs, identical seeds):\n")

    # ---- A. correct: split by base waveform -----------------------------
    (i_tr, i_va, i_te), (g_tr, _, g_te) = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), seed)
    assert len(set(g_tr) & set(g_te)) == 0
    res["group_split"] = evaluate(X, y, snr, i_tr, i_te, seed, n_jobs,
                                  "A. GROUP split (correct)")

    # ---- B. naive: split by row -----------------------------------------
    r_tr, r_va, r_te = row_stratified_split(y, (0.70, 0.15, 0.15), seed)
    shared = len(set(group[r_tr]) & set(group[r_te]))
    res["row_split"] = evaluate(X, y, snr, r_tr, r_te, seed, n_jobs,
                                "B. ROW split (naive, leaks)")
    res["row_split"]["waveforms_shared_train_test"] = int(shared)
    res["row_split"]["frac_test_waveforms_seen"] = float(
        len(set(group[r_te]) & set(group[r_tr])) / len(set(group[r_te])))

    # ---- C. single SNR level, row split ---------------------------------
    for s in [v for v in (40, 20, 0) if (snr == v).any()]:
        m = snr == s
        Xs, ys, gs = X[m], y[m], group[m]
        t_tr, t_va, t_te = row_stratified_split(ys, (0.70, 0.15, 0.15), seed)
        res[f"single_snr_{s}"] = evaluate(Xs, ys, np.full(len(ys), s),
                                          t_tr, t_te, seed, n_jobs,
                                          f"C. single SNR {s} dB only")

    a, b = res["group_split"]["vote"], res["row_split"]["vote"]
    res["inflation"] = float(b - a)
    print(f"\n{'='*78}")
    print(f"  correct (group) split      : {a:.4f}")
    print(f"  naive (row) split          : {b:.4f}")
    print(f"  LEAKAGE INFLATION          : {b - a:+.4f}  "
          f"({100*(b-a)/max(a,1e-9):+.1f}% relative)")
    print(f"  test waveforms also in train: "
          f"{100*res['row_split']['frac_test_waveforms_seen']:.1f}%")
    print(f"{'='*78}\n")
    print("per-SNR, group vs row split:")
    print(f"  {'SNR':>6}{'group':>10}{'row':>10}{'delta':>10}")
    for s in levels_of(snr):
        ga = res["group_split"]["per_snr"][s]
        rb = res["row_split"]["per_snr"][s]
        print(f"  {level_name(s):>6}{ga:>10.4f}{rb:>10.4f}{rb-ga:>+10.4f}")

    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="leakage_audit.json")
    a = ap.parse_args()
    main(a.data, a.seed, a.n_jobs, a.out)
