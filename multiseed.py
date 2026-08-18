"""
multiseed.py -- confidence intervals, so you can tell a real difference from noise.

A single run gives a point estimate. Without a spread you cannot say whether
"weighted vote beats LightGBM by 0.008" is a finding or a coin flip. There are
two independent sources of variation and they answer different questions:

  --mode split   (cheap, default)
      Same waveforms, different 70/15/15 partitions. Answers: "how much does
      my score depend on WHICH waveforms landed in test?" This is the CI you
      want when comparing two models on one dataset.

  --mode data    (expensive: ~8 min of feature extraction per seed)
      Fresh waveforms from a new generator seed, rebuilt end to end. Answers:
      "would I get this number again on a new draw from the model?" This is the
      CI you want when quoting a headline result.

Usage
    python multiseed.py --mode split --seeds 0 1 2 3 4
    python multiseed.py --mode data  --seeds 1 2 3     # rebuilds each time
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
from sklearn.metrics import f1_score

from pipeline import (grouped_stratified_split, level_name, levels_of,
                      make_models, predict_proba)

MODELS = ["rf", "lgbm", "svm", "mlp"]


def one_run(X, y, group, snr, split_seed, model_seed, n_jobs):
    (i_tr, i_va, i_te), (g_tr, _, g_te) = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), split_seed)
    assert len(set(g_tr) & set(g_te)) == 0

    row, Pv, Pt = {}, [], []
    for m in MODELS:
        mdl = make_models(model_seed, n_jobs, fast=True)[m]
        mdl.fit(X[i_tr], y[i_tr])
        Pv.append(predict_proba(m, mdl, X[i_va]))
        Pt.append(predict_proba(m, mdl, X[i_te]))
        row[m] = float(f1_score(y[i_te], Pt[-1].argmax(1) + 1, average="macro"))
        del mdl

    # (1) equal-weight soft vote -- every model counts the same
    row["vote_equal"] = float(f1_score(
        y[i_te], np.mean(Pt, 0).argmax(1) + 1, average="macro"))

    # (2) weighted soft vote. Weights are fitted on the VALIDATION partition,
    # never on test, so this is an honest estimate of the weighted ensemble --
    # the variant the main pipeline actually selects.
    Pv, Pt = np.stack(Pv), np.stack(Pt)
    rs = np.random.default_rng(split_seed)
    best_w, best_s = np.ones(len(MODELS)) / len(MODELS), -1.0
    for _ in range(300):
        w = rs.dirichlet(np.ones(len(MODELS)))
        s_ = f1_score(y[i_va], np.tensordot(w, Pv, 1).argmax(1) + 1,
                      average="macro")
        if s_ > best_s:
            best_s, best_w = s_, w
    ypw = np.tensordot(best_w, Pt, 1).argmax(1) + 1
    row["vote_weighted"] = float(f1_score(y[i_te], ypw, average="macro"))
    row["_weights"] = best_w.round(4).tolist()

    # per-level breakdown of the weighted vote, over whatever levels exist
    for s in levels_of(snr):
        k = snr[i_te] == s
        row[f"lvl_{level_name(s)}"] = float(
            f1_score(y[i_te][k], ypw[k], average="macro"))
    return row


def summarise(rows, out_path, mode):
    keys = [k for k in rows[0] if not k.startswith("_")]
    print(f"\n{'='*78}\n{len(rows)} runs, mode = {mode}\n{'='*78}")
    print(f"{'metric':<16}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}"
          f"{'95% CI':>20}")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        # t-based CI on the mean; with n<30 the normal approximation is wrong
        from scipy import stats
        h = (stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
             if len(v) > 1 else 0.0)
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))
                      if len(v) > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max()),
                      "ci95": [float(v.mean() - h), float(v.mean() + h)],
                      "runs": v.tolist()}
        print(f"{k:<16}{v.mean():>9.4f}{summary[k]['std']:>9.4f}"
              f"{v.min():>9.4f}{v.max():>9.4f}"
              f"   [{v.mean()-h:.4f}, {v.mean()+h:.4f}]")

    # Paired comparisons: does either ensemble really beat the best single
    # model? Pairing is valid because every method saw the identical splits,
    # which removes split-to-split variance from the comparison.
    best = max(MODELS, key=lambda m: summary[m]["mean"])
    print(f"\npaired comparisons against the best single model ({best}):")
    from scipy import stats
    for ens in ("vote_equal", "vote_weighted"):
        d = np.array([r[ens] - r[best] for r in rows])
        line = (f"  {ens:<14} mean {d.mean():+.4f}   "
                f"wins {int((d > 0).sum())}/{len(d)}")
        if len(d) > 1 and d.std(ddof=1) > 0:
            t, p = stats.ttest_rel([r[ens] for r in rows],
                                   [r[best] for r in rows])
            verdict = ("BETTER (p<0.05)" if p < 0.05 and d.mean() > 0 else
                       "WORSE (p<0.05)" if p < 0.05 else "not significant")
            line += f"   p={p:.4f}  -> {verdict}"
            summary[f"{ens}_vs_best"] = {
                "best_single": best, "mean_diff": float(d.mean()),
                "p_value": float(p), "wins": int((d > 0).sum())}
        print(line)

    w = np.array([r["_weights"] for r in rows])
    print(f"\nfitted vote weights (mean over runs): " +
          "  ".join(f"{m}={v:.3f}" for m, v in zip(MODELS, w.mean(0))))
    summary["_mean_weights"] = dict(zip(MODELS, w.mean(0).round(4).tolist()))
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nsaved -> {out_path}")


def main(a):
    rows = []
    if a.mode == "split":
        d = np.load(a.data, allow_pickle=True)
        X, y = d["X"], d["y"].astype(int)
        group, snr = d["group"], d["snr"].astype(int)
        for s in a.seeds:
            r = one_run(X, y, group, snr, split_seed=s, model_seed=0,
                        n_jobs=a.n_jobs)
            rows.append(r)
            print(f"split seed {s}: "
                  + "  ".join(f"{m}={r[m]:.4f}" for m in MODELS)
                  + f"  | equal={r['vote_equal']:.4f}"
                    f"  weighted={r['vote_weighted']:.4f}", flush=True)
    else:
        for s in a.seeds:
            for k in range(5):
                subprocess.run([sys.executable, "build_dataset.py", "--step",
                                str(k), "--n-base", str(a.n_base), "--seed",
                                str(20260807 + s), "--shard-dir",
                                f"data/shards_s{s}"], check=True)
            subprocess.run([sys.executable, "build_dataset.py", "--merge",
                            "--shard-dir", f"data/shards_s{s}",
                            "--out", f"data/dataset_s{s}.npz"], check=True)
            d = np.load(f"data/dataset_s{s}.npz", allow_pickle=True)
            r = one_run(d["X"], d["y"].astype(int), d["group"],
                        d["snr"].astype(int), split_seed=0, model_seed=0,
                        n_jobs=a.n_jobs)
            rows.append(r)
            print(f"data seed {s}: vote={r['vote']:.4f}", flush=True)

    summarise(rows, a.out, a.mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["split", "data"], default="split")
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-base", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="multiseed.json")
    main(ap.parse_args())
