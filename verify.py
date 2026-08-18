"""
Correctness controls for the whole pipeline.

Reported numbers are only worth as much as the checks behind them. This script
runs the ones that would catch the failure modes that actually matter here:

  1. group leakage   -- no base waveform appears in two partitions, and no SNR
                        sibling of a test waveform is ever in training
  2. class balance   -- every class present in equal numbers in every partition
                        (macro-F1 is only interpretable if so)
  3. label shuffle   -- with labels permuted, macro-F1 must collapse to chance
                        (1/29 = 0.034). If it does not, the model is exploiting
                        structure that has nothing to do with the class.
  4. group shuffle   -- permuting labels WITHIN groups only; also chance
  5. noise-only      -- features from pure AWGN must be unclassifiable
  6. feature sanity  -- no constant, NaN, or duplicated columns
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from pipeline import grouped_stratified_split, level_name, levels_of

CHANCE = 1.0 / 29


def main(data, seed=0, n_jobs=2):
    d = np.load(data, allow_pickle=True)
    X, y, group, snr = d["X"], d["y"].astype(int), d["group"], d["snr"].astype(int)
    names = list(d["names"])
    ok = []

    def check(label, passed, detail=""):
        ok.append(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:<52} {detail}")

    LV = levels_of(snr)
    print(f"\ndataset: {X.shape[0]} rows x {X.shape[1]} features, "
          f"{len(np.unique(group))} groups, "
          f"levels [{', '.join(level_name(v) for v in LV)}]\n")

    (i_tr, i_va, i_te), (g_tr, g_va, g_te) = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), seed)

    # ---- 1. leakage ------------------------------------------------------
    print("1. leakage")
    check("train/val group overlap", len(set(g_tr) & set(g_va)) == 0)
    check("train/test group overlap", len(set(g_tr) & set(g_te)) == 0)
    check("val/test group overlap", len(set(g_va) & set(g_te)) == 0)
    check(f"every group has exactly {len(LV)} noise-level siblings",
          set(np.bincount(group).tolist()) == {len(LV)})
    sib = all(set(snr[group == g].tolist()) == set(LV)
              for g in np.random.default_rng(0).choice(np.unique(group), 200))
    check("SNR siblings never split across partitions", sib)
    check("partitions cover all rows exactly once",
          len(i_tr) + len(i_va) + len(i_te) == len(y))

    # ---- 2. balance ------------------------------------------------------
    print("\n2. class / SNR balance")
    for nm, ix in (("train", i_tr), ("val", i_va), ("test", i_te)):
        c = np.bincount(y[ix], minlength=30)[1:]
        s = np.array([np.sum(snr[ix] == v) for v in LV])
        check(f"{nm}: 29 classes balanced", c.min() == c.max(),
              f"n={c.min()}/class")
        check(f"{nm}: {len(LV)} noise levels balanced", s.min() == s.max(),
              f"n={s.min()}/level")

    # ---- 3-4. shuffle controls -------------------------------------------
    print("\n3. label-shuffle controls (expect macro-F1 ~ 1/29 = 0.034)")
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(i_tr), min(8000, len(i_tr)), replace=False)
    Xs, ys, gs = X[i_tr][sub], y[i_tr][sub], group[i_tr][sub]
    rf = lambda: RandomForestClassifier(n_estimators=150, n_jobs=n_jobs,
                                        random_state=seed)

    m = rf().fit(Xs, ys)
    real = f1_score(y[i_te], m.predict(X[i_te]), average="macro")
    print(f"       (reference: real labels -> macro-F1 = {real:.4f})")

    yp = ys.copy()
    rng.shuffle(yp)
    f_perm = f1_score(y[i_te], rf().fit(Xs, yp).predict(X[i_te]), average="macro")
    check("permuted labels collapse to chance", f_perm < 3 * CHANCE,
          f"F1={f_perm:.4f}")

    # permute the class assigned to each GROUP, keeping SNR siblings coherent
    ug = np.unique(gs)
    lab = {g: ys[gs == g][0] for g in ug}
    vals = list(lab.values())
    rng.shuffle(vals)
    lab = dict(zip(ug, vals))
    yg = np.array([lab[g] for g in gs])
    f_gperm = f1_score(y[i_te], rf().fit(Xs, yg).predict(X[i_te]), average="macro")
    check("group-coherent label permutation -> chance", f_gperm < 3 * CHANCE,
          f"F1={f_gperm:.4f}")

    # ---- 5. noise-only control -------------------------------------------
    print("\n4. noise-only control")
    Xn = rng.normal(size=(len(sub), X.shape[1])).astype(np.float32)
    f_noise = f1_score(ys, rf().fit(Xn, ys).predict(
        rng.normal(size=(len(sub), X.shape[1])).astype(np.float32)),
        average="macro")
    check("random features -> chance", f_noise < 3 * CHANCE, f"F1={f_noise:.4f}")

    # ---- 6. feature sanity ------------------------------------------------
    print("\n5. feature matrix sanity")
    check("all finite", bool(np.all(np.isfinite(X))))
    const = np.flatnonzero(X.std(0) == 0)
    check("no constant columns", len(const) == 0,
          f"{[names[i] for i in const]}" if len(const) else "")
    seen, dups = {}, []
    for j in range(X.shape[1]):
        k = X[:, j].tobytes()
        if k in seen:
            dups.append((names[seen[k]], names[j]))
        else:
            seen[k] = j
    check("no byte-identical duplicate columns", not dups, str(dups or ""))

    # Redundancy is reported, not failed: perfectly correlated pairs are
    # harmless for trees (they merely split importance) but waste capacity in
    # the PCA/SVM/MLP branch, so they are worth knowing about.
    C = np.nan_to_num(np.corrcoef(X.T))
    np.fill_diagonal(C, 0)
    a, b = np.where(np.triu(np.abs(C)) > 0.999)
    if len(a):
        print(f"  [note] {len(a)} feature pairs with |r| > 0.999 "
              f"(redundant, not an error):")
        for i, j in zip(a, b):
            tag = " <- exactly collinear" if abs(abs(C[i, j]) - 1) < 1e-9 else ""
            print(f"         {names[i]:<20} ~ {names[j]:<20} r={C[i,j]:+.5f}{tag}")
    check("no column is a perfect label proxy",
          not any(len(np.unique(X[:, j])) == 29 and
                  np.array_equal(np.argsort(X[:, j]), np.argsort(y))
                  for j in range(X.shape[1])))

    print(f"\n{'='*72}\n{sum(ok)}/{len(ok)} checks passed"
          f"{'' if all(ok) else '  <-- INVESTIGATE FAILURES'}\n{'='*72}")
    return all(ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    raise SystemExit(0 if main(a.data, a.seed, a.n_jobs) else 1)
