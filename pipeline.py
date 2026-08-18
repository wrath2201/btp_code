"""
pipeline.py -- splits, base models, ensembles and evaluation.

Protocol
--------
1. Groups (= base waveforms) are split 70/15/15 into train / val / test,
   stratified by class. All 5 SNR copies of a waveform stay together, so no
   noise-augmented sibling of a test waveform is ever seen in training.
2. StratifiedGroupKFold(10) *inside the training partition* produces
   out-of-fold class probabilities for every base model. These OOF
   probabilities -- never in-fold predictions -- are what the stacking
   meta-learner is trained on, which is the only leakage-free way to fit a
   stacker.
3. The validation partition is used for early stopping and for choosing
   between the soft-voting and stacked ensembles.
4. The test partition is touched exactly once, at the end.

Base models (deliberately different inductive biases, for ensemble diversity):
    RF        bagged axis-aligned trees, low variance, no scaling needed
    LightGBM  boosted trees, fits residual structure the RF misses
    SVM-RBF   kernel margin on a PCA-reduced, quantile-normalised space
    MLP       dense non-linear map, learns feature interactions directly
"""

from __future__ import annotations

import argparse
import json
import os
import time
import gc
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

import lightgbm as lgb

warnings.filterwarnings("ignore")

N_CLASSES = 29
SNRS = [40, 30, 20, 10, 0]
CLEAN = 999          # sentinel for "no noise added"; see build_dataset.py


def levels_of(snr):
    """Noise levels actually present in the data, cleanest first."""
    return sorted(set(np.asarray(snr).tolist()), reverse=True)


def level_name(s):
    return "clean" if int(s) == CLEAN else f"{int(s)}dB"


# ==========================================================================
# splitting
# ==========================================================================
def grouped_stratified_split(y, group, fracs=(0.70, 0.15, 0.15), seed=0):
    """
    Split *groups* (base waveforms) into train/val/test, stratified by class.

    Because every group belongs to exactly one class and classes are balanced,
    stratification is achieved exactly by splitting each class's group list
    independently -- no approximation needed.
    """
    rng = np.random.default_rng(seed)
    g_first = {}
    for gi, yi in zip(group, y):
        g_first.setdefault(int(gi), int(yi))
    groups = np.array(sorted(g_first))
    glab = np.array([g_first[int(g)] for g in groups])

    tr, va, te = [], [], []
    for c in np.unique(glab):
        gcls = groups[glab == c]
        rng.shuffle(gcls)
        n = len(gcls)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        tr.append(gcls[:n_tr])
        va.append(gcls[n_tr:n_tr + n_va])
        te.append(gcls[n_tr + n_va:])
    tr, va, te = (np.sort(np.concatenate(z)) for z in (tr, va, te))

    m = lambda gs: np.isin(group, gs)
    idx = (np.flatnonzero(m(tr)), np.flatnonzero(m(va)), np.flatnonzero(m(te)))

    # leakage assertions -- these must never fire
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert sum(len(i) for i in idx) == len(y)
    return idx, (tr, va, te)


# ==========================================================================
# base models
# ==========================================================================
def make_models(seed=0, n_jobs=2, fast=False):
    """Four base learners with genuinely different inductive biases."""
    rf_trees = 150 if fast else 300
    lgb_iter = 120 if fast else 250
    mlp_iter = 120 if fast else 400

    return {
        # min_samples_leaf=2 roughly halves the node count. With 29 classes
        # each node stores a 29-vector, so a fully grown 300-tree forest on
        # 20k rows costs hundreds of MB -- this keeps it inside the sandbox.
        "rf": RandomForestClassifier(
            n_estimators=rf_trees, max_features="sqrt", min_samples_leaf=2,
            n_jobs=n_jobs, random_state=seed, class_weight=None),

        # multiclass LightGBM builds one tree per class per iteration, so cost
        # is n_estimators x 29. A higher learning rate with fewer, shallower
        # trees keeps the 10-fold OOF stage affordable without losing accuracy.
        "lgbm": lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES,
            n_estimators=lgb_iter, learning_rate=0.12, num_leaves=31,
            max_depth=7, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            n_jobs=n_jobs, random_state=seed, verbose=-1),

        # QuantileTransformer, not StandardScaler: several features are ratios
        # with very heavy tails (e.g. band-energy / median on near-clean rows).
        # A monotone rank transform makes them usable by kernel and dense
        # models without discarding information.
        "svm": Pipeline([
            ("qt", QuantileTransformer(output_distribution="normal",
                                       n_quantiles=1000, random_state=seed)),
            ("pca", PCA(n_components=50, random_state=seed)),
            ("clf", SVC(C=10.0, gamma="scale", kernel="rbf",
                        decision_function_shape="ovr", probability=False,
                        random_state=seed)),
        ]),

        "mlp": Pipeline([
            ("qt", QuantileTransformer(output_distribution="normal",
                                       n_quantiles=1000, random_state=seed)),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                  learning_rate_init=1e-3, batch_size=256,
                                  max_iter=mlp_iter, early_stopping=True,
                                  n_iter_no_change=15, validation_fraction=0.12,
                                  random_state=seed)),
        ]),
    }


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predict_proba(name, model, X):
    """
    Uniform probability interface.

    SVC is fitted with probability=False because Platt scaling costs an internal
    5-fold refit (~5x). Its one-vs-rest decision function is converted to a
    probability-like vector with a softmax -- monotone in the margin, adequate
    for soft voting, and re-calibrated anyway by the stacking meta-learner.
    """
    if name == "svm":
        return _softmax(model.decision_function(X) * 2.0)
    return model.predict_proba(X)


# ==========================================================================
# metrics
# ==========================================================================
def scores(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
    }


def per_snr(y_true, y_pred, snr):
    return {int(s): scores(y_true[snr == s], y_pred[snr == s])
            for s in sorted(set(snr.tolist()), reverse=True)}


# ==========================================================================
# main
# ==========================================================================
def run(data_path, out_path, n_folds=10, seed=0, n_jobs=2, fast=False,
        max_new_folds=None):
    d = np.load(data_path, allow_pickle=True)
    X, y, group, snr = d["X"], d["y"].astype(int), d["group"], d["snr"].astype(int)
    names = list(d["names"])
    LV = levels_of(snr)                 # noise levels present, cleanest first
    print(f"data: {X.shape[0]} rows x {X.shape[1]} features, "
          f"{len(np.unique(group))} base waveforms, "
          f"{len(np.unique(y))} classes, "
          f"levels [{', '.join(level_name(v) for v in LV)}]\n")

    # ---- 1. split ---------------------------------------------------------
    (i_tr, i_va, i_te), (g_tr, g_va, g_te) = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), seed)
    print(f"[split] groups  train={len(g_tr)}  val={len(g_va)}  test={len(g_te)}")
    print(f"[split] rows    train={len(i_tr)}  val={len(i_va)}  test={len(i_te)}")
    for nm, ix in (("train", i_tr), ("val", i_va), ("test", i_te)):
        cnt = np.bincount(y[ix], minlength=30)[1:]
        sc = [int((snr[ix] == v).sum()) for v in LV]
        print(f"         {nm:<5} class counts {cnt.min()}-{cnt.max()} "
              f"(balanced={cnt.min()==cnt.max()}), per-level {sorted(sc)}")
    print("[split] group overlap between partitions: 0 (asserted)\n")

    Xtr, ytr, gtr = X[i_tr], y[i_tr], group[i_tr]
    Xva, yva, sva = X[i_va], y[i_va], snr[i_va]
    Xte, yte, ste = X[i_te], y[i_te], snr[i_te]

    model_names = ["rf", "lgbm", "svm", "mlp"]
    timings, results = {}, {}

    # ---- 2. out-of-fold probabilities on the training partition -----------
    print(f"[oof] StratifiedGroupKFold({n_folds}) on the training partition")
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(Xtr, ytr, groups=gtr))
    oof = {m: np.zeros((len(ytr), N_CLASSES)) for m in model_names}
    fold_f1 = {m: [] for m in model_names}

    # Checkpoint after every fold. The 10-fold OOF stage takes ~25 min, which
    # is longer than this sandbox reliably keeps a process alive, so each fold
    # is persisted and completed folds are skipped on restart.
    ck = out_path.replace(".json", "_oof_ckpt.npz")
    done = set()
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        for m in model_names:
            oof[m] = z[f"oof_{m}"]
        fold_f1 = {m: list(z[f"f1_{m}"]) for m in model_names}
        done = set(z["done"].tolist())
        print(f"  resumed from checkpoint: folds {sorted(done)} already done")

    t_oof = time.perf_counter()
    n_new = 0
    for k, (a, b) in enumerate(folds):
        assert len(set(gtr[a]) & set(gtr[b])) == 0, "fold leakage"
        if k in done:
            continue
        if max_new_folds is not None and n_new >= max_new_folds:
            print(f"  stopping after {n_new} new fold(s); "
                  f"{n_folds - len(done)} remaining. Re-run to continue.")
            return None
        n_new += 1
        for m in model_names:
            t0 = time.perf_counter()
            mdl = make_models(seed + k, n_jobs, fast)[m]
            mdl.fit(Xtr[a], ytr[a])
            p = predict_proba(m, mdl, Xtr[b])
            oof[m][b] = p
            fold_f1[m].append(f1_score(ytr[b], p.argmax(1) + 1, average="macro"))
            timings.setdefault(f"oof_{m}", []).append(time.perf_counter() - t0)
            del mdl, p
            gc.collect()
        done.add(k)
        np.savez_compressed(ck, done=np.array(sorted(done)),
                            **{f"oof_{m}": oof[m] for m in model_names},
                            **{f"f1_{m}": np.array(fold_f1[m]) for m in model_names})
        print(f"  fold {k+1}/{n_folds}  " + "  ".join(
            f"{m}={fold_f1[m][-1]:.4f}" for m in model_names) +
            f"   [{time.perf_counter()-t_oof:.0f}s]", flush=True)

    print("\n[oof] cross-validated macro-F1 (mean +/- std over "
          f"{n_folds} folds)")
    for m in model_names:
        v = np.array(fold_f1[m])
        print(f"       {m:<5} {v.mean():.4f} +/- {v.std():.4f}")
        results[f"cv_{m}"] = {"mean": float(v.mean()), "std": float(v.std()),
                              "folds": [float(x) for x in v]}

    # ---- 3. final base models on the full training partition --------------
    # Base-model predictions are checkpointed too, so the ensemble stage can be
    # re-run (or resumed) without paying for another round of refits.
    print("\n[fit] refitting base models on the full training partition")
    ckb = out_path.replace(".json", "_base_ckpt.npz")
    cached = np.load(ckb, allow_pickle=True) if os.path.exists(ckb) else None
    fitted, P_va, P_te = {}, {}, {}
    for m in model_names:
        t0 = time.perf_counter()
        if cached is not None and f"va_{m}" in cached:
            P_va[m], P_te[m] = cached[f"va_{m}"], cached[f"te_{m}"]
            if m == "rf":
                imp_cached = cached["rf_importance"]
        else:
            mdl = make_models(seed, n_jobs, fast)[m]
            mdl.fit(Xtr, ytr)
            fitted[m] = mdl
            P_va[m] = predict_proba(m, mdl, Xva)
            P_te[m] = predict_proba(m, mdl, Xte)
            if m == "rf":
                imp_cached = mdl.feature_importances_
            del mdl
            gc.collect()
        timings[f"fit_{m}"] = time.perf_counter() - t0
    np.savez_compressed(ckb, rf_importance=imp_cached,
                        **{f"va_{m}": P_va[m] for m in model_names},
                        **{f"te_{m}": P_te[m] for m in model_names})

    for m in model_names:
        s_va = scores(yva, P_va[m].argmax(1) + 1)
        s_te = scores(yte, P_te[m].argmax(1) + 1)
        print(f"       {m:<5} val F1={s_va['macro_f1']:.4f}  "
              f"test F1={s_te['macro_f1']:.4f}  "
              f"test bal-acc={s_te['balanced_acc']:.4f}  "
              f"[{timings[f'fit_{m}']:.0f}s]")
        results[f"base_{m}"] = {"val": s_va, "test": s_te,
                                "test_per_snr": per_snr(yte, P_te[m].argmax(1)+1, ste)}

    # ---- 4A. soft-voting ensembles ----------------------------------------
    # Equal-weight averaging is only optimal when the members are comparably
    # strong. Here the SVM is clearly weakest, so uniform voting drags the
    # ensemble below the best single model. Weights are therefore fitted on the
    # OOF probabilities -- data the final models never trained on -- by random
    # search over the simplex. Geometric (log-probability) averaging is also
    # tried: it is less forgiving of a member that is confidently wrong.
    sv_va = np.mean([P_va[m] for m in model_names], 0)
    sv_te = np.mean([P_te[m] for m in model_names], 0)

    Ptr = np.stack([oof[m] for m in model_names])          # (M, n, C)
    rs = np.random.default_rng(seed)
    best_w, best_s = np.ones(len(model_names)) / len(model_names), -1.0
    for _ in range(400):
        w = rs.dirichlet(np.ones(len(model_names)))
        s = f1_score(ytr, np.tensordot(w, Ptr, 1).argmax(1) + 1, average="macro")
        if s > best_s:
            best_s, best_w = s, w
    print(f"\n[ens] OOF-fitted voting weights  " +
          "  ".join(f"{m}={w:.3f}" for m, w in zip(model_names, best_w)) +
          f"   (OOF macro-F1 {best_s:.4f})")
    results["vote_weights"] = dict(zip(model_names, best_w.round(4).tolist()))

    wv_va = np.tensordot(best_w, np.stack([P_va[m] for m in model_names]), 1)
    wv_te = np.tensordot(best_w, np.stack([P_te[m] for m in model_names]), 1)

    lg = lambda P: np.log(np.clip(P, 1e-9, 1))
    gm_va = np.exp(np.tensordot(best_w, np.stack([lg(P_va[m]) for m in model_names]), 1))
    gm_te = np.exp(np.tensordot(best_w, np.stack([lg(P_te[m]) for m in model_names]), 1))

    # ---- 4B. stacked ensemble --------------------------------------------
    # meta-features = concatenated OOF probabilities + estimated SNR.
    # snr_est_db is a *measured* feature, available at inference time; the true
    # SNR label is not, so using it would be an oracle. Both are reported.
    j_snr = names.index("snr_est_db")

    def meta(P_dict, Xsrc, order):
        return np.hstack([P_dict[m] for m in order]
                         + [Xsrc[:, [j_snr]].astype(np.float64)])

    Mtr = meta(oof, Xtr, model_names)
    Mva = meta(P_va, Xva, model_names)
    Mte = meta(P_te, Xte, model_names)

    stk = Pipeline([("qt", QuantileTransformer(output_distribution="normal",
                                               n_quantiles=1000,
                                               random_state=seed)),
                    ("lr", LogisticRegression(max_iter=3000, C=1.0,
                                              multi_class="multinomial"))])
    t0 = time.perf_counter()
    stk.fit(Mtr, ytr)
    timings["fit_stack"] = time.perf_counter() - t0
    st_va = stk.predict_proba(Mva)
    st_te = stk.predict_proba(Mte)

    ENS = {"soft_vote": (sv_va, sv_te), "weighted_vote": (wv_va, wv_te),
           "geometric_vote": (gm_va, gm_te), "stacked": (st_va, st_te)}
    print("\n[ens] ensemble comparison")
    for nm, (pv, pt) in ENS.items():
        s_va = scores(yva, pv.argmax(1) + 1)
        s_te = scores(yte, pt.argmax(1) + 1)
        print(f"       {nm:<15} val F1={s_va['macro_f1']:.4f}   "
              f"test F1={s_te['macro_f1']:.4f}   "
              f"test bal-acc={s_te['balanced_acc']:.4f}")
        results[nm] = {"val": s_va, "test": s_te,
                       "test_per_snr": per_snr(yte, pt.argmax(1) + 1, ste)}

    # selection is made on VALIDATION only; test is never used to choose
    best = max(ENS, key=lambda k: results[k]["val"]["macro_f1"])
    print(f"       -> selected on VALIDATION: {best}")
    results["selected_ensemble"] = best
    best_te = ENS[best][1]

    # ---- 5. per-SNR breakdown --------------------------------------------
    print(f"\n[eval] test-set macro-F1 by SNR level")
    hdr = "       model          " + "".join(f"{level_name(s):>10}" for s in LV) + "     all"
    print(hdr)
    for m in model_names + list(ENS):
        key = f"base_{m}" if m in model_names else m
        r = results[key]["test_per_snr"]
        row = "".join(f"{r[s]['macro_f1']:>10.4f}" for s in LV)
        print(f"       {m:<14}{row}{results[key]['test']['macro_f1']:>8.4f}")

    # ---- 6. per-class x per-SNR heatmap data ------------------------------
    yp = best_te.argmax(1) + 1
    heat = np.zeros((N_CLASSES, len(LV)))
    for ci in range(N_CLASSES):
        for si, s in enumerate(LV):
            m = (yte == ci + 1) & (ste == s)
            heat[ci, si] = (yp[m] == ci + 1).mean() if m.any() else np.nan
    results["per_class_per_snr_recall"] = heat.tolist()

    # confusion matrices at the two extremes present in the data
    for s in dict.fromkeys((LV[0], LV[-1])):
        m = ste == s
        cm = confusion_matrix(yte[m], yp[m], labels=np.arange(1, 30))
        results[f"confusion_snr{s}"] = cm.tolist()

    results["confusion_all"] = confusion_matrix(
        yte, yp, labels=np.arange(1, 30)).tolist()
    results["timings"] = {k: (float(np.mean(v)) if isinstance(v, list) else float(v))
                          for k, v in timings.items()}
    results["levels"] = [int(v) for v in LV]
    results["config"] = {"n_folds": n_folds, "seed": seed, "fast": fast,
                         "n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
                         "n_groups": int(len(np.unique(group))),
                         "split_rows": [len(i_tr), len(i_va), len(i_te)],
                         "split_groups": [len(g_tr), len(g_va), len(g_te)]}

    # ---- 7. feature importance (RF, for the report) -----------------------
    imp = imp_cached
    order = np.argsort(imp)[::-1][:25]
    results["top_features"] = [(names[j], float(imp[j])) for j in order]
    print("\n[imp] top 12 features by RF importance")
    for j in order[:12]:
        print(f"       {names[j]:<22} {imp[j]:.4f}")

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nsaved -> {out_path}")

    # save artefacts needed by the unseen-SNR study and the plots
    np.savez_compressed(out_path.replace(".json", "_preds.npz"),
                        yte=yte, ste=ste, yp=yp,
                        **{f"P_{m}": P_te[m] for m in model_names},
                        **{f"E_{k}": v[1] for k, v in ENS.items()})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-new-folds", type=int, default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    run(a.data, a.out, a.folds, a.seed, a.n_jobs, a.fast, a.max_new_folds)
    print(f"total wall time: {(time.perf_counter()-t0)/60:.1f} min")
