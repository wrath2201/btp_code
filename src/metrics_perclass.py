"""
metrics_perclass.py -- per-class x per-SNR classification metrics.

Everything is derived from the per-SNR confusion matrix, so all five metric
families are mutually consistent by construction:

    for a fixed noise level s, build C_s = confusion(y_true, y_pred) over the
    rows with snr == s, then read one-vs-rest 2x2 tables off its rows/columns.

Per class c at level s, with N = number of test rows at level s:

    TP = C[c,c]                     FP = C[:,c].sum() - TP
    FN = C[c,:].sum() - TP          TN = N - TP - FP - FN

    accuracy_c  = (TP + TN) / N            one-vs-rest accuracy
    precision_c = TP / (TP + FP)
    recall_c    = TP / (TP + FN)
    f1_c        = 2 P R / (P + R)
    kappa_c     = 2 (TP*TN - FN*FP)
                  / ((TP+FP)(FP+TN) + (TP+FN)(FN+TN))    Cohen's kappa, 2x2

Aggregates at level s:

    macro_f1        = mean_c f1_c            (the paper's primary metric)
    macro_precision = mean_c precision_c
    macro_recall     = mean_c recall_c       (= balanced accuracy)
    overall_accuracy = trace(C) / N
    overall_kappa    = multiclass Cohen's kappa on C

Label convention
----------------
This module works exclusively in 1..29 label space. `normalise_labels` accepts
either 0..28 or 1..29 and returns 1..29, because the training scripts in this
repo disagree: pipeline.py and run_dasnet.py emit `argmax + 1` (1..29) while
run_frozen_dualpq.py and run_mgcnn_sdtransformer.py emit `y - 1` (0..28).
Mixing the two silently shifts every per-class attribution by one class.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 29
CLEAN = 999
LEVEL_ORDER = [CLEAN, 40, 30, 20, 10, 0]

METRICS = ("accuracy", "precision", "recall", "f1", "kappa")


def level_name(s) -> str:
    return "clean" if int(s) == CLEAN else f"{int(s)}dB"


def normalise_labels(y: np.ndarray) -> np.ndarray:
    """Coerce a label vector to 1..29, rejecting anything else."""
    y = np.asarray(y).astype(np.int64).ravel()
    lo, hi = int(y.min()), int(y.max())
    if lo == 0 and hi <= N_CLASSES - 1:
        y = y + 1
    elif lo >= 1 and hi <= N_CLASSES:
        pass
    else:
        raise ValueError(f"labels span [{lo}, {hi}]; expected 0..28 or 1..29")
    return y


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """29x29 confusion matrix, rows = true, cols = predicted, labels 1..29."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true - 1, y_pred - 1), 1)
    return cm


def _safe(num, den):
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros_like(den, dtype=np.float64)
    m = den > 0
    out[m] = np.asarray(num, dtype=np.float64)[m] / den[m]
    return out


def per_class_from_confusion(cm: np.ndarray) -> dict:
    """One-vs-rest metrics for all 29 classes, from a single confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = n - tp - fp - fn

    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    f1 = _safe(2 * precision * recall, precision + recall)
    accuracy = _safe(tp + tn, np.full(N_CLASSES, n))
    kappa_den = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    kappa = _safe(2 * (tp * tn - fn * fp), kappa_den)

    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "kappa": kappa,
            "support": (tp + fn).astype(np.int64)}


def overall_from_confusion(cm: np.ndarray) -> dict:
    """Aggregate metrics for a single confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    pc = per_class_from_confusion(cm)

    po = np.trace(cm) / n if n else 0.0
    pe = float((cm.sum(0) * cm.sum(1)).sum() / (n * n)) if n else 0.0
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

    return {"overall_accuracy": float(po),
            "overall_kappa": float(kappa),
            "macro_precision": float(pc["precision"].mean()),
            "macro_recall": float(pc["recall"].mean()),
            "macro_f1": float(pc["f1"].mean()),
            "macro_accuracy": float(pc["accuracy"].mean()),
            "macro_kappa": float(pc["kappa"].mean()),
            "n": int(n)}


def evaluate(y_true, y_pred, snr, levels=None) -> dict:
    """
    Full per-class x per-SNR evaluation of one prediction set.

    Returns
    -------
    {
      "levels":   [999, 40, 30, 20, 10, 0],
      "grids":    {metric: (29, n_levels) array},        per class per SNR
      "per_snr":  {level: {aggregate metrics}},          one column each
      "pooled":   {aggregate metrics over all levels},
      "pooled_per_class": {metric: (29,) array},
      "confusion": {level: (29,29) array, "pooled": (29,29)},
    }
    """
    y_true = normalise_labels(y_true)
    y_pred = normalise_labels(y_pred)
    snr = np.asarray(snr).astype(np.int64).ravel()
    if not (len(y_true) == len(y_pred) == len(snr)):
        raise ValueError("y_true, y_pred and snr must be the same length")

    present = set(snr.tolist())
    levels = [s for s in (levels or LEVEL_ORDER) if s in present]

    grids = {m: np.full((N_CLASSES, len(levels)), np.nan) for m in METRICS}
    support = np.zeros((N_CLASSES, len(levels)), dtype=np.int64)
    per_snr, cms = {}, {}

    for j, s in enumerate(levels):
        m = snr == s
        cm = confusion(y_true[m], y_pred[m])
        cms[int(s)] = cm
        pc = per_class_from_confusion(cm)
        for k in METRICS:
            grids[k][:, j] = pc[k]
        support[:, j] = pc["support"]
        per_snr[int(s)] = overall_from_confusion(cm)

    cm_all = confusion(y_true, y_pred)
    pc_all = per_class_from_confusion(cm_all)

    return {"levels": [int(s) for s in levels],
            "grids": grids,
            "support": support,
            "per_snr": per_snr,
            "pooled": overall_from_confusion(cm_all),
            "pooled_per_class": {k: pc_all[k] for k in METRICS},
            "confusion": {**cms, "pooled": cm_all}}


def aggregate_seeds(evals: list) -> dict:
    """Mean and sample SD (ddof=1) of every grid / scalar across seeds."""
    if not evals:
        raise ValueError("no evaluations to aggregate")
    levels = evals[0]["levels"]
    for e in evals:
        if e["levels"] != levels:
            raise ValueError("seeds disagree on which SNR levels are present")

    grids_mean, grids_sd = {}, {}
    for k in METRICS:
        stack = np.stack([e["grids"][k] for e in evals])
        grids_mean[k] = stack.mean(0)
        grids_sd[k] = stack.std(0, ddof=1) if len(evals) > 1 else np.zeros_like(stack[0])

    per_snr = {}
    for s in levels:
        per_snr[int(s)] = {}
        for key in evals[0]["per_snr"][s]:
            v = np.array([e["per_snr"][s][key] for e in evals], dtype=np.float64)
            per_snr[int(s)][key] = (float(v.mean()),
                                    float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    pooled = {}
    for key in evals[0]["pooled"]:
        v = np.array([e["pooled"][key] for e in evals], dtype=np.float64)
        pooled[key] = (float(v.mean()),
                       float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    pooled_pc = {}
    for k in METRICS:
        stack = np.stack([e["pooled_per_class"][k] for e in evals])
        pooled_pc[k] = (stack.mean(0),
                        stack.std(0, ddof=1) if len(evals) > 1 else np.zeros_like(stack[0]))

    return {"levels": levels, "n_seeds": len(evals),
            "grids_mean": grids_mean, "grids_sd": grids_sd,
            "per_snr": per_snr, "pooled": pooled,
            "pooled_per_class": pooled_pc,
            "support": evals[0]["support"]}
