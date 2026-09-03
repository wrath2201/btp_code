"""
eval_per_class_snr.py -- per-class x per-SNR accuracy / precision / recall /
F1 / kappa for every classifier, plus annotated heatmaps in the style of
figures/fig2_class_snr_heatmap.png.

Usage
-----
    python scripts/eval_per_class_snr.py \
        --model "Classical Ensemble=results/preds/classical_weighted_vote_seed*_preds.npz" \
        --model "DASNet=results/preds/dasnet_seed*_preds.npz" \
        --model "MGCNN-SDTransformer=results/preds/mgcnn_seed*_preds.npz"

Each --model is  LABEL=GLOB ; every file matching GLOB is treated as one seed
and results are reported as mean (+/- sample SD when more than one seed).

Writes, per model, into results/per_class_snr/<slug>/ :
    accuracy_heatmap.png  precision_heatmap.png  recall_heatmap.png
    f1_heatmap.png        kappa_heatmap.png
    metrics.json          per_class_per_snr.csv
and one combined results/per_class_snr/summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.metrics_perclass import (METRICS, N_CLASSES, aggregate_seeds,
                                  evaluate, level_name)
from src.pqmodel import CLASS_NAMES

OUT_ROOT = "results/per_class_snr"
PRETTY = {"accuracy": "one-vs-rest accuracy", "precision": "precision",
          "recall": "recall", "f1": "F1", "kappa": "Cohen's kappa"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def heatmap(grid, levels, title, cbar_label, path, sd=None, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(6.6, 8.6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(levels)), [level_name(s) for s in levels])
    ax.set_yticks(range(N_CLASSES),
                  [f"{i+1}. {CLASS_NAMES[i+1]}" for i in range(N_CLASSES)],
                  fontsize=7)
    for i in range(N_CLASSES):
        for j in range(len(levels)):
            v = grid[i, j]
            txt = "--" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.5, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_seeds(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no prediction files match: {pattern}")
    evals = []
    for f in files:
        d = np.load(f)
        evals.append(evaluate(d["yte"], d["yp"], d["ste"]))
    return files, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL=GLOB, repeatable")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    report, all_agg = [], {}

    for spec in a.model:
        label, _, pattern = spec.partition("=")
        files, evals = load_seeds(pattern)
        agg = aggregate_seeds(evals)
        all_agg[label] = agg
        d = os.path.join(a.out, slug(label))
        os.makedirs(d, exist_ok=True)
        ns = agg["n_seeds"]
        tag = f"mean of {ns} seeds" if ns > 1 else "seed 0"
        print(f"\n=== {label}  ({ns} seed(s): {[os.path.basename(f) for f in files]})")

        for m in METRICS:
            g = agg["grids_mean"][m]
            heatmap(g, agg["levels"],
                    f"Per-class {PRETTY[m]} by SNR ({label}, {tag})",
                    PRETTY[m], os.path.join(d, f"{m}_heatmap.png"),
                    sd=agg["grids_sd"][m])

        # ---- json ----
        payload = {
            "label": label, "n_seeds": ns, "files": files,
            "levels": agg["levels"],
            "class_names": {str(i + 1): CLASS_NAMES[i + 1] for i in range(N_CLASSES)},
            "per_class_per_snr": {
                m: {"mean": agg["grids_mean"][m].tolist(),
                    "sd": agg["grids_sd"][m].tolist()} for m in METRICS},
            "support_per_class_per_snr": agg["support"].tolist(),
            "aggregate_per_snr": agg["per_snr"],
            "aggregate_pooled": agg["pooled"],
            "pooled_per_class": {m: {"mean": agg["pooled_per_class"][m][0].tolist(),
                                     "sd": agg["pooled_per_class"][m][1].tolist()}
                                 for m in METRICS},
        }
        with open(os.path.join(d, "metrics.json"), "w") as fh:
            json.dump(payload, fh, indent=1)

        # ---- csv ----
        with open(os.path.join(d, "per_class_per_snr.csv"), "w") as fh:
            fh.write("class_id,class_name,metric," +
                     ",".join(level_name(s) for s in agg["levels"]) + "\n")
            for m in METRICS:
                for i in range(N_CLASSES):
                    fh.write(f"{i+1},\"{CLASS_NAMES[i+1]}\",{m}," +
                             ",".join(f"{v:.4f}" for v in agg["grids_mean"][m][i]) + "\n")

        # ---- console + markdown ----
        lines = [f"## {label}  ({tag})", ""]
        hdr = "| metric | " + " | ".join(level_name(s) for s in agg["levels"]) + " | pooled |"
        lines += [hdr, "|" + "---|" * (len(agg["levels"]) + 2)]
        rows = [("overall accuracy", "overall_accuracy"),
                ("macro precision", "macro_precision"),
                ("macro recall (= balanced acc)", "macro_recall"),
                ("macro F1", "macro_f1"),
                ("overall kappa", "overall_kappa"),
                ("macro one-vs-rest accuracy", "macro_accuracy")]
        for name, key in rows:
            cells = []
            for s in agg["levels"]:
                mu, sd = agg["per_snr"][s][key]
                cells.append(f"{mu*100:.2f}" + (f" ± {sd*100:.2f}" if ns > 1 else ""))
            mu, sd = agg["pooled"][key]
            cells.append(f"**{mu*100:.2f}**" + (f" ± {sd*100:.2f}" if ns > 1 else ""))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        report += lines
        print("\n".join(lines))

    # ---- cross-model macro-F1 / kappa comparison ----
    levels = next(iter(all_agg.values()))["levels"]
    for key, nm in (("macro_f1", "macro-F1"), ("overall_kappa", "Cohen's kappa"),
                    ("overall_accuracy", "overall accuracy")):
        report.append(f"## Cross-model {nm} by SNR")
        report.append("")
        report.append("| model | " + " | ".join(level_name(s) for s in levels) + " | pooled |")
        report.append("|" + "---|" * (len(levels) + 2))
        for label, agg in all_agg.items():
            cells = [f"{agg['per_snr'][s][key][0]*100:.2f}" for s in levels]
            cells.append(f"**{agg['pooled'][key][0]*100:.2f}**")
            report.append(f"| {label} | " + " | ".join(cells) + " |")
        report.append("")

        fig, ax = plt.subplots(figsize=(7, 4.4))
        xp = range(len(levels))
        for label, agg in all_agg.items():
            v = [agg["per_snr"][s][key][0] for s in levels]
            e = [agg["per_snr"][s][key][1] for s in levels]
            ax.errorbar(xp, v, yerr=e, marker="o", capsize=3, lw=1.6, ms=5,
                        label=label)
        ax.set_xticks(list(xp), [level_name(s) for s in levels])
        ax.set(xlabel="noise level", ylabel=nm, title=f"Test {nm} vs SNR")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"compare_{key}.png"), dpi=130)
        plt.close(fig)

    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("# Per-class x per-SNR evaluation\n\n" + "\n".join(report))
    print(f"\nwrote {a.out}/summary.md and per-model heatmaps")


if __name__ == "__main__":
    main()
