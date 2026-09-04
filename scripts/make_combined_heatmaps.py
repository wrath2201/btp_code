"""
make_combined_heatmaps.py  --  Generate combined 5-panel per-class x per-SNR
heatmaps (Recall, Precision, F1 score, Accuracy 1-vs-rest, Kappa) for every
model with per_class_per_snr.csv data.

Produces one wide figure per model, matching the reference layout:
    [Recall | Precision | F1 score | Accuracy (1-vs-rest) | Kappa]

Usage
-----
    python scripts/make_combined_heatmaps.py

Or for a specific model only:
    python scripts/make_combined_heatmaps.py --model frozen_dasnet_dualpq
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---- Configuration ----

# Metric columns in the CSV and their display names (in panel order)
METRICS = [
    ("recall",    "Recall"),
    ("precision", "Precision"),
    ("f1",        "F1 score"),
    ("accuracy",  "Accuracy (1-vs-rest)"),
    ("kappa",     "Kappa"),
]

SNR_COLS = ["clean", "40dB", "30dB", "20dB", "10dB", "0dB"]

N_CLASSES = 29

# All known per_class_snr directories
MODEL_DIRS = {
    "Classical Ensemble (weighted_vote)":
        "results/per_class_snr/classical_ensemble_weighted_vote",
    "Classical Ensemble (geometric_vote)":
        "results/per_class_snr/classical_ensemble_geometric_vote",
    "DASNet":
        "results/per_class_snr/dasnet",
    "MGCNN-SDTransformer":
        "results/per_class_snr/mgcnn_sdtransformer",
    "Frozen-DASNet DualPQ":
        "results/per_class_snr_frozen/frozen_dasnet_dualpq",
}


def load_csv(filepath):
    """Load per_class_per_snr.csv → dict[metric] → np.array(29, 6)"""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grids = {}
    for metric_key, _ in METRICS:
        grid = np.full((N_CLASSES, len(SNR_COLS)), np.nan)
        metric_rows = [r for r in rows if r["metric"] == metric_key]
        for row in metric_rows:
            cid = int(row["class_id"]) - 1  # 0-indexed
            for j, snr in enumerate(SNR_COLS):
                grid[cid, j] = float(row[snr])
        grids[metric_key] = grid

    # Extract class names in order
    class_names = {}
    for row in rows:
        cid = int(row["class_id"])
        if cid not in class_names:
            class_names[cid] = row["class_name"]
    labels = [f"{i}. {class_names[i]}" for i in range(1, N_CLASSES + 1)]

    return grids, labels


def make_combined_heatmap(grids, class_labels, model_name, outpath):
    """Create a single wide figure with 5 side-by-side heatmaps."""

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(n_metrics * 4.0, 10.5),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    # RdYlGn colourmap, 0 → 1
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for idx, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[idx]
        grid = grids[metric_key]

        im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")

        # X-axis (SNR levels)
        ax.set_xticks(range(len(SNR_COLS)))
        ax.set_xticklabels(SNR_COLS, fontsize=7, rotation=45, ha="right")

        # Y-axis (class names — only on the leftmost panel)
        if idx == 0:
            ax.set_yticks(range(N_CLASSES))
            ax.set_yticklabels(class_labels, fontsize=6.5)
        else:
            ax.set_yticks(range(N_CLASSES))

        # Annotate each cell with its value
        for i in range(N_CLASSES):
            for j in range(len(SNR_COLS)):
                v = grid[i, j]
                if np.isfinite(v):
                    txt = f"{v:.2f}"
                    # Use white text on very dark cells, black otherwise
                    text_color = "white" if v < 0.35 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=5, color=text_color, fontweight="normal")
                else:
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=5, color="gray")

        ax.set_title(metric_title, fontsize=10, fontweight="bold", pad=8)

    # Shared colourbar on the right
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.tolist(), shrink=0.55, pad=0.02, label="score",
    )
    cbar.ax.tick_params(labelsize=8)

    # Super-title
    # Keep the figure title factual: commentary belongs in the caption, not
    # burned into a PNG that may go into a submission.
    fig.suptitle(
        f"Per-class metrics by noise level ({model_name})",
        fontsize=11, fontweight="bold", y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Slug of a single model dir to process (e.g. frozen_dasnet_dualpq)")
    ap.add_argument("--outdir", default="results/per_class_snr",
                    help="Output directory for the combined heatmap PNGs")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    for model_name, model_dir in MODEL_DIRS.items():
        csv_path = os.path.join(model_dir, "per_class_per_snr.csv")

        # If --model is set, only process the matching model
        if a.model and a.model not in model_dir:
            continue

        if not os.path.exists(csv_path):
            print(f"  skipping {model_name}: {csv_path} not found")
            continue

        print(f"\n=== {model_name} ===")
        grids, class_labels = load_csv(csv_path)

        # Determine output filename from model dir slug
        slug = os.path.basename(model_dir)
        outpath = os.path.join(a.outdir, f"combined_heatmap_{slug}.png")

        make_combined_heatmap(grids, class_labels, model_name, outpath)


if __name__ == "__main__":
    main()
