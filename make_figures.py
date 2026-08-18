"""Figures for the results report."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pqmodel import CLASS_NAMES
from pipeline import level_name

_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default="results.json")
_ap.add_argument("--prefix", default="fig", help="output filename prefix")
_A = _ap.parse_args()
PFX = _A.prefix
R = json.load(open(_A.results))
SNRS = R.get("levels", [40, 30, 20, 10, 0])
XP = list(range(len(SNRS)))   # even spacing: "clean" has no dB value
MODELS = ["rf", "lgbm", "svm", "mlp"]
ENS = ["soft_vote", "weighted_vote", "geometric_vote", "stacked"]
sel = R["selected_ensemble"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m in MODELS:
    v = [R[f"base_{m}"]["test_per_snr"][str(s)]["macro_f1"] for s in SNRS]
    ax[0].plot(XP, v, "o--", lw=1, ms=4, alpha=.7, label=m)
for e in ENS:
    v = [R[e]["test_per_snr"][str(s)]["macro_f1"] for s in SNRS]
    ax[0].plot(XP, v, "s-", lw=2 if e == sel else 1.2, ms=5,
               label=e + (" (selected)" if e == sel else ""))
ax[0].set(xlabel="noise level", ylabel="macro F1", title="Test macro-F1 vs SNR")
ax[0].set_xticks(XP, [level_name(s) for s in SNRS])
ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)

for e in ENS:
    v = [R[e]["test_per_snr"][str(s)]["balanced_acc"] for s in SNRS]
    ax[1].plot(XP, v, "s-", lw=2 if e == sel else 1.2, ms=5, label=e)
ax[1].set(xlabel="noise level", ylabel="balanced accuracy",
          title="Test balanced accuracy vs SNR")
ax[1].set_xticks(XP, [level_name(s) for s in SNRS])
ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{PFX}1_snr_degradation.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2
heat = np.array(R["per_class_per_snr_recall"])
fig, ax = plt.subplots(figsize=(6.2, 8.5))
im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(SNRS)), [level_name(s) for s in SNRS])
ax.set_yticks(range(29), [f"{i+1}. {CLASS_NAMES[i+1]}" for i in range(29)],
              fontsize=7)
for i in range(29):
    for j in range(len(SNRS)):
        ax.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", fontsize=6,
                color="black")
ax.set_title(f"Per-class recall by SNR ({sel})", fontsize=10)
fig.colorbar(im, ax=ax, shrink=.5, label="recall")
fig.tight_layout(); fig.savefig(f"{PFX}2_class_snr_heatmap.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for k, s in enumerate((SNRS[0], SNRS[-1])):
    cm = np.array(R[f"confusion_snr{s}"], float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax[k].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax[k].set(title=f"Confusion at {level_name(s)} (row-normalised)",
              xlabel="predicted class", ylabel="true class")
    ax[k].set_xticks(range(0, 29, 2), range(1, 30, 2), fontsize=7)
    ax[k].set_yticks(range(0, 29, 2), range(1, 30, 2), fontsize=7)
    fig.colorbar(im, ax=ax[k], shrink=.7)
fig.tight_layout(); fig.savefig(f"{PFX}3_confusion.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, ax = plt.subplots(figsize=(7, 5.5))
top = R["top_features"][:22][::-1]
ax.barh([t[0] for t in top], [t[1] for t in top], color="#4a7ba7")
ax.set(xlabel="Random-Forest importance", title="Top 22 features")
ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(f"{PFX}4_feature_importance.png"); plt.close(fig)

print(f"wrote {PFX}1_snr_degradation.png {PFX}2_class_snr_heatmap.png "
      f"{PFX}3_confusion.png {PFX}4_feature_importance.png")
