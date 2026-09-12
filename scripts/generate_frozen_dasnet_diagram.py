"""
generate_frozen_dasnet_diagram.py
Research-paper-quality architecture diagram of the Frozen-DASNet DualPQ model.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── colour palette
FROZEN_FACE  = "#1a3a5c"
FROZEN_EDGE  = "#4fc3f7"
TRAIN_FACE   = "#1b4332"
TRAIN_EDGE   = "#52b788"
GATE_FACE    = "#3d1a78"
GATE_EDGE    = "#b47fff"
INPUT_FACE   = "#1a1a2e"
INPUT_EDGE   = "#e2b96f"
OUTPUT_FACE  = "#3b0000"
OUTPUT_EDGE  = "#ff6b6b"
SNR_FACE     = "#1a3320"
SNR_EDGE     = "#80ffdb"
FUSION_FACE  = "#2a1a3a"
FUSION_EDGE  = "#ff9fff"
ARROW_COLOR  = "#c8c8d8"
TEXT_MAIN    = "#f0f4ff"
TEXT_SUB     = "#a0b0cc"
TEXT_FROZEN  = "#4fc3f7"
TEXT_TRAIN   = "#52b788"
BG_COLOR     = "#0d0d1a"

def rounded_box(ax, x, y, w, h, fc, ec, lw=1.5, alpha=0.92, radius=0.025):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    return box

def label(ax, x, y, main, sub=None, main_size=8.5, sub_size=7,
          main_color=TEXT_MAIN, sub_color=TEXT_SUB):
    ax.text(x, y + (0.012 if sub else 0), main,
            ha="center", va="center", fontsize=main_size,
            color=main_color, fontweight="bold", zorder=5)
    if sub:
        ax.text(x, y - 0.025, sub, ha="center", va="center",
                fontsize=sub_size, color=sub_color, zorder=5, fontstyle="italic")

def arrow(ax, x0, y0, x1, y1, color=ARROW_COLOR, lw=1.2, mutation=6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=mutation), zorder=4)

def dashed_box(ax, x, y, w, h, color, lw=1.2, label_str="", label_color=None):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, fill=False, edgecolor=color,
                          linewidth=lw, linestyle="--", zorder=2, alpha=0.6)
    ax.add_patch(rect)
    if label_str:
        ax.text(x - w/2 + 0.01, y + h/2 - 0.01, label_str,
                fontsize=7, color=label_color or color, va="top", zorder=5)

fig, ax = plt.subplots(figsize=(18, 10.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")

ax.text(0.5, 0.975, "Frozen-DASNet DualPQ Architecture",
        ha="center", va="top", fontsize=16, color=TEXT_MAIN, fontweight="bold")
ax.text(0.5, 0.952,
        "Stage-1 pre-trained Deep Expert (frozen) + Trainable Classical Expert + SNR-Conditioned Learned Gate",
        ha="center", va="top", fontsize=9.5, color=TEXT_SUB)

patches_legend = [
    mpatches.Patch(facecolor=FROZEN_FACE, edgecolor=FROZEN_EDGE, linewidth=1.5, label="Frozen (requires_grad=False)"),
    mpatches.Patch(facecolor=TRAIN_FACE,  edgecolor=TRAIN_EDGE,  linewidth=1.5, label="Trainable"),
    mpatches.Patch(facecolor=GATE_FACE,   edgecolor=GATE_EDGE,   linewidth=1.5, label="Routing Gate"),
    mpatches.Patch(facecolor=FUSION_FACE, edgecolor=FUSION_EDGE, linewidth=1.5, label="Fusion"),
    mpatches.Patch(facecolor=SNR_FACE,    edgecolor=SNR_EDGE,    linewidth=1.5, label="SNR Estimator"),
]
ax.legend(handles=patches_legend, loc="lower center", bbox_to_anchor=(0.5, 0.0),
          ncol=5, framealpha=0.15, fontsize=8, facecolor="#111128", edgecolor="#444466",
          labelcolor=TEXT_MAIN)

DY_TOP = 0.76
DY_MID = 0.50
DY_BOT = 0.28
BW = 0.080
BH = 0.088

# Dashed region labels
dashed_box(ax, 0.475, DY_TOP, 0.53, 0.20, FROZEN_EDGE, lw=1.4,
           label_str="Frozen Deep Expert (DASNet backbone)", label_color=FROZEN_EDGE)
dashed_box(ax, 0.475, DY_BOT, 0.53, 0.18, TRAIN_EDGE, lw=1.4,
           label_str="Trainable Classical Expert", label_color=TRAIN_EDGE)
dashed_box(ax, 0.885, DY_MID, 0.23, 0.60, "#ffb347", lw=1.4,
           label_str="Stage-2 (trainable)", label_color="#ffb347")

# Input waveform and features
x_in = 0.055
rounded_box(ax, x_in, DY_TOP, BW - 0.01, BH, INPUT_FACE, INPUT_EDGE, lw=2.0)
label(ax, x_in, DY_TOP, "Waveform", r"$(B,\,1280)$")
rounded_box(ax, x_in, DY_BOT, BW - 0.01, BH, INPUT_FACE, INPUT_EDGE, lw=2.0)
label(ax, x_in, DY_BOT, "Features", r"$x_{feat}\in\mathbb{R}^{191}$")

# SNR Estimator (shared)
x_snr = 0.165
rounded_box(ax, x_snr, DY_MID, BW - 0.005, BH + 0.01, SNR_FACE, SNR_EDGE, lw=1.8)
label(ax, x_snr, DY_MID, "SNR Est.", r"$\hat{s}_{dB}$", main_color=SNR_EDGE)
ax.text(x_snr, DY_MID - 0.046, "Hann FFT\nHarmonic", ha="center", va="center",
        fontsize=5.8, color=TEXT_SUB, zorder=5)
arrow(ax, x_in + BW/2 - 0.005, DY_TOP, x_snr - BW/2 + 0.005, DY_MID + 0.01, color=SNR_EDGE, lw=1.0)
arrow(ax, x_in + BW/2 - 0.005, DY_BOT, x_snr - BW/2 + 0.005, DY_MID - 0.01, color=SNR_EDGE, lw=1.0)

# Cond MLP
x_cmlp = 0.270
rounded_box(ax, x_cmlp, DY_MID, BW - 0.008, BH, FROZEN_FACE, FROZEN_EDGE, lw=1.8)
label(ax, x_cmlp, DY_MID, "Cond MLP", r"$\mathbf{c}\in\mathbb{R}^{64}$", main_color=TEXT_FROZEN, sub_size=6.5)
ax.text(x_cmlp, DY_MID - 0.046, "Linear(1,64)->SiLU\nLinear(64,64)->SiLU",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_snr + BW/2 - 0.005, DY_MID, x_cmlp - BW/2 + 0.005, DY_MID, color=FROZEN_EDGE, lw=1.1)

# DST front-end
x_dst = 0.270
rounded_box(ax, x_dst, DY_TOP, BW - 0.005, BH + 0.01, FROZEN_FACE, FROZEN_EDGE, lw=2.0)
label(ax, x_dst, DY_TOP, "DST", r"$(B,\,320,\,1280)$", main_color=TEXT_FROZEN)
ax.text(x_dst, DY_TOP - 0.046,
        r"$\sigma_t(f)=c/f^p\cdot e^{\delta_f}$" + "\nLearnable",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_in + BW/2 - 0.005, DY_TOP, x_dst - BW/2 + 0.005, DY_TOP, color=FROZEN_EDGE, lw=1.2)

# Log Compression + Pooling
x_log = 0.372
rounded_box(ax, x_log, DY_TOP, BW - 0.003, BH + 0.01, FROZEN_FACE, FROZEN_EDGE, lw=1.8)
label(ax, x_log, DY_TOP, "Log+Pool", r"$(B,\,3,\,320,\,160)$", main_color=TEXT_FROZEN)
ax.text(x_log, DY_TOP - 0.048,
        r"$\log(1+20|S|)$" + "\nAvg+Max pool x8\n+freq coord ch.",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_dst + BW/2 - 0.005, DY_TOP, x_log - BW/2 + 0.003, DY_TOP, color=FROZEN_EDGE, lw=1.2)

# 4 x FiLM ResNet Stages
stage_xs   = [0.474, 0.558, 0.641, 0.725]
stage_dims = ["(B,32,160,80)", "(B,64,80,40)", "(B,128,40,20)", "(B,256,20,10)"]
stage_ch   = ["32ch", "64ch", "128ch", "256ch"]
prev_x = x_log
for i, (sx, sdim, sch) in enumerate(zip(stage_xs, stage_dims, stage_ch)):
    rounded_box(ax, sx, DY_TOP, BW - 0.002, BH + 0.015, FROZEN_FACE, FROZEN_EDGE, lw=1.8)
    label(ax, sx, DY_TOP, f"FiLM Stage {i+1}", f"({sch})", main_color=TEXT_FROZEN, main_size=8)
    ax.text(sx, DY_TOP - 0.052, "Conv3x3->GN->SiLU\nFiLM(c)->Skip down2",
            ha="center", va="center", fontsize=5.3, color=TEXT_SUB, zorder=5)
    ax.text(sx, DY_TOP - 0.073, sdim, ha="center", va="center", fontsize=5.0, color="#607d8b", zorder=5)
    arrow(ax, prev_x + BW/2 - 0.003, DY_TOP, sx - BW/2 + 0.002, DY_TOP, color=FROZEN_EDGE, lw=1.1)
    # FiLM conditioning dashed arrows
    ax.annotate("", xy=(sx, DY_TOP - BH/2 - 0.015), xytext=(sx, DY_MID + 0.052),
                arrowprops=dict(arrowstyle="-|>", color=FROZEN_EDGE, lw=0.8,
                                mutation_scale=5, linestyle="dashed"), zorder=4)
    ax.text(sx - 0.012, (DY_TOP + DY_MID)/2 + 0.01, r"$\mathbf{c}$",
            fontsize=6.5, color=FROZEN_EDGE, zorder=5)
    prev_x = sx

# GAP
x_gap = 0.808
rounded_box(ax, x_gap, DY_TOP, BW - 0.015, BH, FROZEN_FACE, FROZEN_EDGE, lw=1.8)
label(ax, x_gap, DY_TOP, "GAP", r"$(B,\,256)$", main_color=TEXT_FROZEN)
ax.text(x_gap, DY_TOP - 0.045, "Global Avg Pool\nDropout(0.15)",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, stage_xs[-1] + BW/2 - 0.003, DY_TOP, x_gap - BW/2 + 0.007, DY_TOP, color=FROZEN_EDGE, lw=1.2)

# Classical Expert MLP
x_std = 0.270
rounded_box(ax, x_std, DY_BOT, BW - 0.005, BH, TRAIN_FACE, TRAIN_EDGE, lw=1.8)
label(ax, x_std, DY_BOT, "Std Scale", r"$(B,\,191)$", main_color=TEXT_TRAIN, sub_size=6.5)
ax.text(x_std, DY_BOT - 0.044, "StandardScaler\n(fit on train)",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_in + BW/2 - 0.005, DY_BOT, x_std - BW/2 + 0.005, DY_BOT, color=TRAIN_EDGE, lw=1.2)

x_mlp1 = 0.372
rounded_box(ax, x_mlp1, DY_BOT, BW - 0.003, BH, TRAIN_FACE, TRAIN_EDGE, lw=1.8)
label(ax, x_mlp1, DY_BOT, "MLP H1", r"$(B,\,512)$", main_color=TEXT_TRAIN)
ax.text(x_mlp1, DY_BOT - 0.044, "Linear(191,512)\nBN->SiLU->Drop",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_std + BW/2 - 0.005, DY_BOT, x_mlp1 - BW/2 + 0.003, DY_BOT, color=TRAIN_EDGE, lw=1.2)

x_mlp2 = 0.474
rounded_box(ax, x_mlp2, DY_BOT, BW - 0.003, BH, TRAIN_FACE, TRAIN_EDGE, lw=1.8)
label(ax, x_mlp2, DY_BOT, "MLP H2", r"$(B,\,256)$", main_color=TEXT_TRAIN)
ax.text(x_mlp2, DY_BOT - 0.044, "Linear(512,256)\nBN->SiLU->Drop",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_mlp1 + BW/2 - 0.003, DY_BOT, x_mlp2 - BW/2 + 0.003, DY_BOT, color=TRAIN_EDGE, lw=1.2)

x_cls_emb = x_gap
rounded_box(ax, x_cls_emb, DY_BOT, BW - 0.015, BH, TRAIN_FACE, TRAIN_EDGE, lw=1.8)
label(ax, x_cls_emb, DY_BOT, "Embedding", r"$\mathbf{z}_{cls}\in\mathbb{R}^{256}$",
      main_color=TEXT_TRAIN, sub_size=6.5)
arrow(ax, x_mlp2 + BW/2 - 0.003, DY_BOT, x_cls_emb - BW/2 + 0.007, DY_BOT, color=TRAIN_EDGE, lw=1.2)

ax.text(0.61, DY_BOT + BH/2 + 0.01, r"$\mathbf{z}_{cls}$", ha="center", fontsize=7, color=TEXT_TRAIN, zorder=5)
ax.text(0.77, DY_TOP + BH/2 + 0.010, r"$\mathbf{z}_{deep}\in\mathbb{R}^{256}$",
        ha="center", fontsize=7, color=TEXT_FROZEN, zorder=5)

# SNR Gate
x_gate = 0.872
rounded_box(ax, x_gate, DY_MID, BW - 0.01, BH + 0.01, GATE_FACE, GATE_EDGE, lw=2.0)
label(ax, x_gate, DY_MID, "SNR Gate", r"$g\in(0,1)$", main_color=GATE_EDGE)
ax.text(x_gate, DY_MID - 0.048, "Linear(1,64)->SiLU\nLinear(64,1)->Sigmoid",
        ha="center", va="center", fontsize=5.5, color=TEXT_SUB, zorder=5)
arrow(ax, x_snr + BW/2 - 0.005, DY_MID, x_gate - BW/2 + 0.005, DY_MID, color=GATE_EDGE, lw=1.1)
ax.text((x_snr + x_gate)/2, DY_MID + 0.018, r"$\hat{s}/40$",
        ha="center", fontsize=6.5, color=GATE_EDGE, zorder=5)

# Fusion
x_fus = 0.920
rounded_box(ax, x_fus, DY_MID + 0.08, BW - 0.01, BH + 0.025, FUSION_FACE, FUSION_EDGE, lw=2.0)
label(ax, x_fus, DY_MID + 0.08, "Fusion", None, main_color=FUSION_EDGE)
ax.text(x_fus, DY_MID + 0.055,
        r"$\mathbf{z}=g\cdot\mathbf{z}_{deep}+(1-g)\cdot\mathbf{z}_{cls}$",
        ha="center", va="center", fontsize=6.0, color=TEXT_SUB, zorder=5)
ax.text(x_fus, DY_MID + 0.030, r"$\mathbf{z}\in\mathbb{R}^{256}$",
        ha="center", va="center", fontsize=6.0, color=FUSION_EDGE, zorder=5)

arrow(ax, x_gap + BW/2 - 0.007, DY_TOP, x_fus - BW/2 + 0.005, DY_MID + 0.11, color=FROZEN_EDGE, lw=1.2)
arrow(ax, x_cls_emb + BW/2 - 0.007, DY_BOT, x_fus - BW/2 + 0.005, DY_MID + 0.05, color=TRAIN_EDGE, lw=1.2)
arrow(ax, x_gate, DY_MID + BH/2 + 0.005, x_fus - BW/2 + 0.005, DY_MID + 0.07, color=GATE_EDGE, lw=1.1)
ax.text(x_gate + 0.015, DY_MID + 0.090, r"$g$", ha="left", fontsize=7, color=GATE_EDGE, zorder=5)

# Classifier
x_clf = 0.920
rounded_box(ax, x_clf, DY_MID - 0.075, BW - 0.01, BH, TRAIN_FACE, TRAIN_EDGE, lw=2.0)
label(ax, x_clf, DY_MID - 0.075, "Classifier", "Linear(256, K)", main_color=TEXT_TRAIN)
arrow(ax, x_fus, DY_MID + 0.08 - BH/2 - 0.013, x_clf, DY_MID - 0.075 + BH/2 + 0.005, color=FUSION_EDGE, lw=1.2)

# Softmax / Output
x_out = 0.920
rounded_box(ax, x_out, DY_MID - 0.21, BW - 0.008, BH, OUTPUT_FACE, OUTPUT_EDGE, lw=2.0)
label(ax, x_out, DY_MID - 0.21, "Softmax", r"$K=29$ classes", main_color=OUTPUT_EDGE)
arrow(ax, x_clf, DY_MID - 0.075 - BH/2 - 0.005, x_out, DY_MID - 0.21 + BH/2 + 0.005, color=TRAIN_EDGE, lw=1.2)

# Sidebar parameter counts
sidebar_x = 0.018
sidebar_lines = [
    ("Params", "#ffffff", 8.5, True),
    ("Total: 1.54M", TEXT_MAIN, 7.5, False),
    ("Frozen: 1.29M", TEXT_FROZEN, 7.5, False),
    ("Train: 0.25M", TEXT_TRAIN, 7.5, False),
    ("", "", 6, False),
    ("Training", "#ffffff", 8.5, True),
    ("Stage 1: DASNet", TEXT_FROZEN, 7, False),
    ("  (pre-trained)", TEXT_SUB, 6.5, False),
    ("Stage 2: MLP+", TEXT_TRAIN, 7, False),
    ("  Gate+Fusion", TEXT_TRAIN, 6.5, False),
    ("", "", 6, False),
    ("Input", "#ffffff", 8.5, True),
    ("Wave: 1280 pts", TEXT_MAIN, 7, False),
    ("Feats: 191", TEXT_MAIN, 7, False),
    ("fs=6400 Hz", TEXT_MAIN, 7, False),
    ("f0=50 Hz", TEXT_MAIN, 7, False),
    ("f_max=1600 Hz", TEXT_MAIN, 7, False),
]
y_sidebar = 0.91
for txt, col, sz, bold in sidebar_lines:
    if txt:
        ax.text(sidebar_x, y_sidebar, txt, ha="left", va="center", fontsize=sz,
                color=col, fontweight="bold" if bold else "normal", zorder=5)
    y_sidebar -= 0.038 if bold else 0.031

# DST window law annotation
ax.text(0.270, DY_TOP + BH/2 + 0.015,
        r"$\sigma_t(f_n) = \mathrm{softplus}(c)/f_n^p \cdot e^{\delta_n}$",
        ha="center", va="bottom", fontsize=6.5, color=FROZEN_EDGE, zorder=5)

# Lock icons
for xi, yi in [(x_gap, DY_TOP), (x_dst, DY_TOP), (x_log, DY_TOP),
               (x_cmlp, DY_MID)] + [(sx, DY_TOP) for sx in stage_xs]:
    ax.text(xi + BW/2 - 0.025, yi + BH/2 + 0.002, "[FROZEN]",
            fontsize=4.5, va="bottom", ha="center", zorder=6, color=FROZEN_EDGE, fontstyle="italic")

# Path labels
ax.text(0.5, DY_TOP + 0.105, "-- Deep Expert Path --",
        ha="center", va="center", fontsize=8, color=FROZEN_EDGE, fontweight="bold", zorder=5)
ax.text(0.5, DY_BOT - 0.075, "-- Classical Expert Path --",
        ha="center", va="center", fontsize=8, color=TRAIN_EDGE, fontweight="bold", zorder=5)

# CE Loss label
ax.text(x_out, DY_MID - 0.21 - BH/2 - 0.025, "CE Loss (Stage-2 only)",
        ha="center", va="top", fontsize=6.5, color="#ffb347", zorder=5)

out_path = "figures/frozen_dasnet_architecture.pdf"
os.makedirs("figures", exist_ok=True)
plt.savefig(out_path, bbox_inches="tight", dpi=300, facecolor=BG_COLOR)
plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300, facecolor=BG_COLOR)
print(f"Saved -> {out_path}")
print(f"Saved -> {out_path.replace('.pdf', '.png')}")
plt.close(fig)
