# Results Directory Map

This directory contains the final authoritative results, baseline results, and historical/diagnostic artifacts from the 29-Class Power Quality Disturbance classification study.

To prevent breaking references in scripts and documentation, files have been kept in this flat structure. Please use this guide to navigate the files.

## 1. Authoritative Final Results
These are the files that support the final reported metrics (e.g., Frozen-DASNet DualPQ at 74.46%):
- `dualpq_concat_preds.npz` & `dualpq_concat.json`: The **Frozen-DASNet DualPQ** final proposed method predictions and metrics.
- `results_preds.npz` & `results.json`: The **Classical Ensemble** final baseline predictions and metrics.
- `dasnet_results_preds.npz` & `dasnet_results.json`: The **DASNet (Deep-only)** baseline predictions and metrics.

## 2. Baseline & Diagnostic Results
- `mgcnn_sdtransformer_seed*_preds.npz` & `*.json`: The reimplemented MGCNN-SDTransformer baseline from Jiang et al., evaluated across 5 seeds under our protocol.
- `dasnet_ablation_nodst*`: Diagnostics removing the learnable DST from DASNet.
- `unseen_snr.json`: Leave-one-SNR-out diagnostic tests.
- `leakage_audit.json`: The statistical proof of why row-based splitting inflates scores.

## 3. Historical / Superseded Experiments
- `dualpq_feature_learned*`, `dualpq_snr_hard*`, `dualpq_snr_learned*`: Original DualPQ-D architectures and early end-to-end fusion diagnostics that exhibited severe seed instability.
- `pilot_results.json`, `dasnet_results_pilot.json`: Smoke tests and pilot runs used for scaling tests.
- `results_clean.json`: Classical baseline run strictly on the clean (noise-free) shard, used primarily for establishing theoretical upper bounds.

## 4. Multi-Seed Folders
- `multiseed/`: Contains the independent 5-seed runs of DASNet used for computing stability/variance.
- `per_class/`: Contains class-by-class metric breakdowns.

## 5. Figures
- `figures/`: Contains the plots generated from these result files via `scripts/make_figures.py`.
