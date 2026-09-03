# Final Results Manifest

This document is the authoritative manifest for the final results of the study. 
Historical/debug artifacts are excluded from this document and are strictly retained for provenance only.

## 1. Final Benchmark Results

| Model | Mean Macro-F1 | SD (ddof=1) |
|---|---:|---:|
| Classical Ensemble | 71.52 | 0.84 |
| DASNet | 69.72 | 9.11 |
| MGCNN-SDTransformer | 66.59 | 0.98 |
| Original DualPQ-D | 61.63 | 15.58 |
| **Frozen-DASNet DualPQ** | **74.46** | **1.08** |

### Individual Seeds (Macro-F1 %)

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| **Frozen-DASNet DualPQ** | 72.55 | 74.77 | 75.23 | 74.77 | 74.97 |
| **Original DualPQ-D** | 72.67 | 72.56 | 62.43 | 34.91 | 65.61 |
| **DASNet** | 53.45 | 73.84 | 73.89 | 73.10 | 74.34 |
| **Classical Ensemble** | 70.46 | 72.36 | 70.85 | 71.70 | 72.23 |
| **MGCNN-SDTransformer** | 65.01 | 67.34 | 67.37 | 66.31 | 66.91 |

## 2. Per-SNR Performance

| Condition | Classical Ensemble | DASNet | MGCNN-SDTransformer | Original DualPQ-D | **Frozen-DASNet DualPQ** |
|---|---:|---:|---:|---:|---:|
| Clean | 93.30 | 85.59 | 73.18 | 77.29 | **91.18** |
| 40 dB | 93.34 | 85.83 | 73.65 | 77.56 | **91.69** |
| 30 dB | 91.56 | 84.75 | 72.93 | 75.38 | **89.56** |
| 20 dB | 82.52 | 77.62 | 70.36 | 68.32 | **82.36** |
| 10 dB | 52.88 | 58.74 | 60.18 | 49.20 | **62.36** |
| 0 dB  | 15.55 | 25.80 | 49.26 | 22.04 | **27.00** |

## 3. Dataset & Protocol Protocol
- **Size:** 34,800 total samples (1280-length, 29 classes).
- **Conditions:** Clean, 40, 30, 20, 10, 0 dB.
- **Split:** Grouped stratified 70/15/15. All noise variants of a given base waveform remain tightly within a single partition.
- **Seeds:** 5 deep/hybrid training seeds (0, 1, 2, 3, 4) evaluating training run variability on the exact same dataset partition.

## 4. Final Artifact Paths

All final prediction artifacts are stored in `results/multiseed/`.

| Model | Script | Artifact Pattern | Prediction Arrays |
|---|---|---|---|
| **Frozen-DASNet DualPQ** | `scripts/run_frozen_dualpq.py` | `frozen_dualpq_seed[0-4].json` | `frozen_dualpq_seed[0-4]_preds.npz` |
| **Original DualPQ-D** | `scripts/run_dualpq.py` | `dualpq_concat_seed[0-4].json` | `dualpq_concat_seed[0-4]_preds.npz` |
| **DASNet** | `scripts/run_dasnet.py` | `dasnet_seed[0-4].json` | `dasnet_seed[0-4]_preds.npz` |
| **MGCNN-SDTransformer** | `scripts/run_mgcnn.py` | `mgcnn_seed[0-4].json` | `mgcnn_seed[0-4]_preds.npz` |

*(Note: Classical Ensemble results were derived using cross-validation over the training features in `experiments/multiseed.py`).*
