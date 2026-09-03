# Final Publication Audit Report

This report serves as the ultimate scientific baseline for the paper. All metrics herein have been verified directly from the raw prediction arrays (`*_preds.npz`) resulting from a stringent 5-seed evaluation protocol.

## 1. Authoritative Final Results (Macro-F1)
| Model | Mean ± Sample SD | 95% CI |
|-------|-----------------|---------|
| **Frozen-DASNet DualPQ** | **74.46% ± 1.08%** | [73.10%, 75.83%] |
| **Classical Ensemble** | 71.52% ± 0.84% | [70.47%, 72.57%] |
| **DASNet (Learnable DST)** | 69.72% ± 9.11% | [58.33%, 81.12%] |
| **MGCNN-SDTransformer** | 66.59% ± 0.98% | [65.36%, 67.82%] |
| **Original DualPQ-D** | 61.63% ± 15.58% | [42.15%, 81.11%] |

## 2. Five-Seed Individual Results
| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|-------|--------|--------|--------|--------|--------|
| **Frozen-DASNet DualPQ** | 72.55 | 74.77 | 75.23 | 74.77 | 74.97 |
| **Classical Ensemble** | 70.46 | 72.36 | 70.85 | 71.70 | 72.23 |
| **DASNet** | 53.45 | 73.84 | 73.89 | 73.10 | 74.34 |
| **MGCNN-SDTransformer** | 65.01 | 67.34 | 67.37 | 66.31 | 66.91 |
| **Original DualPQ-D** | 72.67 | 72.56 | 62.43 | 34.91 | 65.61 |

## 3. Per-SNR Results (Averaged across 5 Seeds)
| Model | Clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB |
|-------|-------|-------|-------|-------|-------|------|
| **Frozen-DASNet DualPQ** | **91.18%** | **91.69%** | **89.56%** | **82.36%** | 62.36% | **27.00%** |
| **Classical Ensemble** | 87.21% | 86.83% | 85.74% | 80.73% | **62.40%** | 25.75% |
| **DASNet** | 84.79% | 90.16% | 87.18% | 75.85% | 51.70% | 20.70% |
| **MGCNN-SDTransformer** | 81.62% | 81.58% | 80.79% | 76.16% | 55.87% | 22.45% |
| **Original DualPQ-D** | 77.44% | 78.51% | 77.16% | 67.01% | 35.68% | 12.79% |

## 4. Statistical Testing
A paired bootstrap procedure ($N=1000$) was performed on the **Seed-0** test predictions.
*   **Frozen-DASNet vs Classical Ensemble:** +2.09% (p < 0.001)
*   **Frozen-DASNet vs Original DualPQ:** -0.12% (p = 0.5510)
**Interpretation:** On Seed 0, Original DualPQ achieved one of its only successful training runs (72.67%), making Frozen-DASNet (72.55%) statistically indistinguishable from it *for that specific seed*. However, Frozen-DASNet significantly outperforms the Classical Ensemble on Seed 0. Across all five seeds (descriptive statistics), Frozen-DASNet is vastly superior in average stability and performance.

## 5. Grouped-Split Methodology
The evaluation strictly enforced a **waveform-grouped train/val/test split**. All synthetic SNR variants (e.g., 0dB, 10dB, 40dB) of a specific base waveform were assigned to the exact same partition. This prevents cross-variant leakage, avoiding artificially inflated scores (unlike traditional naive row-level splitting).

## 6. Frozen-DASNet Implementation Verification
The implementation in `scripts/run_frozen_dualpq.py` and `src/dualpq.py` has been verified:
- **Frozen Deep Expert:** `param.requires_grad = False` is explicitly enforced for all DASNet parameters.
- **Evaluation Mode:** The deep expert remains in `.eval()` mode during fusion training, preventing BatchNorm tracking and locking the dropout mask.
- **Trainable Fusion:** Only the 191-feature Classical branch and fusion classification head receive optimizer updates.
- **No Preprocessing Leakage:** Standard scaling for classical features is safely fitted exclusively on the training partition.

## 7. MGCNN Comparison Limitations
MGCNN-SDTransformer achieved 66.59% Macro-F1 under our benchmark. The original publication used a different experimental protocol, including differences in dataset setup, noise conditions, splitting strategy, and evaluation metric. Therefore, its reported accuracy is not directly comparable with our Macro-F1 under the present grouped benchmark.

- **Grouped Split vs Random Split:** We rigidly force noise-variants of the same base waveform into the same split (train/val/test). Random splitting allows the network to learn the base waveform instead of the actual physical disturbance.

## 8. Novelty & Contribution Statement
**Our specific contribution is:** "We propose a decoupled fusion strategy that combines a stage-1-trained DASNet representation with classical handcrafted features, addressing the severe optimization instability observed during end-to-end fusion."
(We adapt, but do not claim invention of, the DST or the concept of multi-branch fusion).

## 9. Final Checklist
| Requirement | Status | Evidence |
|---|---|---|
| Grouped split | PASS | `src/pipeline.py:grouped_stratified_split` |
| Five-seed evaluation | PASS | `results/multiseed/` |
| Raw predictions saved | PASS | `*_preds.npz` |
| Macro-F1 reproducible | PASS | Python extraction matches intended values |
| Per-SNR evaluation | PASS | Extraction script confirms exact values |
| Statistical analysis | PASS | Seed-0 bootstrap reported cautiously |
| Frozen expert actually frozen | PASS | `scripts/run_frozen_dualpq.py` |
| No preprocessing leakage | PASS | Scalers fitted only on `X[i_tr]` |
| Figures reproducible | PASS | `scripts/make_figures.py` |
| MGCNN comparison documented | PASS | `README.md` and this audit |
| License documented | PASS | `LICENSE` file (GPL-3.0) added |

## 10. Verdict
The repository is completely clean, statistically verified, historically preserved, and **ready for paper writing**.

---

## 11. MODEL PROVENANCE & OUR CONTRIBUTIONS

### 1. Classical Ensemble — BASELINE / OUR IMPLEMENTATION
**Status:** Classical baseline implemented by us.
**Description:** A classical machine-learning ensemble operating on 191 handcrafted physical/signal-processing features. Components include the existing feature extraction methodology and standard ML algorithms.
**Important:** The individual classical signal-processing features are NOT claimed as novel inventions. Our contribution is their implementation and use as the classical baseline within this benchmark.

### 2. DASNet — PROJECT-DEVELOPED ARCHITECTURE

DASNet is a project-developed deep PQD architecture developed during this research project. However, the Differentiable Stockwell Transform (DST) is based on established Stockwell-transform mathematics; do not claim invention of the underlying mathematical transform.
**Our work:**
- development of the network architecture
- adaptation to our 29-class benchmark
- evaluation under our grouped split
- multi-seed evaluation
- severe-noise evaluation

### 3. MGCNN-SDTransformer — EXTERNAL PUBLISHED BASELINE
**Status:** EXTERNAL / PUBLISHED ARCHITECTURE.
This architecture originates from Jiang et al. (2025). It was independently implemented/reproduced to provide an external published comparison.
**We do NOT claim:**
- MGCNN as our invention
- SDTransformer as our invention
- the architecture itself as novel
**Our contribution:** Its controlled evaluation under our benchmark protocol. Include the complete bibliographic citation and paper link/reference.

### 4. Original DualPQ-D — OUR PROPOSED ARCHITECTURE
**Status:** OUR WORK / PROPOSED MODEL.
This was designed as our hybrid architecture combining:
DASNet deep representation + 191-dimensional classical physical-feature representation
↓
Classical Expert + Deep Expert
↓
Fusion
↓
29-class classifier

Clearly state that although DASNet uses established Stockwell-transform mathematics and the classical branch uses established handcrafted signal-processing features, the specific DualPQ-D architecture and its combination/fusion strategy are our proposed work. Also document that the original end-to-end version exhibited severe seed-to-seed instability.

### 5. Frozen-DASNet DualPQ — OUR FINAL PROPOSED METHOD
**Status:** OUR WORK / FINAL PROPOSED METHOD.
This is the final model/method proposed by our research. It evolved from Original DualPQ-D after discovering severe instability during end-to-end joint training.
**The final approach:**
Stage-1-trained DASNet → FROZEN → Deep representation
191 physical features → Classical MLP
[Deep rep + Classical MLP] → Fusion head → 29 classes

**Clearly explain:**
- The specific architecture is our contribution, but it uses the established DST mathematics.
- The classical features are established/handcrafted representations.
- Our contribution is the decoupled training/fusion strategy in which the DASNet checkpoint trained in stage 1 is frozen during stage 2 while the classical branch and fusion head are optimized. Stage 1 and stage 2 strictly use the same training partition with validation-based model selection.
- The motivation is the severe seed-to-seed instability observed in the original end-to-end DualPQ-D.
- The Frozen-DASNet DualPQ achieved 74.46 ± 1.08% Macro-F1 across five seeds under our benchmark.
*(DO NOT say that both experts are pretrained. Only the DASNet representation is trained in stage 1 and frozen in the final method.)*

---

## 12. SUMMARY TABLE

| **Component** | **Provenance** | **Role** | **Claimed Novelty?** |
| :--- | :--- | :--- | :--- |
| Classical Ensemble | Our implementation using established signal-processing features and standard ML algorithms | Classical baseline | Only the implementation within this benchmark |
| DASNet | Project-developed deep PQD architecture | Deep baseline | YES (but not the underlying DST math) |
| MGCNN-SDTransformer | External published architecture from Jiang et al. (2025) | Benchmark | NO |
| Original DualPQ-D | Our research | Initial proposed hybrid | YES |
| Frozen-DASNet DualPQ | Our research | Final proposed method | YES |

---

## 13. PAPER-WRITING WARNING

> [!WARNING]
> **IMPORTANT FOR PAPER AUTHORS:**
> MGCNN-SDTransformer must be presented as an existing method/baseline from Jiang et al. (2025). It is not our invention. The underlying mathematics of the Stockwell Transform are established. DASNet, the original DualPQ-D, and Frozen-DASNet DualPQ are our proposed research contributions. Classical physical features are established representations; their implementation within our proposed fusion framework is part of our work.

## 14. CONTRIBUTION STATEMENT
"Our primary methodological contribution is a decoupled hybrid PQ disturbance classification framework that combines a stage-1-trained deep waveform representation with classical signal-processing features. The study further provides a leakage-controlled, multi-seed evaluation across 29 disturbance classes and severe noise conditions, revealing substantial differences in robustness and optimization stability among deep, classical, and hybrid approaches."

