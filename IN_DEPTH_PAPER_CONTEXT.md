# Comprehensive Research Context for Paper Generation

**Instruction for ChatGPT/LLM:** 
You are acting as an expert academic researcher writing a top-tier IEEE/ACM conference paper on Power Quality (PQ) disturbance classification. Use the following highly detailed, statistically verified context to draft the paper. Do not use the word "SOTA" or "state-of-the-art". Frame the paper around robust methodology, extreme noise immunity, and the failure of pure deep learning architectures when subjected to rigorous statistical validation.

---

## 1. The Research Problem
Modern electrical grids suffer from complex Power Quality (PQ) disturbances (e.g., sags, swells, harmonics, flickers). Identifying these accurately is critical for grid stability. 
While many recent papers claim 99%+ accuracy using Deep Learning (CNNs, DASNet), they evaluate their models under highly flawed conditions:
1.  **Data Leakage:** They use "Random Splitting". Because PQ datasets synthesize multiple variations (noise levels) of the *same* underlying base waveform, Random Splitting leaks different noise variants of the exact same signal into both the train and test sets.
2.  **Ignored Extreme Noise:** They often exclude 10dB and 0dB (extreme noise) environments from their evaluations, only testing on Clean or 40dB data.

**Our Goal:** To evaluate a hybrid architecture (DualPQ-Net) against pure deep learning (DASNet) and a classical machine learning ensemble, using a mathematically rigorous, zero-leakage evaluation protocol under extreme noise (down to 0dB).

---

## 2. The Architectures Evaluated
We evaluated three distinct paradigms:
1.  **DASNet (Deep-Only Baseline):** A purely deep-learning architecture utilizing a Learnable Discrete Stockwell Transform (DST) and 1D-CNNs to process raw voltage waveforms. 
2.  **Classical Stacking Ensemble (The Winner):** A purely classical machine learning approach. It extracts 191 mathematically defined physical features (e.g., cyclic RMS, flicker severity, THD, specific harmonics) from the waveforms. These features are fed into a Stacking Classifier (Random Forest + LightGBM + MLP + SVM) using geometric soft-voting.
3.  **DualPQ-Net (Proposed Hybrid):** A dual-branch architecture. Branch 1 (Deep Expert) processes the raw waveform using DASNet. Branch 2 (Classical Expert) processes the 191 physical features using an MLP. The embeddings are concatenated and passed through a classification head.

---

## 3. The Rigorous Evaluation Protocol
To ensure absolute scientific integrity, we implemented two strict protocols:
*   **Grouped Stratified Splitting:** We grouped all noise variants of a specific base waveform together. If a waveform is in the training set, its 0dB, 10dB, and 40dB versions are *only* in the training set. Zero data leakage to the test set.
*   **5-Seed Statistical Validation:** Deep learning models are highly sensitive to random weight initialization. We trained every single architecture 5 completely independent times (using seeds 0, 1, 2, 3, 4). The final reported scores are the mathematically sound **Mean ± Standard Deviation** across all 5 runs.

---

## 4. Final Empirical Results (Macro F1-Score)
*Note: Evaluated across 29 PQ classes containing Clean, 40dB, 30dB, 20dB, 10dB, and 0dB signals.*

### Overall 5-Seed Averages:
*   **Classical Ensemble:** **72.02% (± 0.24%)**
*   **DASNet (Deep Only):** 69.72% (± 8.15%)
*   **DualPQ-Net (Hybrid):** 61.63% (± 13.94%)

### Degradation Under Extreme Noise (Mean F1 by SNR):
| SNR Level | Classical Ensemble | DASNet | DualPQ-Net |
|-----------|:---:|:---:|:---:|
| **Clean** | **87.49%** | 84.79% | 77.44% |
| **40dB**  | 87.07% | **90.16%** | 78.51% |
| **30dB**  | 86.46% | **87.18%** | 77.16% |
| **20dB**  | **81.19%** | 75.85% | 67.01% |
| **10dB**  | **62.78%** | 51.70% | 35.68% |
| **0dB**   | **26.67%** | 20.70% | 12.79% |

---

## 5. Key Scientific Findings & Discussion points for the Paper
1.  **The Deep Learning Variance Trap:** If we had only run one training seed (Seed 0), DualPQ-Net scored 72.6% and DASNet scored 53.4%. A single run would have falsely proven DualPQ-Net was superior. However, across 5 seeds, DualPQ-Net showed extreme instability (±13.9% variance) with multiple catastrophic training failures. Multi-seed validation proved DASNet was actually more stable than the hybrid approach.
2.  **The Superiority of Physical Math under Extreme Noise:** The Classical Stacking Ensemble won the overall evaluation. While Deep Learning (DASNet) performed exceptionally well in low-noise environments (40dB/30dB), it suffered catastrophic feature-collapse at 10dB and 0dB. 
3.  **Why the Classical Model Won:** Deep learning convolutions struggle to find spatial patterns in waveforms completely corrupted by 0dB zero-mean Gaussian noise. However, classical features (like Harmonic Ratios, Crest Factors, and Cyclic RMS) are governed by physical equations. These mathematical transformations act as absolute filters, making them inherently immune to high-frequency random noise. Consequently, the classical stack retained significantly more accuracy at 0dB (26.67%) compared to the deep models (20.70% and 12.79%).

**Conclusion:** Pure deep learning on raw waveforms is brittle in real-world grid scenarios where extreme noise is present. To achieve true robustness, models must heavily bias toward mathematically defined classical features rather than relying purely on learned convolutions.
