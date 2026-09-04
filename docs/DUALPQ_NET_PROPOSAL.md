# DUALPQ-NET: Project Proposal

> [!WARNING]
> **Historical / Proposal Document**
> This document is the original project proposal and contains early hypotheses and claims that have since been revised. While DASNet is a project-developed architecture, the Differentiable Stockwell Transform (DST) relies on established mathematical formulations and is not our invention. The final authoritative methodology and novelty claims are documented in `README.md` and `PUBLICATION_AUDIT.md`.

**Formal Proposal for Conference Paper Architecture**

This document outlines the transition from the initial Deep Learning attempt (DASNet) to the proposed hybrid architecture (DualPQ-Net) based on extensive literature research and experimental failure analysis.

---

## 1. What OpenCode Did (DASNet)

OpenCode designed a pure Deep Learning architecture called **DASNet**. Its core novelty was the **Differentiable Stockwell Transform (DST)**. 

### The Innovation
In all traditional literature, the Stockwell (S-) Transform uses a fixed mathematical window. OpenCode made this window learnable via gradient descent. This was a brilliant move because it directly solved the "flicker blindness" problem documented in your baseline. 

### The Experimental Result
* **Clean Data Score:** 0.9303 (Massive improvement over the 0.8859 baseline).
* **Overall Score:** 0.6355 (Failed to beat the 0.7212 baseline).
* **The Failure Mode:** The CNN collapsed completely under heavy noise (e.g., scoring only 0.0643 at 0dB, compared to the baseline's 0.2588). 

**Conclusion:** Pure Deep Learning (CNNs over raw S-transform images) cannot handle extreme background noise as effectively as your 191 handcrafted signal-processing features, which were mathematically designed to filter noise out before classification.

---

## 2. Literature Research Findings

Before redesigning the architecture, a comprehensive search of arXiv, IEEE, and Crossref was performed:
1. **Differentiable Stockwell Transform:** Extremely novel. While learnable STFTs exist in speech processing, making the Stockwell Transform learnable in the Power Quality domain is virtually unpublished.
2. **Hybrid CNN + Handcrafted Features:** Common, but usually done via simple concatenation.
3. **SNR-Conditioned Gating:** Highly novel in this context. Dynamically shifting the network's reliance between two distinct pathways based on real-time noise estimation has not been applied to PQ disturbance classification.

---

## 3. What We Are Going To Do (DualPQ-Net)

To secure a state-of-the-art (SOTA) result and a strong conference publication, we will build **DualPQ-Net** (Dual-Path Power Quality Network). 

This architecture explicitly acknowledges that Deep Learning and Classical Signal Processing have opposite strengths, and mathematically fuses them.

### The Architecture
1. **Path 1 (Deep Learning):** 
   - Takes the raw 1280-point waveform as input.
   - Passes through the novel Differentiable Stockwell Transform (DST) and a CNN backbone.
   - *Purpose:* Dominates on clean data and complex spatial patterns.
2. **Path 2 (Classical Expert):**
   - Takes your 191 handcrafted features as input.
   - Passes through a regularized Multi-Layer Perceptron (MLP).
   - *Purpose:* Dominates on highly noisy data due to mathematical noise immunity.
3. **The SNR-Conditioned Gate (The Core Novelty):**
   - We will use the existing internal SNR Estimator to calculate the noise level of the current signal.
   - A Sigmoid gating mechanism will learn to output a weight between 0 and 1. 
   - At high SNR (clean), the gate heavily weights Path 1. 
   - At low SNR (noisy), the gate heavily weights Path 2.

### Next Steps for Implementation
1. **Modify Data Loader:** Update `scripts/run_dasnet.py` to feed both the waveform and the 191 features into the PyTorch training loop simultaneously.
2. **Rewrite the PyTorch Model:** Update `src/dasnet.py` to include the Path 2 MLP and the Sigmoid Gating mechanism.
3. **Train & Evaluate:** Run the model on the exact same leakage-free dataset splits to prove that it beats the 0.7212 baseline across the entire noise spectrum.

*Outcome:* Frozen-DASNet DualPQ reaches 74.46 ± 1.08, clearing the 0.7212
target set here by +2.34 pp — but only in the clean/40 dB/30 dB regime. Across
"the entire noise spectrum" it does **not** hold: at 20, 10 and 0 dB the
difference from the classical ensemble is not statistically distinguishable
from zero (README §8). See `README.md` §7 for how 0.7212 relates to the
five-seed figures quoted in the paper.
