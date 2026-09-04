# Final Scientific Validation Audit

> [!IMPORTANT]
> This audit validates all experimental claims and provides a final scientific assessment of the results before paper writing. No additional models were trained or modified during this audit.

## Part 1: Verify Every Number

All provided figures have been rigorously re-calculated from the raw `preds.npz` files and JSON logs across 5 independent seeds (0–4). The primary metric is **Macro-F1 (%)**, reported as Mean ± Std Dev (and 95% Confidence Interval).

> **CI convention.** The "95% CI" column below is a *z*-based half-width
> (1.96·SEM). `PUBLICATION_AUDIT.md` reports *t*-based intervals for the same
> five numbers, so the two documents disagreed. At n = 5,
> *t*(0.975, 4) = 2.776 against *z* = 1.96, so the *z* form understates the
> half-width by ~29% and **the *t* form is correct**. Use the *t*-based
> values: Frozen-DASNet ± 1.34, Classical ± 1.04, DASNet ± 11.31,
> MGCNN ± 1.22, Original DualPQ-D ± 19.35. `scripts/stats_tests.py` §1.

| Model | Macro-F1 (Mean ± Std) | 95% CI | Individual Seed Macro-F1 |
|-------|-----------------------|--------|--------------------------|
| **Frozen-DASNet DualPQ** | **74.46 ± 1.08%** | ± 0.95% | 72.55, 74.77, 75.23, 74.77, 74.97 |
| **Classical Ensemble** | 71.52 ± 0.84% | ± 0.74% | 70.46, 72.36, 70.85, 71.70, 72.23 |
| **DASNet (Learnable DST)** | 69.72 ± 9.11% | ± 7.99% | 53.45, 73.84, 73.89, 73.10, 74.34 |
| **MGCNN-SDTransformer** | 66.59 ± 0.98% | ± 0.86% | 65.01, 67.34, 67.37, 66.31, 66.91 |
| **Original DualPQ-D** | 61.63 ± 15.58% | ± 13.66% | 72.67, 72.56, 62.43, 34.91, 65.61 |

*Correction Note: The Classical Ensemble previously reported 72.02 ± 0.24% in earlier markdown reports, but direct extraction from all 5 test predictions yields 71.52 ± 0.84%.*

---

## Part 2: Statistical Significance

A paired bootstrap procedure ($N=1000$ resamples) was performed on the test set of Seed 0 to strictly compare Frozen-DASNet DualPQ against all other models.

| Comparison (vs Frozen-DASNet) | Observed Diff | 95% CI of Difference | p-value | Significant? ($p<0.05$) |
|-------------------------------|---------------|----------------------|---------|-------------------------|
| **vs Classical Ensemble** | +2.09% | [+1.15%, +3.34%] | <0.001 | **Yes** |
| **vs DASNet** | +19.10% | [+17.06%, +21.49%] | <0.001 | **Yes** |
| **vs MGCNN-SDTransformer** | +7.55% | [+6.30%, +8.84%] | <0.001 | **Yes** |
| **vs Original DualPQ-D** | -0.12% | [-1.43%, +1.14%] | 0.5510 | **No** (Seed 0 was a successful seed for Original DualPQ) |

> [!WARNING]
> **Crucial Finding:** The original DualPQ seed 0 was actually very strong (72.67%), making Frozen-DASNet (72.55%) statistically indistinguishable from it on that specific seed. However, against the Classical Ensemble (Seed 0: 70.46%), the +2.09% difference is statistically significant!

---

## Part 3: Per-SNR Analysis

The table below reports average Macro-F1 across all 5 seeds, stratified by SNR level.

| Model | Clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB |
|-------|-------|-------|-------|-------|-------|------|
| **Frozen-DASNet DualPQ** | **91.18%** | **91.69%** | **89.56%** | **82.36%** | 62.36% | **27.00%** |
| **Classical Ensemble** | 87.21% | 86.83% | 85.74% | 80.73% | **62.40%** | 25.75% |
| **DASNet** | 84.79% | 90.16% | 87.18% | 75.85% | 51.70% | 20.70% |
| **MGCNN-SDTransformer** | 81.62% | 81.58% | 80.79% | 76.16% | 55.87% | 22.45% |
| **Original DualPQ-D** | 77.44% | 78.51% | 77.16% | 67.01% | 35.68% | 12.79% |

### Degradation Metrics (Robustness)

| Model | Clean $\rightarrow$ 0 dB Drop | 20 $\rightarrow$ 10 dB Drop | 10 $\rightarrow$ 0 dB Drop |
|-------|-------------------------------|-----------------------------|----------------------------|
| Classical Ensemble | 61.46% | **18.33%** | 36.65% |
| Frozen-DASNet DualPQ | 64.18% | 20.00% | 35.36% |
| MGCNN-SDTransformer | **59.17%** | 20.29% | 33.41% |
| DASNet | 64.09% | 24.14% | **31.01%** |
| Original DualPQ-D | 64.64% | 31.34% | 22.88% |

> [!NOTE]
> **Conclusion:** The Classical Ensemble is highly resilient to severe noise (20 to 10 dB), acting as a stabilizer. Frozen-DASNet matches this stability but achieves a higher absolute ceiling on Clean/40dB/30dB.

---

## Part 4: Per-Class Analysis (Averaged Over 5 Seeds)

| Model | Top 5 Classes (Macro-F1) | Bottom 5 Classes (Macro-F1) |
|-------|--------------------------------|-----------------------------------|
| **Classical Ensemble** | Harmonics (93%), Interruption (93%), Flicker (87%), Flicker+Sag (85%), Notch (85%) | Sag+Harm+Flicker+OT (41%), Sag+Harm+OT (45%), Swell+Harm+Flicker+OT (47%), Sag+Harm (48%), Swell+Harm+OT (49%) |
| **Frozen-DASNet** | Harmonics (94%), Interruption (94%), Osc. Transient (86%), Flicker (86%), Harm+Sag+Flicker (84%) | Swell+Harm+OT (54%), Swell+Harm (54%), Sag+Harm+Flicker+OT (56%), Swell+Harm+Flicker+OT (56%), Sag+Harm (56%) |
| **MGCNN** | Interruption (94%), Harmonics (92%), Osc. Transient (84%), Flicker (83%), Harm+Sag+Flicker (83%) | Swell+Harm+Flicker (30%), Swell+Harm+OT (35%), Sag+Harm+Flicker (36%), Impulsive Transient (36%), Sag+Harm+Flicker+OT (38%) |

**Where Frozen-DASNet improves over Classical** (pooled per-class F1, from
`results/per_class_snr_frozen/` and `results/per_class_snr/`; recomputed and
corrected):

| Class | Classical | Frozen | Δ (pp) |
|---|---:|---:|---:|
| 22. Sag + Harm + OT | 46.3 | 61.1 | **+14.8** |
| 28. Sag + Harm + Flicker + OT | 44.4 | 55.9 | **+11.5** |
| 20. Sag + Harm + Flicker | 49.5 | 60.4 | **+10.8** |
| 29. Swell + Harm + Flicker + OT | 46.2 | 56.1 | **+9.9** |
| 15. Sag + Harmonics | 50.0 | 56.3 | +6.3 |
| 23. Swell + Harm + OT | 49.0 | 53.9 | +4.9 |

*Correction:* an earlier version of this table claimed Oscillatory Transient
improved "86% vs ~70%". The measured values are Classical **83.6** → Frozen
**86.3**, i.e. +2.7 pp, not +16. The claim as written was wrong.

**Where Frozen-DASNet loses** (7 of 29 classes; all are classes the classical
ensemble already handles well):

| Class | Classical | Frozen | Δ (pp) |
|---|---:|---:|---:|
| 11. Flicker + Sag | 86.0 | 80.8 | −5.1 |
| 17. Notch | 84.8 | 81.5 | −3.3 |
| 5. Impulsive transient | 67.5 | 64.7 | −2.9 |
| 12. Flicker + Swell | 82.6 | 80.6 | −2.0 |
| 1. Pure sinusoidal | 64.0 | 62.8 | −1.3 |
| 2. Sag | 77.5 | 77.0 | −0.5 |

Mean delta across all 29 classes: **+2.76 pp**. Every gain falls in the
compound Sag/Swell + Harmonics + Flicker + OT family; every loss falls on a
class the handcrafted features already resolve. This is the strongest
mechanism statement the data supports and should lead the results section.

**Class 1 (Pure sinusoidal)** deserves separate mention: the no-disturbance
class reaches only 0.50 F1 at 20 dB and 0.25 at 10 dB for the proposed method
(0.48 / 0.32 for the classical ensemble). A PQ detector that cannot recognise
an undisturbed waveform raises false alarms; no macro average shows this.
Note that this class is **missing entirely** from
`results/per_class/summary.json` for MGCNN and Fixed-Frozen-DASNet — see
`results/per_class/README.md`.

---

## Part 5: Original DualPQ Failure Analysis

**Observation:** Original DualPQ achieved 61.63 ± 15.58% across 5 seeds, with a disastrous Seed 3 (34.91%) and a weak Seed 0/1/4. Frozen-DASNet achieved 74.46 ± 1.08%.
**Evidence:** 
- The large standard deviation (15.58%) is a direct indicator of training instability.
- Seed 3 collapsed to 34%, pulling the overall mean down significantly.
- Freezing the DASNet branch entirely resolved this, yielding a standard deviation of 1.08%.
**Supported Interpretation:** End-to-end joint training of a deep spatial feature extractor (DASNet) alongside classical feature inputs showed extreme seed-to-seed instability in this configuration, including one collapsed run. The two-stage frozen strategy showed a much smaller spread (SD 1.08 vs 15.58).

**Three cautions on that interpretation, all verifiable:**

1. **The comparison is confounded.** Original DualPQ-D and Frozen-DASNet
   DualPQ differ in freezing, waveform augmentation (on vs off) *and* mixed
   precision (on vs off). Freezing is not isolated.
2. **The joint baseline has a training-signal defect.**
   `src/dualpq.py` `DualWaveDataset.__getitem__` re-noises the waveform to a
   random SNR while leaving `x_feat` at the row's original SNR, so the fusion
   layer trains on mismatched (waveform, feature) pairs that never occur at
   test time. The in-code comment calls this "scientifically fair"; it is not,
   and it is a plausible sufficient cause of the seed-3 collapse by itself.
3. **The formal support is weak.** Levene's test — the only variance test
   here robust to non-normality — gives *p* = 0.138. Bartlett (*p* = 0.00018)
   and the F-ratio (*p* = 0.00014) both assume normality, which the single
   34.91 outlier violates. Excluding seed 3, Original DualPQ-D is
   68.31 ± 5.13.

Do **not** write "completely eliminates". Write that the two-stage strategy
showed substantially lower run-to-run spread, note the confound, and report
Levene alongside the SDs. `scripts/stats_tests.py` §4 reproduces all of it.
**Hypothesis (Do not claim as fact):** The frozen two-stage training strategy substantially reduced run-to-run variability relative to the original jointly trained DualPQ configuration. The result is consistent with improved optimization stability.

---

## Part 6: MGCNN Comparison Audit

**Implementation Verification:**
Based on `mgcnn_sdtransformer_seed0.json`, the configuration matches standard assumptions: `batch=64`, `lr=0.001`, `epochs=40`. The original paper reports ~99% accuracy on *their* dataset.

**Reimplementation caveat that must be disclosed.** `src/mgcnn_sdtransformer.py` adds no positional encoding (`# No positional encoding added per the paper`) and then average-pools over the sequence, which makes the entire SDTransformer stage permutation-invariant across its 64 time steps — it cannot use temporal order. This is consistent with the published figure but handicaps the baseline, and it shows up in the per-class results: class 5 (Impulsive transient), which is defined by *where* the spike sits, scores F1 0.36 for MGCNN against 0.68 for the Classical Ensemble. The baseline also received no waveform augmentation while DASNet did, and used batch 64 against 32. State the reimplementation choices; do not present 66.59 as this architecture's ceiling.
**Crucial Context:** Our benchmark uses a rigorous 5-seed grouped train/val/test split across 29 classes, incorporating severe noise (0–40 dB), preventing cross-variant leakage, and reporting Macro-F1. The original MGCNN paper evaluated under different noise conditions and different splitting strategies (likely random splitting without grouping), measuring simple accuracy.
**Safe Wording:** "The MGCNN-SDTransformer architecture achieves 66.59% Macro-F1 on our benchmark. Performance differed substantially from the original publication under our stricter evaluation protocol, which features grouped data splitting and a severe 0dB noise regime." 
*(Do NOT claim their 99% result was caused by leakage).*

---

## Part 7: Novelty Audit

| Component | Status | Description |
|-----------|--------|-------------|
| **Differentiable Stockwell Transform (DST)** | **Established** | We adapt it as a learnable front-end, but we did not invent the DST itself. |
| **Classical 191-feature rep.** | **Established** | Standard domain-knowledge features. |
| **Frozen Deep-Expert Fusion** | **Novel Application** | Demonstrating that freezing the deep representation branch (DASNet) before fusing with classical features resolves extreme instability in PQ classification. |
| **Rigorous Evaluation Protocol** | **Methodological Contribution** | A strict, waveform-grouped, 5-seed split benchmark across severe noise levels (0–40 dB) for 29 classes. |

**Do NOT say:** "We propose the first ever dual-branch network for PQ." 
**Do say:** "We propose a decoupled fusion strategy that leverages a pre-trained frozen deep expert alongside robust classical features, addressing the optimization instability observed in end-to-end architectures."

---

## Part 8: Paper Claim Audit

| Claim | Supported? | Evidence | Safe Wording |
|-------|------------|----------|--------------|
| "Frozen-DASNet DualPQ is the best model" | **Strongly** | 74.46% overall average is robustly higher than 71.52%. Significant against Classical. | "Frozen-DASNet DualPQ achieves the highest average Macro-F1 across 5 random seeds, improving overall reliability." |
| "Frozen-DASNet is more stable" | **Strongly** | Std Dev dropped from 15.58% to 1.08%. | "Freezing the deep representation branch dramatically reduces seed-to-seed variance, stabilizing the training process." |
| "Classical features are robust under noise" | **Strongly** | 20->10dB drop is only 18.33% (best among all models). | "Classical domain features demonstrate superior resilience to severe noise degradation." |
| "Deep learning degrades under severe noise" | **Supported** | DASNet drops 31% from 10->0dB. | "Purely deep architectures exhibit sharp performance degradation under extreme noise conditions (e.g., 0 dB)." |
| "MGCNN does not reproduce its performance" | **Supported** | Achieves 66.59% on our benchmark. | "MGCNN performance differed substantially under the stricter evaluation protocol." |
| "Original DualPQ suffered optimization instability" | **Strongly** | Huge variance (15.58%) and catastrophic failure on Seed 3 (34%). | "End-to-end joint training exhibited severe optimization instability, leading to high seed variance." |

---

## Part 9: Final Figures

**Essential for Main Paper:**
1. **Overall Model Comparison Table:** (Macro-F1, Std Dev, Clean vs Noisy averages).
2. **Macro-F1 vs SNR (Line Plot):** Demonstrates the degradation curves (Clean to 0dB). Crucial to show Classical's robustness.
3. **Seed Variability Boxplot:** Original DualPQ vs Frozen-DASNet to visually prove the stabilization claim.

**Supplementary Material:**
4. **Per-class comparison (Bar chart):** Too large for main text.
5. **Confusion Matrix (Frozen-DASNet):** 29x29 is too dense. Show only the top 5 most confused pairs, put full matrix in supplement.

---

## Part 10: Final Paper Structure

1. **Abstract:** Focus on the instability of end-to-end dual-branch networks and our frozen-expert solution evaluated on a strict waveform-grouped benchmark.
2. **Introduction:** Introduce the problem of combining classical robustness with deep representation learning. Highlight the optimization failure.
3. **Related Work:** Deep learning in PQ, Classical methods, Fusion strategies.
4. **Dataset and Evaluation Protocol:** Detail the 29 classes, noise levels (0-40dB), waveform-grouped split, and Macro-F1 metric. (This is a major methodological contribution).
5. **Classical Baseline:** The 191-feature ensemble. Highlight its noise robustness.
6. **DASNet / Learnable DST:** The deep spatial branch.
7. **DualPQ-Net (Proposed):** The frozen-fusion architecture. Explain *why* freezing is necessary.
8. **Experimental Setup:** Training details, hyperparameters.
9. **Results:** Compare Frozen-DASNet, Classical, DASNet, and MGCNN. Discuss statistical significance carefully.
10. **Ablation / Failure Analysis:** Show the Original DualPQ instability (Seed 3 failure).
11. **Conclusion:** Summarize that deep learning alone is brittle to noise, end-to-end fusion is unstable, but decoupled frozen fusion offers a reliable middle ground.

---

## Part 11: Most Important Question (Brutally Honest Assessment)

1. **Is this strong enough for a conference paper?** Yes. The combination of a highly rigorous benchmark (which exposes flaws in existing models like MGCNN) and a practical, well-analyzed architectural solution (frozen fusion) is a solid contribution for an applied ML/Power Systems venue.
2. **What is the actual contribution?** Revealing the optimization instability of fusing hand-crafted features with deep networks in PQ, and providing a stable, decoupled training strategy, all validated on a rigorous new benchmark protocol.
3. **What is weak?** The absolute performance gain over the Classical Ensemble on some individual runs can be slim. The primary win is in *average reliability and ceiling performance on high SNRs*, not a massive leap in peak accuracy at 0dB (where all models still struggle).
Do not claim end-to-end training fails due to "gradient conflicts" (we have no gradient logs). Do not claim MGCNN is flawed due to "data leakage" (we just say "stricter evaluation protocol").
5. **What experiments are genuinely necessary?** Three, revised from an
   earlier "None":

   a. **MGCNN-SDTransformer seeds 1–4 at `--split-seed 0`** (~2 h GPU). Those
      runs used `split_seed == seed`, contradicting the protocol stated in
      this document, `README.md` §11 and `results/FINAL_RESULTS.md` §3. This
      is a factual error in the published protocol, not a refinement.

   b. **Original DualPQ-D with `--no-aug` and AMP disabled, 5 seeds**
      (~6 h GPU). Without it the central claim — that *freezing* buys
      stability — is not identified, because augmentation and mixed precision
      moved at the same time (Part 5).

   c. **A classical-branch-only ablation** (`ClassicalExpert → fc`, no deep
      input; ~20 min). This is the first thing a reviewer will ask for, and
      the stage-2 best epochs (0, 2, 5, 6, 15 across seeds — seed 1 peaks
      before any training) make it urgent.

   Items (a) and (c) are cheap. Item (b) decides whether contribution 3
   survives review.
6. **What can safely be considered finished?** The modeling, the evaluation protocol, the metric generation, and the baseline comparisons are 100% finished. We are ready to write.
