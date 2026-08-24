# 29-Class Power-Quality Disturbance Classification
## Multi-SNR ensemble over Stockwell + time + frequency features

Dataset generator: `pqmodel.m` — Igual, Medrano, Arcega & Mantescu, *Integral
mathematical model of power quality disturbances* (2017), ported to NumPy.

---

## 1. Headline result

| | macro F1 | balanced acc | accuracy |
|---|---|---|---|
| **Weighted soft vote (best)** | **0.6885** | 0.6903 | 0.6903 |
| Geometric vote *(selected on validation)* | 0.6870 | 0.6883 | 0.6883 |
| LightGBM (best single model) | 0.6807 | 0.6821 | 0.6821 |
| Equal-weight soft vote | 0.6699 | 0.6713 | 0.6713 |
| Stacked meta-learner | 0.6657 | 0.6694 | 0.6694 |

Test macro-F1 by SNR, weighted vote:

| SNR | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB |
|---|---|---|---|---|---|
| all 29 classes | 0.868 | 0.865 | 0.811 | 0.623 | 0.268 |
| **21 classes, excluding the 4 degenerate pairs** | **0.983** | **0.961** | **0.922** | 0.709 | 0.335 |

That second row is the important one. **The residual error is not spread across
the taxonomy — it is concentrated almost entirely in four class pairs that the
generator makes nearly indistinguishable by construction.** On everything else
the system is at 98% at 40 dB and 92% at 20 dB. Section 5 quantifies why.

---

## 2. Configuration

Confirmed generator settings, `pqmodel(ns, fs, f, n, A)`:

| parameter | value | reason |
|---|---|---|
| `fs` | 6400 Hz | 128 samples/cycle. Nyquist 3200 Hz clears the 300–900 Hz oscillatory-transient band; notch widths (0.2–1.0 ms) span 1.3–6.4 samples so class 17 survives sampling. |
| `f` | 50 Hz | |
| `n` | 10 cycles | 1280 points, 200 ms |
| `A` | 1.0 pu | |
| `ns` | 200 / class | 5800 base waveforms |
| SNR | 40 / 30 / 20 / 10 / 0 dB | AWGN, calibrated to within 0.01 dB |

**29,000 rows = 29 classes × 200 base waveforms × 5 SNR levels.**
191 features. S-transform band-limited to 0–1600 Hz at 5 Hz resolution
(321 × 1280 complex per signal).

### Three contradictions in the original brief, and how they were resolved

1. **"75/15/15" sums to 105%** → used 70/15/15.
2. **"stratified 10-fold" and a fixed holdout split are different protocols** →
   reconciled as: 70/15/15 grouped holdout, with `StratifiedGroupKFold(10)`
   *inside the training partition* to produce the out-of-fold probabilities the
   stacking design requires. Both requests are satisfied and neither is
   decorative.
3. **SNR levels differed between the two paragraphs** (40/30/20/10/0 vs
   −6/0/6/12/18) → used 40/30/20/10/0 dB as specified in the requirement line.

---

## 3. Protocol

```
5800 base waveforms
   └── grouped, stratified 70/15/15 split      (4060 / 870 / 870 groups)
         └── each group carries all 5 SNR copies  (20300 / 4350 / 4350 rows)
               └── StratifiedGroupKFold(10) inside train → OOF probabilities
                     └── meta-learner + voting weights fitted on OOF
                           └── validation selects the ensemble
                                 └── test scored once
```

Splitting is by **base waveform, never by row**, so no noise-augmented sibling
of a test waveform is ever seen in training. This makes the evaluation a test of
generalisation to *new disturbances*, not to new noise draws. All partitions are
exactly balanced: 700 / 150 / 150 rows per class, and equal counts at every SNR
level.

### Base models

| model | 10-fold CV macro-F1 | test macro-F1 | OOF voting weight |
|---|---|---|---|
| Random Forest (300 trees) | 0.6476 ± 0.0152 | 0.6681 | 0.591 |
| LightGBM (250 iter) | 0.6581 ± 0.0109 | 0.6807 | 0.231 |
| SVM-RBF (quantile → PCA-50) | 0.5681 ± 0.0113 | 0.5791 | **0.004** |
| MLP (256-128) | 0.6346 ± 0.0106 | 0.6541 | 0.174 |

CV and test agree closely (0.648 vs 0.668 for RF), so the split is not
optimistic and the models are not badly overfit.

**Equal-weight soft voting scored *below* the best single model** (0.670 vs
0.681). This is expected once the members are unequal: averaging a
poorly-calibrated SVM at 0.579 into a LightGBM at 0.681 destroys more than it
adds. Fitting weights on the out-of-fold probabilities fixes it — the search
drove the SVM weight to 0.004, effectively deleting it — and recovers +0.019
over equal weighting.

### Both ensemble claims are statistically significant

A single run cannot distinguish a real +0.009 from a lucky split. Repeating over
5 independent 70/15/15 partitions of the same data and applying a paired t-test
(`multiseed.py`) settles both directions:

| method vs best single model (LightGBM) | mean diff | wins | p | verdict |
|---|---|---|---|---|
| equal-weight vote | −0.0084 | 0/5 | 0.016 | **significantly worse** |
| weighted vote | +0.0089 | 5/5 | 0.0038 | **significantly better** |

Pairing is what makes this resolvable. Each method's own scores scatter by
σ ≈ 0.009 across splits, so their confidence intervals overlap heavily and a
naive CI comparison would call it a tie. But every method sees the *identical*
splits, so that variance is shared and cancels in the difference: the paired
differences scatter by only ≈ 0.003. **Comparing overlapping confidence
intervals is the wrong test for a paired design, and here it would have hidden
a real effect.**

A note on the fitted weights (`rf=0.576  lgbm=0.234  svm=0.031  mlp=0.160`,
averaged over splits): Random Forest takes the largest share despite LightGBM
being the stronger individual model. Ensemble weight follows *complementarity*,
not individual accuracy.

---

## 4. Finding 1 — the S-transform is blind to flicker at the fundamental

In the first full run, **13 of the 14 largest confusions were flicker /
no-flicker pairs**. Four rounds of controlled AUC experiments
(`exp_flicker*.py`) traced this to a property of the transform itself.

The S-transform window at frequency *f* has time width σ_t = 1/*f*, hence
frequency width **σ_f = f / 2π**:

| S-transform row | modulation bandwidth | effect on 8–25 Hz flicker |
|---|---|---|
| 50 Hz (fundamental) | σ_f ≈ 8 Hz | **destroyed** — attenuated up to ~140× at 25 Hz |
| 150 Hz (3rd harmonic) | σ_f ≈ 24 Hz | preserved |
| 250 Hz (5th harmonic) | σ_f ≈ 40 Hz | preserved |

Any flicker feature read off the fundamental row of |S| — which is the standard
construction in the literature — is measuring a signal that the transform has
already low-passed away. The control experiment confirmed the detectors
themselves were fine: pure-sinusoid vs pure-flicker gave AUC = 1.000; it was
only the *measurement* that was broken.

**The fix turns the defect into an asset.** Because the 50 Hz row contains the
rectangular sag/swell but *not* the flicker, it is an ideal baseline estimator.
So:

- detect the sag/swell event from the (flicker-free) S-envelope;
- measure flicker on the full-bandwidth **Hilbert envelope**, restricted to
  samples outside the event and one cycle clear of its edges;
- detect it **coherently** — projection onto exp(j2π·ff·t) swept over
  8–25 Hz — rather than from a periodogram, which recovers the ~30 dB of
  processing gain available over a 1280-sample window.

Separation of the resulting feature `flk_h1_out` on clean signals:

| pair | without flicker → with flicker |
|---|---|
| sag → sag+flicker | 0.0008 → 0.0388 (48×) |
| swell → swell+flicker | 0.0006 → 0.0418 (70×) |
| harm+sag → +flicker | 0.0000 → 0.0367 |

`flk_h1_out` ranks 6th of 191 in Random-Forest importance. Every
"global-flicker" pair — classes 11, 12, 18, 19, 26, 27 — now sits at 0.93–1.00
recall down to 20 dB.

---

## 5. Finding 2 — four class pairs are near-degenerate by construction

After the fix, the confusions that remain at 40 dB are **exclusively** these:

```
 14/150   c22 Sag+Harm+OT            ↔  c28 Sag+Harm+Flicker+OT
 14/150   c29 Swell+Harm+Flicker+OT  ↔  c23 Swell+Harm+OT
 12/150   c20 Sag+Harm+Flicker       ↔  c15 Sag+Harmonics
 10/150   c16 Swell+Harmonics        ↔  c21 Swell+Harm+Flicker
```

Reading the model equations shows flicker enters the 29 classes in two
structurally different ways:

**GLOBAL** — the factor `(1 + λ·sin(2π·ff·t))` multiplies the whole signal:
```matlab
class 18 = A*AFinal.*(harm).*(1 + lambda*sin(2*pi*ff*t))
```

**GATED** — it multiplies *only* the sag-gated harmonic term, which is **zero
outside the event window**:
```matlab
class 15 = A*( sin(w0*t-th1) + (harm).*(-alpha*u) )
class 20 = A*( sin(w0*t-th1) + (harm).*(-alpha*u).*(1 + lambda*sin(2*pi*ff*t)) )
```
Outside `u`, classes 15 and 20 are *identical*. All the evidence separating them
is `α · λ · (harmonic amplitude)`, confined to part of the window.

Measuring the waveform difference ‖δ‖ directly from matched-parameter pairs
(`exp_degeneracy.py`) and converting to the matched-filter ceiling Φ(d′/2):

| kind | pairs | ‖δ‖ rms | dB below signal | ceiling AUC @ 20 dB | @ 10 dB | @ 0 dB |
|---|---|---|---|---|---|---|
| GLOBAL | 2→11, 3→12 | 0.0370 | 25.6 | 1.000 | 0.998 | 0.825 |
| GLOBAL | 8→18, 9→19 | 0.0299 | 27.5 | 1.000 | 0.992 | 0.775 |
| **GATED** | 15→20, 16→21 | **0.0132** | **34.6** | 1.000 | 0.854 | 0.630 |
| **GATED** | 22→28, 23→29 | **0.0126** | **35.0** | 0.999 | 0.843 | 0.625 |

The gated pairs carry roughly **10× less evidence** than the global ones, and
what little there is sits inside a sub-window. This is a property of the
Igray/Igual generator, not of the feature set — and it is worth knowing before
comparing against published 29-class accuracies on this model.

Collapsing each gated pair to a single label (25 classes) gives **0.975 macro-F1
at 40 dB**; excluding those 8 classes entirely (21 classes) gives **0.983**.

Importantly, the ceiling is *not* yet reached: at ≥20 dB the matched filter
scores ~1.000 while the classifier scores ~0.5–0.7 on these pairs. The
information is present; the current features do not extract it. Recommendation
1 targets exactly this gap.

---

## 6. SNR robustness — leave-one-SNR-out

Train on four SNR levels, test on held-out waveforms at the fifth:

| held out | regime | vote macro-F1 | reference (trained on all 5) | gap |
|---|---|---|---|---|
| 40 dB | extrapolation | 0.8295 | 0.8682 | +0.039 |
| 30 dB | interpolation | 0.8446 | 0.8686 | +0.024 |
| 20 dB | interpolation | 0.7621 | 0.8099 | +0.048 |
| 10 dB | interpolation | 0.4403 | 0.6228 | **+0.183** |
| 0 dB | extrapolation | **0.0508** | 0.2574 | **+0.207** |

**The model extrapolates upward in SNR but not downward — at all.** Held out,
0 dB collapses to 0.051, barely above the 1/29 = 0.034 chance floor. Feature
distributions shift so much with noise power that the models are learning
noise-level-specific decision boundaries rather than noise-invariant
representations. This is the clearest single diagnosis of the low-SNR failure,
and Recommendations 4–6 follow directly from it.

---

## 7. Verification — 19/19 checks passed

| check | result |
|---|---|
| train/val, train/test, val/test group overlap | 0 in all three |
| every group has exactly 5 SNR siblings, never split | pass |
| 29 classes balanced in every partition | 700 / 150 / 150 |
| 5 SNR levels balanced in every partition | 4060 / 870 / 870 |
| **permuted labels → chance** | **0.0358** (chance 0.0345) |
| **group-coherent label permutation → chance** | **0.0393** |
| **random features → chance** | **0.0315** |
| feature matrix finite, no constant/duplicate columns | pass |
| no column is a label proxy | pass |

The shuffle controls matter most: with labels permuted the same pipeline drops
from 0.646 to 0.036, so the reported scores reflect real class structure and not
leakage or a protocol artefact.

Noted but not failed: 3 feature pairs are exactly collinear
(`env_mean` = 2·`h1_mean`, `hf_perio_ratio` = `hf_perio_freq`/50,
`interharm_frac` = 1 − `harm_frac`).

---

## 8. Recommended changes, in priority order

### Tier 1 — attack the actual bottleneck (the 4 gated pairs)

These 8 classes cost 0.103 macro-F1 (0.689 → 0.792 when excluded). Section 5
shows the matched-filter ceiling is ~1.0 at ≥20 dB while the classifier is at
0.5–0.7, so this is recoverable, not fundamental.

**1. Replace the coherent envelope projection with a generalised likelihood
ratio test.** Current detectors read flicker off |S| magnitude envelopes, which
discards phase and squares the noise. Instead: estimate the sag support *û* and
depth *α̂*, form the residual r(t) = x(t) − x̂₍no-flicker₎(t), and project *r*
onto the 2-D subspace spanned by {H(t)·û(t)·sin(2π·ff·t),
H(t)·û(t)·cos(2π·ff·t)}, sweeping ff over 8–25 Hz. That is the actual GLRT for
this hypothesis pair and should recover most of the 30 dB of coherent gain the
ceiling analysis says is available. *Expected: gated-pair recall at 40 dB from
~0.5–0.7 to >0.9.*

**2. Lengthen the observation window from 10 to 20–30 cycles.** Evidence for
gated flicker grows as √(N · duty cycle), so this is worth +3 to +5 dB of
effective SNR, and it simultaneously fixes the modulation-domain resolution
(Δf 5 Hz → 1.7 Hz, currently only ~4 usable bins across the entire 8–25 Hz
band). One parameter change in `pqmodel`, and the cheapest structural
improvement available. *Expected: large, and it compounds with #1.*

**3. Two-stage hierarchical classifier.** Stage 1 predicts the 25-way collapsed
label — already at 0.975 at 40 dB. Stage 2 runs a dedicated binary
flicker/no-flicker discriminator per gated pair, trained only on those two
classes using the GLRT statistic from #1. Specialising removes the pressure of
27 irrelevant classes on a decision boundary driven by a 0.3% perturbation.

### Tier 2 — attack the low-SNR collapse

**4. Subtract the estimated noise floor from every band-energy feature.** At
0 dB the noise power equals the signal power, so raw band energies measure noise.
Use E_corrected = max(0, E − σ̂²·B) with σ̂² from the existing `noise_floor_db`
estimate. Cheap, and it makes the energy features approximately SNR-invariant.

**5. Train with continuously sampled SNR** — SNR ~ U(−5, 45) dB per waveform —
instead of 5 discrete levels. Section 6 shows the model memorises level-specific
boundaries and scores 0.051 at an unseen 0 dB. Continuous sampling removes the
+0.21 extrapolation gap and costs nothing extra to generate.

**6. Denoise before extraction.** S-domain or wavelet soft-thresholding, or a
Wiener filter built from the estimated noise PSD. At 10 and 0 dB this is worth
more than any classifier change. Pair it with test-time augmentation: extract at
several denoising strengths and average the predictions.

### Tier 3 — modelling

**7. Add a CNN on the S-matrix (or the raw waveform).** All four current models
consume the *same* 191 handcrafted features, so their errors are strongly
correlated — which is precisely why the ensemble gains only +0.008 over
LightGBM alone. A learned-representation model is the one addition that would be
genuinely decorrelated, and it is where the largest ensemble gain remains.

**8. Fix or drop the SVM.** Its fitted weight is 0.004: the search deleted it.
Either calibrate it properly (Platt or isotonic, fitted on OOF) instead of the
current softmax-over-margins, or replace it with ExtraTrees or a second MLP seed.

**9. Temperature-scale every model on OOF before averaging.** Soft voting
currently averages probability vectors with different sharpness, which is part of
why equal-weight voting underperformed.

**10. SNR-specialist sub-ensembles with a gating network** (design option B).
Now justified by evidence rather than assumption: the per-class error structure
differs qualitatively between 40 dB and 0 dB (see `fig2`), so specialists gated
on the measured `snr_est_db` should help most at 10 and 0 dB.

**11. Drop the 3 exactly-collinear features.** Harmless for trees, wasteful in
the PCA/SVM/MLP branch.

### Tier 4 — protocol

**12. Increase to ~1000 base waveforms per class.** 700 training rows per class
against 191 features is modest; the CV/test agreement suggests headroom rather
than overfitting.

**13. Repeat over 3–5 seeds** and report confidence intervals. Fold-to-fold
std is already ~0.011–0.015, so single-seed differences below ~0.02 should not
be interpreted.

---

## 9. Files

| file | purpose |
|---|---|
| `pqmodel.py` | NumPy port of `pqmodel.m`, all 29 classes + AWGN |
| `features.py` | S-transform, 191 features across 8 groups |
| `build_dataset.py` | resumable dataset build (`--step 0..4`, `--merge`) |
| `pipeline.py` | splits, 4 base models, 4 ensembles, evaluation (checkpointed) |
| `unseen_snr.py` | leave-one-SNR-out study |
| `verify.py` | 19 correctness controls |
| `make_figures.py` | the four figures |
| `test_pqmodel.py`, `test_features.py` | port and transform validation |
| `exp_flicker*.py` | the four flicker-detector experiments |
| `exp_degeneracy.py` | matched-filter ceiling analysis |
| `results/results.json`, `unseen_snr.json` | all metrics |
| `fig1..fig4*.png` | SNR degradation, class×SNR heatmap, confusions, importances |

Reproduce:
```bash
for k in 0 1 2 3 4; do python scripts/build_dataset.py --step $k --n-base 200; done
python scripts/build_dataset.py --merge
python scripts/run_pipeline.py --data /tmp/dataset.npz --out results/results.json --folds 10
python scripts/verify.py  &&  python scripts/make_figures.py
for k in 0 1 2 3 4; do python experiments/unseen_snr.py --only $k; done
python experiments/unseen_snr.py --merge
```
