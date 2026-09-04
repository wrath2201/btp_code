# RESEARCH_FROZEN_DUALPQ

> Working notes from the frozen-representation experiment. Sections 6, 8, 9 and
> the closing answer were revised after the pre-submission audit: the per-SNR
> standard deviations disagreed with `results/FINAL_RESULTS.md`, the variance
> claim overstated what the tests support, and the collapse was attributed to
> two seeds when only one collapsed. See `README.md` §7/§9 and
> `FINAL_SCIENTIFIC_AUDIT.md` Part 5 for the reconciled versions, and
> `docs/PRE_SUBMISSION_CHECKLIST.md` for what remains open.

## 1. Prediction-count consistency check
I previously verified that Seed 0 and Seed 3 both predict all 29 unique classes. Seed 3 did not 'drop' 5 classes, but rather suffered from extreme class bias (predicting some classes 600+ times and others only 25 times), which severely damaged its F1 score. This confirms the original failure was an optimization collapse (losing discriminative power), not a hardcoded bug.

*Not currently verifiable from this repository:* the Original DualPQ-D
prediction arrays (`results/multiseed/dualpq_concat_seed*_preds.npz`) are not
committed, so the prediction-count check above cannot be re-run by a reader.
Committing them is `docs/PRE_SUBMISSION_CHECKLIST.md` OPEN-2.

## 2. Exact Frozen-DualPQ architecture
The **Frozen-DASNet DualPQ** model uses the exact same architecture as the original DualPQ-D. However, the Deep Expert (DASNet with Learnable-DST) is loaded with the previously trained weights and completely **FROZEN** (requires_grad = False). The Classical Expert (MLP) and the Concatenation Fusion Head remain trainable.

The architecture is identical, but the **training configuration is not**:
`run_dualpq.py` applies waveform AWGN augmentation and mixed precision;
`run_frozen_dualpq.py` applies neither (AMP is disabled explicitly, and
`PQDataset` has no augment path). So this experiment does not isolate the
effect of freezing — three factors move together. `DualWaveDataset` also
re-noises the waveform while leaving `x_feat` at its original SNR, so the
joint run trains the fusion layer on mismatched (waveform, feature) pairs that
never occur at test time. The isolating run is OPEN-4.

## 3. Trainable/frozen parameter counts
- **Total Parameters:** 1,536,828
- **Frozen Parameters:** 1,290,783 (Deep Expert completely frozen)
- **Trainable Parameters:** 246,045 (Classical MLP + Fusion)

## 4. Seed-by-seed results
| Seed | Best Val F1 | Test F1 | Best Epoch |
|---|---|---|---|
| 0 | 0.7290 | 0.7255 | 15 |
| 1 | 0.7636 | 0.7477 | 0 |
| 2 | 0.7625 | 0.7523 | 5 |
| 3 | 0.7546 | 0.7477 | 6 |
| 4 | 0.7620 | 0.7497 | 2 |

## 5. Mean ± SD
**Frozen-DASNet DualPQ:** 74.46% ± 1.08% (Med: 74.77%, Min: 72.55%, Max: 75.23%, 95%CI: [73.10%, 75.83%])

## 6. Per-SNR results
| Model | Clean | 40dB | 30dB | 20dB | 10dB | 0dB | Overall |
|---|---|---|---|---|---|---|---|
| **Frozen-DASNet DualPQ** | 91.18±1.42 | 91.69±1.05 | 89.56±0.93 | 82.36±0.95 | 62.36±1.45 | 27.00±2.71 | 74.46±1.08 |

Standard deviations corrected to match `results/FINAL_RESULTS.md` and
`results/per_class_snr_frozen/`; the earlier row understated every one of them
(e.g. 0 dB ±2.4 against the actual ±2.71). Regenerate with
`python scripts/stats_tests.py`.

## 7. Original DualPQ vs Frozen DualPQ comparison
| Seed | Original DualPQ | Frozen DualPQ | Difference |
|---|---|---|---|
| 0 | 0.7267 | 0.7255 | -0.0011 |
| 1 | 0.7256 | 0.7477 | +0.0222 |
| 2 | 0.6243 | 0.7523 | +0.1281 |
| 3 | 0.3491 | 0.7477 | +0.3985 |
| 4 | 0.6561 | 0.7497 | +0.0936 |

**Original DualPQ-D:** 61.63% ± 15.58%
**Frozen-DASNet DualPQ:** 74.46% ± 1.08%

## 8. Whether variance decreased
The standard deviation went from **±15.58%** to **±1.08%** — a variance ratio
of 207.

How far that is supported:

| Test | *p* | Assumes normality? |
|---|---:|---|
| **Levene** | **0.138** | no — this is the one to report |
| Bartlett | 0.00018 | yes |
| F-ratio | 0.00014 | yes |

The only test robust to non-normality is not significant at n = 5, and
normality is violated by the single collapsed run. Excluding seed 3, Original
DualPQ-D is **68.31 ± 5.13**.

**One seed collapsed, not two.** The per-seed values are 72.67, 72.56, 62.43,
**34.91**, 65.61 — seed 3 is the collapse; seed 2 (62.43) and seed 4 (65.61)
are weak runs, not collapses. The table in §7 above shows this directly. An
earlier version of this section said "Seeds 2 and 3", which the data does not
support.

So: report the SD drop descriptively, note the confound from §2, and quote
Levene alongside. Do not write that the variance was eliminated.

## 9. Does the experiment support the optimization-instability hypothesis?
It is **consistent with** it, and does not establish it.

Consistent: freezing the deep branch produced five runs within 2.7 pp of each
other where joint training produced one collapse and two weak runs.

Not established, for three reasons:

1. Freezing, augmentation and mixed precision all changed at once (§2), so the
   contribution of freezing is not identified.
2. The joint run trains on mismatched (waveform, feature) pairs (§2) — a
   defect in the baseline that could produce the instability on its own,
   independently of any optimization-dynamics argument.
3. Stage-1 quality does not predict the stage-2 outcome once the collapsed
   seed-0 DASNet run is excluded: *r* falls from 0.987 to 0.464 (*p* = 0.54).
   If joint optimization of the deep branch were the mechanism, some coupling
   would be expected across the normal operating range.

There are also no gradient logs, so any claim about "gradient conflict"
specifically is unsupported.

## 10. Recommended next experiment
Two, in order:

1. **Original DualPQ-D with `--no-aug` and AMP disabled, 5 seeds** (OPEN-4,
   ~6 h GPU). Isolates freezing from the augmentation and precision changes.
   If the gap largely closes, the finding becomes *"the augmentation/feature
   mismatch in naive joint fusion is what destabilises it"* — a sharper and
   more useful result than the current framing.
2. **Classical-branch-only ablation** (OPEN-3, ~20 min). Stage-2 validation
   peaks at epoch 0, 2, 5, 6 and 15 across the five seeds — seed 1's selected
   model precedes any training — which is consistent with the trainable
   classical MLP carrying most of the load. Nothing currently separates its
   contribution from the frozen deep branch's.

An alternate fusion mechanism (e.g. cross-attention) is worth exploring, but
only after (2) establishes that the deep branch contributes at all.

---
### Did freezing the experts make DualPQ reliably reproducible?
**In this configuration, yes — but the cause is not established.** All five
seeds converged within 2.7 pp of each other (74.46 ± 1.08) where joint
training gave 61.63 ± 15.58 with one collapse. That is a real and useful
difference in behaviour.

What it does not yet show is *why*: the two configurations differ in three
respects, the joint baseline has a training-signal defect of its own, and the
formal variance test that survives non-normality is not significant at n = 5.
The complementarity of the two representations is likewise not established
while OPEN-3 is outstanding — no run isolates the classical branch, so the
deep branch's contribution to the 74.46 is unquantified.
