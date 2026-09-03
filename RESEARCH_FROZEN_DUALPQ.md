# RESEARCH_FROZEN_DUALPQ

## 1. Prediction-count consistency check
I previously verified that Seed 0 and Seed 3 both predict all 29 unique classes. Seed 3 did not 'drop' 5 classes, but rather suffered from extreme class bias (predicting some classes 600+ times and others only 25 times), which severely damaged its F1 score. This confirms the original failure was an optimization collapse (losing discriminative power), not a hardcoded bug.

## 2. Exact Frozen-DualPQ architecture
The **Frozen-DASNet DualPQ** model uses the exact same architecture as the original DualPQ-D. However, the Deep Expert (DASNet with Learnable-DST) is loaded with the previously trained weights and completely **FROZEN** (requires_grad = False). The Classical Expert (MLP) and the Concatenation Fusion Head remain trainable. This tests the hypothesis of whether end-to-end updating of the DST/CNN branch contributes to the instability.

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
| **Frozen-DASNet DualPQ** | 91.2±1.3 | 91.7±0.9 | 89.6±0.8 | 82.4±0.9 | 62.4±1.3 | 27.0±2.4 | 74.46 |

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
The standard deviation went from **±15.58%** to **±1.08%**.
Variance was massively reduced, eliminating the catastrophic optimization collapses seen in Seeds 2 and 3 of the original run.

## 9. Does the experiment support the optimization-instability hypothesis?
Yes. These results are consistent with the hypothesis that end-to-end joint optimization of the Deep/DST branch contributes to the observed instability. By freezing the Deep branch, the catastrophic failures disappeared.

## 10. Recommended next experiment
Since the representations themselves are fundamentally complementary (as evidenced by the stable high performance here), but end-to-end training destabilizes them, the next logical step is to explore the precise optimization dynamics on the original end-to-end model. We hypothesize that gradient conflicts or similar optimization issues may occur, which requires further investigation, or one could explore an alternate fusion mechanism (like Transformer cross-attention in MGCNN) that naturally regulates the gradient flow.

---
### Did freezing the experts make DualPQ reliably reproducible?
**YES.** By freezing the DASNet branch, the massive seed-to-seed variance (±15.58%) was eliminated, and all 5 seeds successfully converged to a stable, high Macro-F1. This provides evidence that the representations are complementary and the previous catastrophic failures were caused by the difficult optimization dynamics of training the deep branch jointly with the classical MLP.
