# All-model per-SNR metrics

Assembled from the committed `metrics.json` files. Mean ± sample SD (ddof=1) over seeds.

> MGCNN-SDTransformer seeds 1–4 used `split_seed == seed`; every other model used
> `split_seed = 0`. Its spread mixes split variance with training variance and its row is
> not paired with the others'.

## Macro-F1 (%)

| Model | Seeds | clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB | Pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical Ensemble (weighted_vote) | 1 | 87.73 | 87.32 | 87.55 | 79.63 | 61.34 | 25.83 | **71.70** |
| Classical Ensemble (geometric_vote) | 1 | 87.50 | 87.16 | 86.95 | 79.85 | 62.35 | 25.15 | **71.63** |
| DASNet | 5 | 84.86 ± 14.38 | 90.16 ± 3.30 | 87.18 ± 5.13 | 75.85 ± 12.32 | 51.65 ± 19.48 | 20.69 ± 9.95 | **69.73** ± 9.12 |
| MGCNN-SDTransformer | 5 | 81.62 ± 1.57 | 81.58 ± 1.38 | 80.79 ± 0.99 | 76.16 ± 0.99 | 55.87 ± 0.68 | 22.45 ± 0.87 | **66.59** ± 0.98 |
| Frozen-DASNet DualPQ | 5 | 91.18 ± 1.42 | 91.69 ± 1.05 | 89.56 ± 0.93 | 82.36 ± 0.95 | 62.36 ± 1.45 | 27.00 ± 2.71 | **74.46** ± 1.08 |

## Cohen's kappa (%)

| Model | Seeds | clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB | Pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical Ensemble (weighted_vote) | 1 | 87.50 | 87.02 | 87.26 | 79.17 | 60.24 | 23.81 | **70.83** |
| Classical Ensemble (geometric_vote) | 1 | 87.26 | 86.90 | 86.67 | 79.29 | 61.07 | 22.98 | **70.69** |
| DASNet | 5 | 85.05 ± 13.51 | 89.98 ± 3.24 | 87.26 ± 4.36 | 75.79 ± 11.57 | 51.60 ± 18.30 | 20.36 ± 9.99 | **68.34** ± 10.11 |
| MGCNN-SDTransformer | 5 | 81.93 ± 1.02 | 81.95 ± 0.84 | 81.19 ± 0.75 | 76.17 ± 0.93 | 55.00 ± 0.72 | 20.29 ± 0.85 | **66.09** ± 0.67 |
| Frozen-DASNet DualPQ | 5 | 91.02 ± 1.37 | 91.48 ± 1.06 | 89.31 ± 0.92 | 81.90 ± 0.86 | 61.57 ± 1.33 | 25.90 ± 2.73 | **73.53** ± 1.04 |

## Overall accuracy (%)

| Model | Seeds | clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB | Pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical Ensemble (weighted_vote) | 1 | 87.93 | 87.47 | 87.70 | 79.89 | 61.61 | 26.44 | **71.84** |
| Classical Ensemble (geometric_vote) | 1 | 87.70 | 87.36 | 87.13 | 80.00 | 62.41 | 25.63 | **71.70** |
| DASNet | 5 | 85.56 ± 13.04 | 90.32 ± 3.13 | 87.70 ± 4.21 | 76.62 ± 11.17 | 53.26 ± 17.67 | 23.10 ± 9.65 | **69.43** ± 9.76 |
| MGCNN-SDTransformer | 5 | 82.55 ± 0.98 | 82.57 ± 0.81 | 81.84 ± 0.72 | 76.99 ± 0.90 | 56.55 ± 0.69 | 23.03 ± 0.82 | **67.26** ± 0.64 |
| Frozen-DASNet DualPQ | 5 | 91.33 ± 1.33 | 91.77 ± 1.02 | 89.68 ± 0.89 | 82.53 ± 0.83 | 62.90 ± 1.29 | 28.46 ± 2.64 | **74.44** ± 1.01 |

## Macro precision (%)

| Model | Seeds | clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB | Pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical Ensemble (weighted_vote) | 1 | 87.73 | 87.34 | 87.63 | 79.78 | 62.32 | 25.88 | **71.86** |
| Classical Ensemble (geometric_vote) | 1 | 87.67 | 87.34 | 87.21 | 80.18 | 63.70 | 25.35 | **71.93** |
| DASNet | 5 | 85.76 ± 12.91 | 90.75 ± 2.39 | 88.14 ± 3.65 | 77.20 ± 10.44 | 53.67 ± 18.60 | 24.13 ± 9.88 | **73.16** ± 3.77 |
| MGCNN-SDTransformer | 5 | 83.39 ± 0.61 | 83.77 ± 1.02 | 82.88 ± 0.83 | 77.07 ± 1.23 | 56.79 ± 0.88 | 23.22 ± 0.53 | **67.05** ± 0.91 |
| Frozen-DASNet DualPQ | 5 | 91.48 ± 1.37 | 91.83 ± 1.07 | 89.89 ± 0.85 | 82.92 ± 0.78 | 63.44 ± 1.42 | 29.41 ± 3.44 | **75.09** ± 1.21 |

Macro recall equals overall accuracy exactly (the test set is balanced at 30 rows per
class per SNR level), so it is omitted.

## Where the deep branch actually helps

Pooled per-class F1, Frozen-DASNet DualPQ vs Classical Ensemble (weighted_vote).

| Class | Classical | Frozen | Δ (pp) |
|---|---:|---:|---:|
| 22. Sag + Harm + OT | 46.3 | 61.1 | +14.8 |
| 28. Sag + Harm + Flicker + OT | 44.4 | 55.9 | +11.5 |
| 20. Sag + Harm + Flicker | 49.5 | 60.4 | +10.8 |
| 29. Swell + Harm + Flicker + OT | 46.2 | 56.1 | +9.9 |
| 15. Sag + Harmonics | 50.0 | 56.3 | +6.3 |
| 23. Swell + Harm + OT | 49.0 | 53.9 | +4.9 |
| 21. Swell + Harm + Flicker | 52.2 | 57.0 | +4.8 |
| 26. Harm + Sag + Flicker + OT | 74.6 | 79.2 | +4.6 |
| 19. Harm + Swell + Flicker | 79.4 | 82.7 | +3.3 |
| 3. Swell | 73.5 | 76.6 | +3.1 |
| 16. Swell + Harmonics | 51.3 | 54.3 | +3.0 |
| 18. Harm + Sag + Flicker | 81.0 | 83.8 | +2.8 |
| 6. Oscillatory transient | 83.6 | 86.3 | +2.8 |
| 14. Swell + Osc. transient | 77.2 | 79.8 | +2.5 |
| 13. Sag + Osc. transient | 80.8 | 83.3 | +2.5 |
| 24. Harm + Sag + OT | 78.8 | 81.1 | +2.3 |
| 4. Interruption | 92.1 | 93.8 | +1.7 |
| 27. Harm + Swell + Flicker + OT | 79.9 | 81.0 | +1.1 |
| 7. Harmonics | 93.3 | 94.3 | +1.1 |
| 9. Harmonics + Swell | 81.4 | 82.3 | +0.9 |
| 8. Harmonics + Sag | 82.6 | 82.9 | +0.3 |
| 10. Flicker | 86.1 | 86.3 | +0.2 |
| 25. Harm + Swell + OT | 83.5 | 83.5 | -0.0 |
| 2. Sag | 77.5 | 77.0 | -0.5 |
| 1. Pure sinusoidal | 64.0 | 62.8 | -1.3 |
| 12. Flicker + Swell | 82.6 | 80.6 | -2.0 |
| 5. Impulsive transient | 67.5 | 64.7 | -2.9 |
| 17. Notch | 84.8 | 81.5 | -3.3 |
| 11. Flicker + Sag | 86.0 | 80.8 | -5.1 |

Frozen is worse on 7 of 29 classes; mean delta +2.76 pp.

