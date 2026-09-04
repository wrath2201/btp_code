# Per-class x per-SNR evaluation

## Frozen-DASNet DualPQ  (mean of 5 seeds)

| metric | clean | 40dB | 30dB | 20dB | 10dB | 0dB | pooled |
|---|---|---|---|---|---|---|---|
| overall accuracy | 91.33 ± 1.33 | 91.77 ± 1.02 | 89.68 ± 0.89 | 82.53 ± 0.83 | 62.90 ± 1.29 | 28.46 ± 2.64 | **74.44** ± 1.01 |
| macro precision | 91.48 ± 1.37 | 91.83 ± 1.07 | 89.89 ± 0.85 | 82.92 ± 0.78 | 63.44 ± 1.42 | 29.41 ± 3.44 | **75.09** ± 1.21 |
| macro recall (= balanced acc) | 91.33 ± 1.33 | 91.77 ± 1.02 | 89.68 ± 0.89 | 82.53 ± 0.83 | 62.90 ± 1.29 | 28.46 ± 2.64 | **74.44** ± 1.01 |
| macro F1 | 91.18 ± 1.42 | 91.69 ± 1.05 | 89.56 ± 0.93 | 82.36 ± 0.95 | 62.36 ± 1.45 | 27.00 ± 2.71 | **74.46** ± 1.08 |
| overall kappa | 91.02 ± 1.37 | 91.48 ± 1.06 | 89.31 ± 0.92 | 81.90 ± 0.86 | 61.57 ± 1.33 | 25.90 ± 2.73 | **73.53** ± 1.04 |
| macro one-vs-rest accuracy | 99.40 ± 0.09 | 99.43 ± 0.07 | 99.29 ± 0.06 | 98.80 ± 0.06 | 97.44 ± 0.09 | 95.07 ± 0.18 | **98.24** ± 0.07 |

## Cross-model macro-F1 by SNR

| model | clean | 40dB | 30dB | 20dB | 10dB | 0dB | pooled |
|---|---|---|---|---|---|---|---|
| Frozen-DASNet DualPQ | 91.18 | 91.69 | 89.56 | 82.36 | 62.36 | 27.00 | **74.46** |

## Cross-model Cohen's kappa by SNR

| model | clean | 40dB | 30dB | 20dB | 10dB | 0dB | pooled |
|---|---|---|---|---|---|---|---|
| Frozen-DASNet DualPQ | 91.02 | 91.48 | 89.31 | 81.90 | 61.57 | 25.90 | **73.53** |

## Cross-model overall accuracy by SNR

| model | clean | 40dB | 30dB | 20dB | 10dB | 0dB | pooled |
|---|---|---|---|---|---|---|---|
| Frozen-DASNet DualPQ | 91.33 | 91.77 | 89.68 | 82.53 | 62.90 | 28.46 | **74.44** |
