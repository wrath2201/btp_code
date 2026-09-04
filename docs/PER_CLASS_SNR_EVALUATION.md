# Per-class × per-SNR evaluation

`results/FINAL_RESULTS.md` reports macro-F1 per model and per noise level. This
adds the layer below it: for every one of the 29 classes, at every one of the
six noise conditions, the **accuracy, precision, recall, F1 and Cohen's kappa**,
plus the aggregate macro-F1 / kappa / accuracy those cells roll up into.

Motivation: a macro-F1 of 71.5 at 0 dB-to-clean tells you the benchmark is hard;
it does not tell you *which* disturbances a model cannot separate, whether a
model fails by over-predicting a class or by missing it, or how many classes have
collapsed entirely at a given SNR. Those are the questions a per-class grid
answers, and they are the ones a reviewer asks.

## What was missing before

`results/results.json` (and each `baseline_seed*.json`) stores
`per_class_per_snr_recall` — a 29 × 6 recall grid, which is what
`figures/fig2_class_snr_heatmap.png` plots. It stores full confusion matrices
only for the two extreme levels (`confusion_snr999`, `confusion_snr0`), because
`src/pipeline.py` loops `(LV[0], LV[-1])`. The deep and hybrid scripts store
less: `run_dasnet.py` keeps only `confusion_all`, and
`run_frozen_dualpq.py` / `run_dualpq.py` / `run_mgcnn_sdtransformer.py` keep no
confusion matrix at all. `*_preds.npz` was gitignored.

Consequence: precision, F1 and kappa **could not be computed per class per SNR
for any model** from the committed artefacts, and for four of the five models
not even at a single SNR level.

## What was added

| File | Purpose |
|---|---|
| `src/metrics_perclass.py` | The metric definitions. Everything is derived from the per-SNR confusion matrix, so all five families are mutually consistent by construction. Validated cell-for-cell against `sklearn` (`precision_score`, `recall_score`, `f1_score`, `cohen_kappa_score`, `accuracy_score`) on random data, and against this repo's own committed confusion matrices. |
| `scripts/reconstruct_preds.py` | Regenerates the test-set prediction arrays from committed checkpoints (DASNet, MGCNN-SDTransformer) or by refitting (Classical Ensemble). |
| `scripts/eval_per_class_snr.py` | Consumes prediction arrays and writes the metrics, the CSVs and the heatmaps. |
| `scripts/repro_frozen_dualpq.py` | Reproduces Frozen-DASNet DualPQ, whose weights were never saved (see *Known gaps*). |
| `results/preds/*.npz` | The reconstructed predictions (`yte`, `yp`, `ste`), 68 KB total. |
| `results/per_class_snr/` | Per-model heatmaps (5 metrics × 4 classifiers), `per_class_per_snr.csv`, `metrics.json`, `summary.md`, and cross-model comparison plots. |

## Metric definitions

For a fixed noise level *s*, build the 29 × 29 confusion matrix `C` over the
test rows with `snr == s`, then read one-vs-rest 2 × 2 tables off its rows and
columns. With `N` = rows at that level:

```
TP = C[c,c]                  FP = C[:,c].sum() - TP
FN = C[c,:].sum() - TP       TN = N - TP - FP - FN

accuracy_c  = (TP + TN) / N                            one-vs-rest accuracy
precision_c = TP / (TP + FP)
recall_c    = TP / (TP + FN)
f1_c        = 2 P R / (P + R)
kappa_c     = 2 (TP·TN - FN·FP)
              / ((TP+FP)(FP+TN) + (TP+FN)(FN+TN))      Cohen's kappa, 2x2
```

Aggregates at level *s*: `macro_f1 = mean_c f1_c` (the benchmark's primary
metric), `macro_precision`, `macro_recall`, `overall_accuracy = trace(C)/N`, and
multiclass `overall_kappa` on `C`.

Two notes on interpreting these:

- The test set is exactly class-balanced (**30 rows per class per SNR level**),
  so `macro_recall`, `balanced_accuracy` and `overall_accuracy` are numerically
  identical. Reporting all three is not three pieces of evidence.
- `overall_kappa` tracks `macro_f1` to within 0.3 pp down to 30 dB and then
  diverges — +0.5 pp at 20 dB, +1.1 pp at 10 dB, +2.0 pp at 0 dB — as chance
  agreement stops being negligible. On balanced data kappa is close to a
  monotone rescaling of accuracy; it is included because it is expected in
  power-systems venues, not because it is independent.
- One-vs-rest accuracy is dominated by true negatives with 29 balanced classes
  (every model scores 94–99% on it at every SNR). It is reported for
  completeness and should not be quoted as a headline number.

## Reproducing

```bash
# 1. data (deterministic; seed 20260807)
for k in 0 1 2 3 4 5; do python scripts/build_dataset.py --step $k --n-base 200; done
python scripts/build_dataset.py --merge
python scripts/build_waveforms.py --n-base 200

# 2. test-set predictions
python scripts/reconstruct_preds.py classical --seed 0 --split-seed 0
for k in 0 1 2 3 4; do python scripts/reconstruct_preds.py dasnet --seed $k --split-seed 0; done
for k in 0 1 2 3 4; do python scripts/reconstruct_preds.py mgcnn  --seed $k; done   # split-seed follows seed

# 3. metrics + heatmaps
python scripts/eval_per_class_snr.py \
  --model "Classical Ensemble (weighted_vote)=results/preds/classical_weighted_vote_seed*_preds.npz" \
  --model "Classical Ensemble (geometric_vote)=results/preds/classical_geometric_vote_seed*_preds.npz" \
  --model "DASNet=results/preds/dasnet_seed*_preds.npz" \
  --model "MGCNN-SDTransformer=results/preds/mgcnn_seed*_preds.npz"
```

### Reconstruction fidelity

| Model | Route | Agreement with the published numbers |
|---|---|---|
| MGCNN-SDTransformer, 5 seeds | committed `.pt` → forward pass | **exact** — every seed and every per-SNR cell to 4 dp |
| DASNet, 5 seeds | committed `.pt` → forward pass | **±0.05 pp** (fp32 CPU here vs the fp16 autocast used at publication) |
| Classical Ensemble, seed 0 | full refit, `fast=False` | **−0.42 pp** pooled. SVM exact, LightGBM +0.09, RF −0.58, MLP −0.77 — library drift (scikit-learn 1.8 vs the ≥1.4 in `requirements.txt`) |

The two committed confusion matrices (`confusion_snr999`, `confusion_snr0` in
`results/results.json`) reproduce the published macro-F1 for those levels
exactly (88.59 and 25.88), which is the independent check on the metric code.

## Known gaps

1. **Frozen-DASNet DualPQ and Original DualPQ-D cannot be reconstructed.**
   `run_frozen_dualpq.py` and `run_dualpq.py` never call `torch.save`, and
   `*_preds.npz` was gitignored, so neither the weights nor the predictions of
   the final proposed model survive. `scripts/repro_frozen_dualpq.py` retrains
   stage 2 from the committed stage-1 DASNet checkpoints; because the deep
   expert is frozen, kept in `eval()` mode, and stage 2 applies no waveform
   augmentation, `z_deep` is constant per row and is cached once, after which
   the head trains in seconds. CPU RNG differs from the CUDA RNG of the
   published runs, so this reproduces the *distribution* of outcomes, not the
   published value to the decimal. Adding one `torch.save` to
   `run_frozen_dualpq.py` would remove this gap for future runs.

2. **`results/per_class/summary.json` is off by one class for two models.**
   Position *j* of the `accuracy` / `precision` / `recall` / `f1` arrays for
   `MGCNN-SDTransformer` and `Fixed-Frozen-DASNet` holds the values for class
   *j + 2*, not *j + 1*. Verified cell-for-cell against the reconstructed MGCNN
   grid: the implied index shift is exactly +1 at all 28 populated positions.
   Cause: `run_frozen_dualpq.py` and `run_mgcnn_sdtransformer.py` emit labels in
   0..28 (`y = d_feat["y"] - 1`) while `src/pipeline.py` and
   `scripts/run_dasnet.py` emit 1..29 (`argmax + 1`); the aggregation applied one
   convention to all five models. Symptoms: class 1 (*Pure sinusoidal*) is
   absent for those two models, and the 29th entry is a phantom with
   `precision = recall = f1 = 0` and `accuracy = 1.0`. The `macro_f1` and
   `kappa` scalars in that file are computed separately and are unaffected. The
   label convention is fixed in `src/metrics_perclass.normalise_labels`, which
   accepts either convention and rejects anything else; the emitting scripts are
   fixed separately. The stored arrays cannot be fully repaired without
   rerunning those two models, because the class-1 values were never written.

3. **Per-cell precision is limited by support.** With 30 test samples per class
   per SNR level, one misclassification moves a cell by 3.33 pp and the binomial
   standard error at *p* ≈ 0.5 is roughly ±9 pp. The two decimals printed in the
   heatmaps are not all significant. Row and column aggregates and the
   five-seed means are the trustworthy readings; a claim resting on an
   individual cell needs more base waveforms per class.
