# results/per_class

## `summary.json` is off by one class for two of the five models

For `MGCNN-SDTransformer` and `Fixed-Frozen-DASNet`, position *j* of the
`accuracy`, `precision`, `recall` and `f1` arrays holds the values for **class
*j* + 2**, not class *j* + 1.

Verified cell-for-cell against the reconstructed MGCNN grid in
`../per_class_snr/mgcnn_sdtransformer/`: the implied index shift is exactly +1
at all 28 populated positions, with zero residual.

Symptoms visible in the file itself:

- class 1 (*Pure sinusoidal*) is **absent** for those two models;
- the 29th entry is a phantom with `precision = recall = f1 = 0.0` and
  `accuracy = 1.0`.

Cause: `scripts/run_frozen_dualpq.py` and `scripts/run_mgcnn_sdtransformer.py`
emitted labels in 0..28 (`y = d_feat["y"] - 1`), while `src/pipeline.py` and
`scripts/run_dasnet.py` emit 1..29 (`argmax + 1`). The aggregation that produced
this file applied a single convention to all five models.

`DualPQ`, `DASNet` and `Baseline` in this file are unaffected. The `macro_f1`
and `kappa` scalars are unaffected for every model, because they were computed
inside each training script with a self-consistent label set — a macro average
over a shifted-but-consistent labelling is numerically identical.

The per-class *names* quoted in `FINAL_SCIENTIFIC_AUDIT.md` Part 4 are correct;
whoever wrote that table mapped the positions properly. It is the stored arrays
that are unsafe to index.

### Status

The emitting scripts are fixed: both now write 1..29, matching the rest of the
repository. `src/metrics_perclass.normalise_labels` additionally accepts either
convention and rejects anything else, so the mismatch cannot recur silently.

The arrays in `summary.json` **cannot be fully repaired without rerunning
MGCNN-SDTransformer and Frozen-DASNet DualPQ**, because the class-1 values were
never written. Do not consume this file by index. Use
`../per_class_snr/<model>/metrics.json` instead, which is generated from
prediction arrays with an asserted label convention, or regenerate this file
after a rerun.

Also note that `summary.json` carries no per-SNR breakdown — it pools all six
noise conditions. `../per_class_snr/` is the per-SNR version.
