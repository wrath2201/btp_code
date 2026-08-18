# 29-Class Power-Quality Disturbance Classification

A soft-voting ensemble over Stockwell-transform, time-domain and
frequency-domain features, evaluated under a strict leakage-free protocol across
five AWGN levels plus noise-free data.

Dataset generator: `pqmodel.m` — Igual, Medrano, Arcega & Mantescu,
*Integral mathematical model of power quality disturbances* (2017),
ported to NumPy as `pqmodel.py`.

---

## Headline results

Test macro-F1, weighted soft vote, waveform-level (leakage-free) split:

| | clean | 40 dB | 30 dB | 20 dB | 10 dB | 0 dB |
|---|---|---|---|---|---|---|
| all 29 classes | 0.886 | 0.877 | 0.869 | 0.807 | 0.628 | 0.259 |
| 21 classes (excl. degenerate pairs) | **0.990** | 0.983 | 0.961 | 0.922 | 0.709 | 0.335 |

Two findings drive the interpretation:

**1. The S-transform is blind to flicker at the fundamental.** Its window at
frequency *f* has σ_t = 1/*f*, hence σ_f = *f*/2π ≈ 8 Hz at 50 Hz — so the
fundamental row low-passes away the 8–25 Hz flicker band before any detector
sees it. Reading flicker off that row, the standard construction, measures
something the transform already destroyed. Fixed by detecting the sag/swell
event on the (flicker-free) S-envelope and measuring flicker coherently on the
full-bandwidth Hilbert envelope outside it.

**2. Four class pairs are near-degenerate by construction.** In classes
20/21/28/29 the flicker factor multiplies *only* the sag-gated harmonic term,
which is zero outside the event — so outside the sag, classes 15 and 20 are
identical signals. Measured evidence: ‖δ‖ = 0.013 vs 0.037 for the pairs that
work, i.e. ~8× less signal power. After the flicker fix, these four pairs are
the *only* remaining confusions at 40 dB.

**3. A naive train/test split inflates the score by up to +0.127.** Each
waveform appears once per noise level; splitting by row instead of by waveform
lets a waveform's clean copy sit in training while its noisy twin is in test.
Measured: 0.866 → 0.993 on clean data, with 99.9% of "unseen" test waveforms
already seen. The published figure for this generator is 99.41%.

Both ensemble claims are significant over 5 paired splits: weighted voting beats
the best single model (+0.0089, 5/5 wins, p = 0.004) while **equal-weight voting
is significantly worse** (−0.0084, 0/5, p = 0.016).

---

## Quick start

```bash
pip install -r requirements.txt

# build the dataset (~10 min; steps 0-4 are noisy levels, 5 is noise-free)
python build_dataset.py --step 0 --n-base 200 --shard-dir data/shards
python build_dataset.py --step 1 --n-base 200 --shard-dir data/shards
python build_dataset.py --step 2 --n-base 200 --shard-dir data/shards
python build_dataset.py --step 3 --n-base 200 --shard-dir data/shards
python build_dataset.py --step 4 --n-base 200 --shard-dir data/shards
python build_dataset.py --merge --steps 0 1 2 3 4 --shard-dir data/shards --out data/dataset.npz

# train and evaluate (~25 min)
python pipeline.py --data data/dataset.npz --out results.json --folds 10

# checks, figures, audits
python verify.py --data data/dataset.npz          # 19 correctness controls
python make_figures.py
python audit_leakage.py --data data/dataset.npz   # the leakage measurement
python multiseed.py --mode split --seeds 0 1 2 3 4
```

`--steps 0 1 2 3 4` on the merge is not optional: without it the merge sweeps up
every shard present, including the noise-free one, and you silently train on six
levels while believing you used five.

Low on RAM? Add `--max-new-folds 1` to `pipeline.py` and run it ten times; it
checkpoints after every fold.

---

## Documentation

| File | For |
|---|---|
| `START_HERE.md` | Plain-English walkthrough, no programming assumed |
| `REPORT.md` | Full results, both findings, prioritised recommendations |
| `REPLICATION_GUIDE.md` | Independent replication with expected values and tolerances |
| `Colab_Run.ipynb` | Run the whole pipeline on Google Colab |

## Code map

| File | Role |
|---|---|
| `pqmodel.py` | NumPy port of the 29-class generator + AWGN |
| `features.py` | S-transform and 191 features across 8 groups |
| `build_dataset.py` | Resumable dataset build (`--step 0..5`, `--merge`) |
| `pipeline.py` | Splits, 4 base models, 4 ensembles, evaluation |
| `verify.py` | 19 correctness controls incl. label-shuffle |
| `audit_leakage.py` | Group-split vs row-split inflation |
| `multiseed.py` | Confidence intervals and paired significance tests |
| `unseen_snr.py` | Leave-one-SNR-out robustness |
| `make_figures.py` | The four result figures |
| `test_pqmodel.py`, `test_features.py` | Validation against analytic ground truth |
| `exp_flicker*.py`, `exp_degeneracy.py` | The investigation behind findings 1 and 2 |

Generated data is not committed — `build_dataset.py` reproduces it
deterministically from a fixed seed in about ten minutes.

---

## Licence and citation

⚠ **`pqmodel.py` is a port of `pqmodel.m`, which is licensed GPL-3.0.** It is
therefore a derivative work, and the modules that import it are affected.

GPL obligations attach on *distribution*, so a private repository triggers
nothing. **But if you make this repository public, publish the code, or share it
outside your institution, the project must be released under GPL-3.0** and
include the original copyright notice. Decide this before flipping the
repository to public.

The original authors also ask that the paper be cited by anyone who uses or
modifies the model:

> R. Igual, C. Medrano, F. J. Arcega, G. Mantescu, "Integral mathematical model
> of power quality disturbances", *18th International Conference on Harmonics and
> Quality of Power (ICHQP)*, 2018.

Original model and data: https://data.mendeley.com/datasets/6kmkk9bjdx/1
