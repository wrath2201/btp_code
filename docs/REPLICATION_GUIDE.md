# Replication guide
## Reproducing the 29-class PQ ensemble, and establishing what the number really is

Target: Python, using the code in this folder, with fresh runs.
Total compute: **~50 minutes** for the core path, ~90 with all audits.

> [!WARNING]
> **Historical Baseline Investigation Document**
> This repository performs an extensive multi-seed evaluation of PQ classification across 29 classes and 6 evaluation conditions. Following this guide verbatim will exactly reproduce the findings reported in the scientific audit.
> - These historical numerical checkpoints are intentionally preserved for transparency.
> - The old 5-SNR / 29,000-row setup belongs to an earlier experiment and is NOT the final methodology.
> - The final authoritative benchmark (6 SNRs, robust multi-seed evaluation) uses the current methodology documented in `README.md` and `PUBLICATION_AUDIT.md`.
> - Readers should use `README.md` and `PUBLICATION_AUDIT.md` for the authoritative final methodology and results.

---

## 0. Read this first — what a re-run can and cannot prove

You asked what separates "clean-room" from "re-run my code". It matters for how
you read your results:

| | tests | catches |
|---|---|---|
| Re-running this code (what you're doing) | **reproducibility** | environment drift, version breakage, nondeterminism |
| Clean-room reimplementation | **correctness** | my bugs |

A re-run reproduces a bug perfectly. So Stages 3–4 below tell you the pipeline
*runs the same*, not that it is *right*.

**But three of the stages here are genuine validation, not replay**, because
they are new experiments whose answers I could not have tuned:

- **Stage 2** checks the S-transform and generator against *analytic ground
  truth* (a pure sinusoid must give |S| = A/2 at a known bin). If my
  implementation were wrong, this fails — regardless of whose code runs it.
- **Stage 5** (leakage audit) produces a number that appears nowhere in the
  original work.
- **Stage 6** (multi-seed) can overturn the headline claim that the ensemble
  beats the best single model. It partly does — see §6.

So: Stages 3–4 build understanding; Stages 2, 5, 6, 7 build evidence.

---

## 1. Setup

The dependency minimums are declared in `requirements.txt`. The environments that were *actually tested* during development and validation are:
- Linux: Python 3.10.12, CUDA 12.6
- Windows: Python 3.12, CUDA 12.6

For deep-learning models (DASNet, MGCNN, DualPQ), you must also install PyTorch as specified in `requirements-deep.txt`:
```bash
pip install -r requirements.txt
pip install -r requirements-deep.txt
```

### ⚠ Delete the checkpoints before you start

This is the one mistake that will silently fake your whole replication. The
pipeline resumes from checkpoints, so if these files exist it will **skip all
the work and re-emit my results**:

```bash
cd pq_ensemble
rm -f results_oof_ckpt.npz results_base_ckpt.npz results/results.json results_preds.npz
rm -rf /tmp/shards /tmp/dataset.npz
```

If you'd rather keep mine for comparison, rename them instead and use
`--out myresults/results.json` throughout.

### On determinism

Everything is seeded: `np.random.default_rng` (PCG64, bit-reproducible across
platforms), and `random_state` on every sklearn/LightGBM estimator. You should
match me to ~4 decimals. LightGBM's multithreaded histogram build is not
strictly deterministic, so **treat differences below ±0.005 as agreement**;
anything larger means something real has changed.

---

## 2. Stage 1 — validate the primitives (2 min) ← *real validation*

```bash
python tests/test_pqmodel.py
python tests/test_features.py
```

These check against analytic truth, not against my outputs. The ones that matter:

| check | must give |
|---|---|
| class 1 is a pure 50 Hz sinusoid | peak amplitude 1.000, FFT bin 10 |
| class 7 harmonic amplitudes | 3rd/5th/7th each in [0.05, 0.15] |
| class 4 interruption depth | reaches ≤ 0.15 pu |
| AWGN calibration | achieved SNR within 0.5 dB of target at all 5 levels |
| **S-transform of a pure sinusoid** | **\|S\| = 0.5000 = A/2, std < 1e-3, at bin 10** |
| S-transform of 50 Hz + 0.3×250 Hz | 0.500 and 0.150 respectively |
| S-transform on a 50% sag | envelope = 0.502 deep inside, 0.974 outside |

The amplitude normalisation (|S| = A/2) is the single most useful check — it's
the property most S-transform implementations get wrong, and it's why the
feature scales are physically interpretable here.

Expect `test_features.py` to end with 191 features, ~15 ms/signal, `all finite:
True`.

---

## 3. Stage 2 — build the dataset (~8 min)

29 classes × 200 base waveforms × 6 evaluation conditions = 34,800 rows.

```bash
for k in 0 1 2 3 4 5; do python scripts/build_dataset.py --step $k --n-base 200; done
python scripts/build_dataset.py --merge
```

One SNR level per step, ~1.5 min each. Each prints its achieved SNR — all five
must read within 0.01 dB of target:

```
SNR 40 dB (achieved 40.00 dB): (5800, 191) in 1.5 min -> /tmp/shards/shard_0.npz
...
merged -> /tmp/dataset.npz   X=(34800, 191)  groups=5800  classes=29  SNRs=[0, 10, 20, 30, 40, 999]
```

*Why split into steps:* the whole build in one process gets OOM-killed on a
small machine. Steps are independent and resumable — if one dies, re-run just
that `k`.

---

## 4. Stage 3 — run the pipeline (~25 min)

The 10-fold OOF stage is the expensive part (~2 min per fold). It checkpoints
after every fold, so you can run it in pieces:

```bash
# one fold at a time (safe on any machine)
for i in $(seq 1 10); do
  python scripts/run_pipeline.py --data /tmp/dataset.npz --out results/results.json --folds 10 --max-new-folds 1
done

# then the final fit + ensembles
python scripts/run_pipeline.py --data /tmp/dataset.npz --out results/results.json --folds 10
```

On a machine with plenty of RAM you can just run the last line and let it do all
10 folds in one go.

### Checkpoints you should hit

**Split geometry** — must be exact, not approximate:
```
[split] groups  train=4060  val=870  test=870
[split] rows    train=20300  val=4350  test=4350
         train class counts 700-700 (balanced=True), per-SNR [4060 ×5]
[split] group overlap between partitions: 0 (asserted)
```

**10-fold CV macro-F1** (tolerance ±0.01):

| model | expected |
|---|---|
| rf | 0.6476 ± 0.0152 |
| lgbm | 0.6581 ± 0.0109 |
| svm | 0.5681 ± 0.0113 |
| mlp | 0.6346 ± 0.0106 |

**Final test macro-F1** (tolerance ±0.005):

| | value |
|---|---|
| rf / lgbm / svm / mlp | 0.6681 / 0.6807 / 0.5791 / 0.6541 |
| soft_vote (equal weight) | 0.6699 |
| **weighted_vote** | **0.6885** |
| geometric_vote | 0.6870 |
| stacked | 0.6657 |

**OOF-fitted voting weights** — `rf=0.591 lgbm=0.231 svm=0.004 mlp=0.174`.
That SVM weight of 0.004 is the search deleting a member that was hurting the
average. It is the most informative single number in the run.

Then:
```bash
python scripts/verify.py --data /tmp/dataset.npz      # expect 19/19
python scripts/make_figures.py
```

`verify.py` is worth reading carefully. The three shuffle controls are the ones
that would expose leakage: with labels permuted, the same pipeline must fall
from 0.646 to ≈0.034 (= 1/29). If a permuted-label run scores meaningfully above
chance, something is leaking and every other number is void.

---

## 5. Stage 4 — the leakage audit (~3 min) ← **the most valuable single run**

```bash
python experiments/audit_leakage.py --data /tmp/dataset.npz
```

This deliberately runs the *wrong* split so you can measure what it buys. Each
base waveform appears 5 times (once per SNR); splitting by row lets a
waveform's 40 dB copy sit in training while its 10 dB copy is in test.

What I get:

```
  A. GROUP split (correct)      rf=0.6677  lgbm=0.6728   vote=0.6782
  B. ROW split (naive, leaks)   rf=0.7060  lgbm=0.7308   vote=0.7330
  C. single SNR 40 dB only      rf=0.8488  lgbm=0.8508   vote=0.8588
  C. single SNR 20 dB only      rf=0.7979  lgbm=0.7819   vote=0.7887
  C. single SNR  0 dB only      rf=0.2370  lgbm=0.2461   vote=0.2598

  LEAKAGE INFLATION           : +0.0548  (+8.1% relative)
  test waveforms also in train: 99.6%

  per-SNR:   SNR     group       row     delta
             40 dB   0.8635    0.9561   +0.0926
             30 dB   0.8624    0.9569   +0.0945
             20 dB   0.8075    0.8897   +0.0821
             10 dB   0.6014    0.6234   +0.0221
              0 dB   0.2482    0.2693   +0.0211
```

**Read this carefully — it is the crux of "true performance".**

Under the naive split, 99.6% of test waveforms have a sibling in training, and
at 40 dB the score jumps from an honest 0.864 to 0.956. Nearly ten points, for
free, from a one-line change in how you index the data. The inflation shrinks at
low SNR because heavy noise destroys the waveform-identity cue the model was
exploiting.

If you report the row-split number you are measuring *"can the model recognise
this specific waveform it has seen before"*, not *"can it classify an unseen
disturbance"*. Both are computable from the same array; only one is the thing
anyone cares about.

---

## 6. Stage 5 — confidence intervals (~5 min) ← *can overturn a claim*

```bash
python experiments/multiseed.py --mode split --seeds 0 1 2 3 4
```

Same waveforms, 5 different 70/15/15 partitions. What I measured over 4 seeds:

| model | mean | std | range |
|---|---|---|---|
| rf | 0.6644 | 0.0073 | 0.6545–0.6714 |
| lgbm | **0.6699** | 0.0043 | 0.6635–0.6728 |
| svm | 0.5836 | 0.0060 | 0.5778–0.5894 |
| mlp | 0.6483 | 0.0048 | 0.6429–0.6541 |
| soft vote (equal) | 0.6606 | 0.0043 | 0.6553–0.6652 |

The script now tests **both** ensembles with a paired t-test. Result on the
6-level dataset (5 splits, `--fast` settings):

```
                     mean      std           95% CI
lgbm               0.7025   0.0090   [0.6913, 0.7136]   <- best single model
vote_equal         0.6941   0.0067   [0.6857, 0.7024]
vote_weighted      0.7114   0.0090   [0.7003, 0.7225]

paired comparisons against lgbm:
  vote_equal     mean -0.0084   wins 0/5   p=0.0164  -> WORSE  (p<0.05)
  vote_weighted  mean +0.0089   wins 5/5   p=0.0038  -> BETTER (p<0.05)

fitted weights: rf=0.576  lgbm=0.234  svm=0.031  mlp=0.160
```

**Three conclusions, both ensemble claims now settled:**

1. **Weighted voting genuinely beats the best single model.** 5 wins out of 5,
   p = 0.0038. An earlier draft of this guide called the +0.008 margin "probably
   real, not established" and warned against quoting it — that caution is now
   resolved, and it is safe to report as a real effect.

2. **Equal-weight voting is significantly *worse* than LightGBM alone**
   (0 wins out of 5, p = 0.016). Averaging an uncalibrated SVM at 0.62 into a
   LightGBM at 0.70 costs more than the diversity buys. Weighting fixes it;
   equal weighting does not. This is the single most useful practical finding
   in the project: the standard "just average the models" recipe actively hurts
   here.

3. **Read the paired test, not the confidence intervals.** The CIs for `lgbm`
   and `vote_weighted` overlap heavily — on that evidence alone you would call
   it a tie. But every method sees the *identical* splits, so the split-to-split
   variance is shared and cancels in the difference: individual σ ≈ 0.009, while
   the paired differences scatter by only ≈ 0.003. That is why a +0.009 margin
   reaches p = 0.004. Comparing overlapping CIs is the wrong test for paired
   designs, and it would have hidden a real effect here.

One more thing worth noticing in the fitted weights: **Random Forest receives
the largest weight (0.576) even though LightGBM is the stronger individual model
(0.7025 vs 0.6971).** Ensemble weight goes to whichever member is most
*complementary*, not to whichever scores highest alone.

For a headline number you'd quote publicly, use the more expensive mode, which
draws fresh waveforms each time (~10 min/seed, mostly feature extraction):

```bash
python experiments/multiseed.py --mode data --seeds 1 2 3
```

---

## 7. Stage 6 — reproduce the two structural findings (~10 min)

### 6a. The S-transform is blind to flicker at the fundamental

```bash
python experiments/exp_flicker2.py     # ~3 min
```

Look at the row `pure / pure flicker (control)`. Every full-bandwidth
demodulator (Hilbert, square-law, quarter-cycle RMS) scores **AUC = 1.000**,
while the S-transform envelope scores ~0.45 — chance. Same detector, same
signals; the only difference is which envelope it reads.

That isolates the cause: the S-transform window at frequency *f* has σ_t = 1/*f*
hence **σ_f = f/2π ≈ 8 Hz at 50 Hz**, so its fundamental row low-passes away the
8–25 Hz flicker before any detector sees it. Rows at 150 Hz (σ_f ≈ 24 Hz) and
250 Hz (≈ 40 Hz) keep it.

Sanity-check the consequence directly:

```bash
python -c "
import numpy as np; from features import PQFeatureExtractor
from pqmodel import pqmodel
fx=PQFeatureExtractor(6400.,50.,10,1600.); n=fx.feature_names()
out=pqmodel(ns=8,fs=6400.,f=50.,n=10,A=1.,seed=7)
F=fx.transform(out.transpose(0,2,1).reshape(-1,1280)); cls=np.tile(np.arange(1,30),8)
j=n.index('flk_h1_out')
for a,b in [(2,11),(3,12),(8,18),(9,19)]:
    print(f'c{a}->c{b}: {F[cls==a,j].mean():.4f} -> {F[cls==b,j].mean():.4f}')
"
```
Expect ~0.001 → ~0.04, i.e. a 40–70× jump when flicker is present.

### 6b. Four class pairs are near-degenerate by construction

```bash
python experiments/exp_degeneracy.py   # ~1 min
```

This generates matched parameter pairs, measures the actual waveform difference
‖δ‖, and converts it to the matched-filter ceiling Φ(d′/2). Expect:

| kind | pairs | ‖δ‖ rms | dB below signal |
|---|---|---|---|
| GLOBAL | 2→11, 3→12 | 0.0370 | 25.6 |
| **GATED** | 15→20, 16→21 | **0.0132** | **34.6** |
| **GATED** | 22→28, 23→29 | **0.0126** | **35.0** |

The reason is visible in the model source. Compare in `pqmodel.m`:

```matlab
% GLOBAL - flicker multiplies everything
class 18 = A*AFinal.*(harm).*(1 + lambda*sin(2*pi*ff*t))

% GATED - flicker multiplies only the sag-gated term, which is 0 outside u
class 15 = A*( sin(w0*t-th1) + (harm).*(-alpha*u) )
class 20 = A*( sin(w0*t-th1) + (harm).*(-alpha*u).*(1+lambda*sin(2*pi*ff*t)) )
```

Outside the sag window `u`, classes 15 and 20 are *identical signals*. Confirm
the consequence in your own confusion matrix — the residual 40 dB errors should
be almost exclusively 15↔20, 16↔21, 22↔28, 23↔29:

```bash
python -c "
import json,numpy as np; from pqmodel import CLASS_NAMES
R=json.load(open('results/results.json')); cm=np.array(R['confusion_snr40'])
off=cm.copy(); np.fill_diagonal(off,0)
for a,b in np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:8],off.shape))[0]:
    print(f'{off[a,b]:>3}/150  c{a+1} {CLASS_NAMES[a+1]:<26} -> c{b+1} {CLASS_NAMES[b+1]}')
"
```

And quantify the ceiling's cost — 0.868 → 0.983 at 40 dB once those 8 classes
are removed:

```bash
python -c "
import json,numpy as np; from sklearn.metrics import f1_score
z=np.load('results_preds.npz'); yte,ste=z['yte'],z['ste']; yp=z['E_weighted_vote'].argmax(1)+1
g=[15,16,20,21,22,23,28,29]; keep=sorted(set(range(1,30))-set(g))
for s in [40,30,20,10,0]:
    m=ste==s; k=m&~np.isin(yte,g)
    print(f'{s:>3}dB  all={f1_score(yte[m],yp[m],average=\"macro\"):.4f}   excl-gated={f1_score(yte[k],yp[k],average=\"macro\",labels=keep):.4f}')
"
```

---

## 8. Stage 7 — benchmark against the literature

Published work on this same 29-class Igual generator reports roughly **99.4%
clean, >98% at 20–50 dB**, and S-transform + kernel-ELM methods report **>97% at
20 dB**. Our honest number at 20 dB is **0.811**. Before concluding the pipeline
is weak, account for these, in order of size:

| difference | effect on our 20 dB number |
|---|---|
| Row split instead of group split | +0.082 (measured, Stage 4) |
| Excluding / merging the 4 gated pairs | +0.111 (measured, 0.811 → 0.922) |
| Reporting on 9 or 16 classes, not 29 | large; the 99.78% figure is explicitly 9-class |
| Wavelet denoising before feature extraction | not implemented here; likely material at ≤20 dB |
| Longer observation window than 10 cycles | not tested; §8 of `REPORT.md` predicts +3–5 dB |

Row-split plus gated-pair merging alone takes 0.811 to roughly 0.95 without
touching the model. **When you compare against a paper, the first two questions
to ask are "how many classes?" and "what was the unit of the train/test split?"**
Many papers do not state the second at all, which makes the comparison
unresolvable rather than favourable.

The honest summary of our system: **0.98 macro-F1 at 40 dB and 0.92 at 20 dB
over the 21 classes the generator makes separable**, under a strict
waveform-level split, with no denoising.

---

## 9. Every number you should see, in one place

| stage | quantity | expected | tolerance |
|---|---|---|---|
| 1 | S-transform of pure sinusoid | 0.5000 | ±0.02 |
| 1 | achieved SNR, all levels | target | ±0.5 dB |
| 1 | feature count | 191 | exact |
| 2 | dataset shape | (34800, 191), 5800 groups | exact |
| 3 | split rows | 20300 / 4350 / 4350 | exact |
| 3 | class balance | 700 / 150 / 150 | exact |
| 3 | CV lgbm | 0.6581 ± 0.0109 | ±0.01 |
| 3 | test lgbm | 0.6807 | ±0.005 |
| 3 | test weighted_vote | 0.6885 | ±0.005 |
| 3 | SVM voting weight | 0.004 | qualitative |
| 4 | verify.py | 19/19 | exact |
| 4 | permuted-label F1 | ≈0.034 | < 0.10 |
| 5 | leakage inflation | +0.055 overall, +0.093 @40 dB | ±0.02 |
| 6 | split-to-split σ | 0.004–0.007 | ±0.003 |
| 6 | equal vote vs lgbm | negative, 0/4 wins | sign |
| 7 | GATED ‖δ‖ | 0.013 (vs 0.037 global) | ±0.002 |
| 7 | 40 dB, excl. gated pairs | 0.983 | ±0.01 |

---

## 10. Troubleshooting

**Pipeline finishes in seconds and matches me exactly** → you didn't delete the
checkpoints. See §1.

**Process dies silently, no traceback** → OOM. Exit code 137. Use
`--max-new-folds 1`, and build the dataset one `--step` at a time. The RF is the
memory peak (~550 MB): 300 trees × 29 classes stores a 29-vector per node.

**`build_dataset.py` killed during feature extraction** → joblib's memmap path
overflowing `/dev/shm`. The current version streams one SNR level at a time with
threads and should be fine; if not, lower `BATCH` from 250.

**Numbers off by >0.01** → check LightGBM version first, then confirm your split
geometry line reads exactly `train=20300 val=4350 test=4350`. A wrong split is
far more likely than a wrong model.

**Everything scores ~0.034** → that's chance for 29 classes; your labels and
features are misaligned. Check the class-major reshape in `clean_set`.

---

## 11. What you'll be able to say afterwards

- The pipeline reproduces to ~4 decimals on a fresh build. *(Stage 3)*
- The S-transform and generator match analytic ground truth. *(Stage 2 — real
  validation, independent of my code)*
- No leakage: permuted labels collapse to chance. *(Stage 4)*
- A naive row split inflates the 40 dB score by **+0.093**, and you measured it
  yourself on your own data. *(Stage 5)*
- Differences below ~0.01 in this setup are not interpretable, and equal-weight
  voting is reliably worse than the best single model. *(Stage 6)*
- The residual error is concentrated in four pairs that the generator makes
  near-identical, with a measured evidence gap of 10×. *(Stage 7)*

The thing a re-run cannot give you is proof that my feature code is bug-free.
If you want that, the cheapest high-value check is to re-implement `features.py`
group (A) — the fundamental envelope statistics — from scratch and confirm the
`rel_min` / `rel_max` columns agree to 1e-4. Those drive most of the
sag/swell/interruption performance, so a bug there would matter most.
