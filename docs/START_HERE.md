# Start here — the plain-English version

No programming knowledge assumed. This file explains what the project is, the final proposed method, the baselines, and exactly how the data and metrics work.

---

## Part 1 — What this project actually is

Imagine you're training a system to diagnose faults on a power line. You build a machine that produces synthetic power-line recordings with faults deliberately inserted — voltage dips, spikes, harmonics, flicker — 29 different fault types in total. 

We do four things:
1. **Add Extreme Noise.** Real measurements have static. We add noise at five levels — from 40 dB (clean) down to 0 dB, where the static is as loud as the underlying signal.
2. **Prevent Cheating (Grouped Splitting).** We group all noise variants of the same base waveform together. If the clean version of a signal is in the training set, all of its noisy versions are also in the training set. This prevents cross-variant leakage where the model just memorizes a specific waveform.
3. **Train on Five Independent Seeds.** We run the entire training process five separate times from scratch to ensure the results aren't just a lucky fluke.
4. **Evaluate using Macro-F1.** The final score is the Macro-F1 metric, which ensures that performing poorly on rare classes heavily penalizes the overall score.

---

## Part 2 — The Final Research Story

Our research evolved through several stages:

### 1. The Baselines
- **Classical Ensemble:** We built a baseline that measures 191 specific handcrafted quantities (like how far the voltage dipped) and feeds them into standard machine learning models.
- **DASNet (Deep Baseline):** We evaluated a project-developed deep-learning architecture that learns features directly from the raw waveform using a Learnable Discrete Stockwell Transform.
- **MGCNN-SDTransformer:** An external published baseline (Jiang et al., 2025) that we reimplemented and evaluated under our rigorous benchmark.

### 2. The Initial DualPQ-D Architecture
We proposed a hybrid architecture (Original DualPQ-D) that combined the deep DASNet representation with our 191 classical features and trained them together end-to-end. However, we discovered severe seed-to-seed instability — sometimes it performed incredibly well, and sometimes it collapsed completely.

### 3. The Final Proposed Method: Frozen-DASNet DualPQ
To solve the instability, we proposed a **decoupled fusion strategy**. We take a pretrained DASNet model and completely **FREEZE** it. We then combine this frozen deep representation with the classical feature branch and train only the classical branch and the final fusion layer. 

**This final method achieved 74.46% ± 1.08% Macro-F1 across five seeds**, vastly outperforming the baselines and resolving the optimization instability.

---

## Part 3 — Where are the numbers?

The authoritative results are directly extracted from saved prediction artifacts (the `*_preds.npz` files in the `results/` folder).

| Model | Macro-F1 (Mean ± Sample SD) |
|---|---:|
| **Frozen-DASNet DualPQ** | **74.46% ± 1.08%** |
| Classical Ensemble | 71.52% ± 0.84% |
| DASNet | 69.72% ± 9.11% |
| MGCNN-SDTransformer | 66.59% ± 0.98% |
| Original DualPQ-D | 61.63% ± 15.58% |

---

## Part 4 — Which files should I read?

- **`README.md`**: The main page describing the final project architecture and results.
- **`PUBLICATION_AUDIT.md`**: The rigorous scientific audit validating all reported numbers.
- **`REPLICATION_GUIDE.md`**: Instructions on how to reproduce the results.

---

> [!WARNING]
> **Historical / Investigation Instructions Below**
> The remaining sections below contain the original setup and execution instructions for the Classical Ensemble baseline investigation. While this provides valuable historical context (such as how the dataset is built and why the classical baseline struggles with flicker), it does not describe the execution of the final Frozen-DASNet architecture. For current execution, see the `README.md`.

## Part 4 — One-time setup

You only ever do this once.

**Step A — Install Python.**
Go to **python.org/downloads** and click the big yellow download button.

⚠ **When the installer opens, tick the box that says "Add python.exe to PATH"
at the bottom of the first screen before clicking Install.** If you miss it,
nothing below will work and you'll have to reinstall.

**Step B — Open the black window.**
Press the **Windows key**, type `cmd`, press **Enter**. A black window opens.
This is the Command Prompt. You type commands here and press Enter.

**Step C — Check Python is there.** Type this and press Enter:
```
python --version
```
You should see something like `Python 3.12.1`. If you see "not recognised",
Step A's tick box was missed.

**Step D — Go to the project folder.** Type these two lines, pressing Enter
after each:
```
D:
cd \BTP\pq_ensemble
```
The start of the line should now read `D:\BTP\pq_ensemble>`.

**Step E — Install the toolkits.** One line, then Enter. Takes a few minutes.
```
pip install numpy scipy scikit-learn pandas lightgbm matplotlib joblib
```

Setup done. You never repeat Part 4.

---

## Part 5 — Running it

Every time from now on, start by opening the black window and typing:
```
D:
cd \BTP\pq_ensemble
```

### ⚠ First, clear my old results

Otherwise the programs find my saved answers and skip all the work, and you'll
have "reproduced" nothing. Type:
```
del results_oof_ckpt.npz results_base_ckpt.npz results/results.json results_preds.npz
```
If it says "Could not find" — that's fine, it just means they were already gone.

### Step 1 — Build the recordings (~8 minutes)

Six lines. Type one, press Enter, **wait for it to finish**, then type the next.
Each of the first five takes about 90 seconds.

```
python scripts/build_dataset.py --step 0 --n-base 200 --shard-dir data\shards
python scripts/build_dataset.py --step 1 --n-base 200 --shard-dir data\shards
python scripts/build_dataset.py --step 2 --n-base 200 --shard-dir data\shards
python scripts/build_dataset.py --step 3 --n-base 200 --shard-dir data\shards
python scripts/build_dataset.py --step 4 --n-base 200 --shard-dir data\shards
python scripts/build_dataset.py --merge --steps 0 1 2 3 4 --shard-dir data\shards --out data\dataset.npz
```

⚠ **The `--steps 0 1 2 3 4` on the last line is not optional.** Without it, the
merge silently sweeps up *every* batch it can find in the folder — including the
clean one from Step 1b if you've already built it — and you end up training on
six levels while believing you used five.

Why five separate lines? Each one handles a different static level. Doing all
five at once uses too much memory on a normal laptop and crashes.

**You'll know it worked** when the last line prints:
```
merged -> data\dataset.npz   X=(29000, 191)  groups=5800  classes=29
```
29,000 recordings, 191 measurements each. That's the spreadsheet.

### Step 1b — Clean data, with no noise at all (optional, ~2 minutes)

Steps 0–4 add noise. **Step 5 adds no noise at all** — the recordings exactly as
the factory made them. This gives you the best-case number: how well the method
does when nothing is working against it.

Build the clean batch once (this does not disturb what you already have):
```
python scripts/build_dataset.py --step 5 --n-base 200 --shard-dir data\shards
```

Then pick which dataset you want. **These are two different experiments** — see
Part 7b for which to use when.

**A — clean data only** (the "best case" number):
```
python scripts/build_dataset.py --merge --steps 5 --shard-dir data\shards --out data\dataset_clean.npz
```

**B — clean added as a 6th level alongside the noisy ones:**
```
python scripts/build_dataset.py --merge --steps 0 1 2 3 4 5 --shard-dir data\shards --out data\dataset_all6.npz
```

⚠ **Once shard 5 exists, a plain `--merge` with no `--steps` picks up all six.**
If you want your original 5-level dataset back, say so explicitly:
```
python scripts/build_dataset.py --merge --steps 0 1 2 3 4 --shard-dir data\shards --out data\dataset.npz
```

### Step 2 — Teach and examine (~25 minutes)

One line. Then go make tea.
```
python scripts/run_pipeline.py --data data\dataset.npz --out results/results.json --folds 10
```

It prints progress as it goes. Near the end you should see:
```
       rf    val F1=0.6674  test F1=0.6681
       lgbm  val F1=0.6687  test F1=0.6807
       svm   val F1=0.5789  test F1=0.5791
       mlp   val F1=0.6433  test F1=0.6541
       weighted_vote  ... test F1=0.6885
```
Those are the four learners and their vote. **Your numbers should match mine to
about three decimal places.**

*If it crashes partway* (the window closes or an error appears), your laptop ran
out of memory. Run this line ten times instead — it does one chunk at a time and
remembers where it got to:
```
python scripts/run_pipeline.py --data data\dataset.npz --out results/results.json --folds 10 --max-new-folds 1
```
Then run the full line above once at the end.

**To run the clean experiment instead**, point the same command at the other
dataset and give the answers a different name (so your noisy results aren't
overwritten):
```
del results_clean_oof_ckpt.npz results_clean_base_ckpt.npz results/results_clean.json results_clean_preds.npz
python scripts/run_pipeline.py --data data\dataset_clean.npz --out results/results_clean.json --folds 10
python scripts/make_figures.py --results results/results_clean.json --prefix figclean
```
The clean run is much faster — about 4 minutes, because there are 5,800
recordings instead of 29,000.

### Step 3 — Prove we didn't cheat (~1 minute)

```
python scripts/verify.py --data data\dataset.npz
```
You want the last line to say **`19/19 checks passed`**. See Part 6 for why this
is the most important thing on the page.

### Step 4 — Draw the pictures (~10 seconds)

```
python scripts/make_figures.py
```
Four `.png` files appear in the folder. Double-click them to view. Start with
`fig2_class_snr_heatmap.png`.

### Step 5 — The leakage test (~3 minutes)

```
python experiments/audit_leakage.py --data data\dataset.npz
```
Explained in Part 6. This is the most interesting thing you'll run.

---

## Part 6 — The cheating problem (the important bit)

Each of the 5,800 recordings gets copied five times with different amounts of
static. So recording #1 appears five times in the spreadsheet — same fault, same
underlying wiggly line, just noisier each time.

Now: when you put 15% in the locked drawer, **do you move all five copies
together, or do you shuffle all 29,000 rows and grab 15% at random?**

It sounds like a boring bookkeeping question. It isn't.

If you shuffle rows, the clean copy of a recording can end up in the study pile
while its noisy twin sits in the locked drawer. The computer isn't diagnosing
the fault any more — it's recognising *a recording it has already seen*. Like
letting a student memorise the answer sheet, then testing them on the same
questions with slightly smudged printing.

`audit_leakage.py` does it both ways and shows you the difference:

```
  correct way (move all five copies together)  : 0.864
  sloppy way  (shuffle the rows)               : 0.956
  test recordings also seen during study       : 99.6%
```

**Nearly ten points of free score, for a bookkeeping choice.** Under the sloppy
method, 99.6% of "unseen" test recordings had already been seen.

This is very likely a big part of why published papers on this same fault model
report 97–99%. Many don't state which method they used. When you compare your
work against a paper, the two questions to ask are: *how many fault types?* and
*how did they split the data?*

Everything in this project uses the correct method. That's why 0.69 looks modest
— it's an honest number, not a flattering one.

`verify.py` is the other half of this. Its key test scrambles all the labels at
random — deliberately destroying any real pattern — and re-runs everything. The
score must collapse to 0.034 (pure guessing). It does. If it didn't, something
would be leaking and every other number in the project would be worthless.

---

## Part 7 — Why 0.69 and not higher

Almost all the remaining errors are four specific pairs of faults:

- Fault 15 vs 20 · Fault 16 vs 21 · Fault 22 vs 28 · Fault 23 vs 29

Look at `fig2_class_snr_heatmap.png` — you'll see eight pale rows in a sea of
green. Those are these eight.

Here's the reason, and it's a property of the fault model itself, not of our
method. In these pairs, the only difference is a flicker that is switched on
**only during the voltage dip**. Outside the dip, the two faults produce
*literally identical signals*. And during the dip, the difference amounts to
about a 0.3% wobble on the measurement.

We measured this directly: these pairs carry about **10× less distinguishing
evidence** than the other flicker pairs. They are near-duplicates by
construction.

Set those eight aside and the score on the remaining 21 fault types is:

| Static level | Score |
|---|---|
| 40 dB (clean) | **0.98** |
| 30 dB | 0.96 |
| 20 dB | 0.92 |
| 10 dB | 0.71 |
| 0 dB (extreme) | 0.33 |

So the honest summary for your report is: **98% at low noise and 92% at moderate
noise, on the 21 fault types the model makes genuinely distinguishable, using a
strict no-cheating split.**

---

## Part 7b — What the clean run tells you

I ran it. On clean, noise-free data:

| | score |
|---|---|
| All 29 fault types | **0.885** |
| The 21 types excluding the degenerate pairs | **0.990** |
| The 8 degenerate types collapsed into 4 | 0.987 |

**17 of the 29 fault types are identified perfectly — every single test
recording correct.** And the ten worst classes on clean data are, in order:
22, 15, 20, 16, 29, 23, 21, 28 — those first eight are *exactly* the eight
degenerate ones from Part 7.

This is the cleanest proof of the argument in Part 7. With noise removed
entirely — nothing left to blame — the method still can't separate those four
pairs, because the signals genuinely are near-identical. Everything else lands
at 99%.

### Which dataset should you use for what?

| Question you're answering | Use |
|---|---|
| "How good is this method at its best?" | **A — clean only.** This is the number most papers quote as their headline. |
| "How does performance decay with noise?" | Your existing 5-level dataset. Clean adds little — 40 dB is already only 1% noise. |
| "I want a complete curve including a noise-free point" | **B — all six levels.** |
| "What do I report as my main result?" | The 5-level one. It's the honest, realistic number. Quote clean separately as the upper bound. |

⚠ **Don't quote the 6-level number as if it were the 5-level number.** Adding an
easy level pulls the average up without the method having improved at all. If
you report B, say it's six levels including noise-free.

Also note: **clean (0.885) is barely better than 40 dB (0.868).** That's not a
mistake. 40 dB means the noise is about 1% of the signal, which is already
negligible. It confirms that what limits this system at high signal quality is
the degenerate class pairs, not noise.

---

## Part 8 — If something goes wrong

| What you see | What it means |
|---|---|
| `'python' is not recognized` | The "Add to PATH" box in Part 4 Step A wasn't ticked. Reinstall Python. |
| Step 2 finishes in seconds | You skipped the `del` command. My old answers were still there. Delete them and re-run. |
| The window closes, no message | Out of memory. Use the `--max-new-folds 1` version in Step 2. |
| `No such file or directory: data\dataset.npz` | Step 1 didn't finish. Re-run Step 1. |
| `ModuleNotFoundError` | Part 4 Step E didn't complete. Run the `pip install` line again. |
| Everything scores about 0.034 | Something is badly wrong — that's the random-guessing score. Re-run Step 1 from scratch. |
| Numbers off by more than 0.01 | Check the line that reads `train=20300 val=4350 test=4350`. If those three numbers are different, the split went wrong. |

---

## Part 9 — What to do first

If you want the fastest path to understanding, do this in order:

1. Run **Part 5, Steps 1–4** (about 35 minutes, mostly waiting).
2. Open `fig2_class_snr_heatmap.png`. Find the eight pale rows. That picture is
   the whole result in one image.
3. Run **Step 5** (the leakage test). That number is the most defensible thing
   in the whole project and it's yours, measured on your own machine.
4. Then read `REPORT.md`, which will make far more sense once you've seen the
   pictures.

Ask me about any single step and I'll go slower on it.
