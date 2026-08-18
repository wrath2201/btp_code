# Start here — the plain-English version

No programming knowledge assumed. This explains what the thing does, what each
file is for, and exactly which keys to press to run it.

If you only read one section, read **Part 5**. That's the actual instructions.

---

## Part 1 — What this project actually is

Imagine you're training a junior engineer to diagnose faults on a power line.

You can't wait around for real faults, so you **fake them**. You build a machine
that produces power-line recordings with faults you deliberately put in —
voltage dips, spikes, harmonics, flicker — 29 different fault types in total.
Because you created them, you know the right answer for every single one.

Then you do four things:

1. **Make it realistic.** Real measurements have static in them, so you add
   static at five levels — from barely any, to so much static that the noise is
   as strong as the signal itself.

2. **Take measurements.** You can't hand a wiggly line to a computer and expect
   an answer. So for each recording you measure 191 specific quantities: how far
   the voltage dipped, how long it stayed down, how much 150 Hz content there
   was, whether there was a fast burst, and so on. Each recording becomes a row
   of 191 numbers.

3. **Teach, then examine.** You hide 15% of the recordings in a locked drawer.
   You let the computer study the other 85%, learning which patterns of numbers
   go with which fault. Then you give it the locked-drawer recordings — ones it
   has never seen — and see how many it gets right.

4. **Check you didn't cheat.** This is the part most people skip, and it's the
   part that decides whether your number means anything. More on this in Part 6.

The final score is **0.69 out of 1.0** across all 29 fault types — or **0.98**
if you set aside four fault types that are essentially impossible to tell apart
(explained in Part 7).

---

## Part 2 — Words you'll see, in plain English

| Word | What it means here |
|---|---|
| **signal** / **waveform** | One recording of a power line — 0.2 seconds long. A wiggly line. |
| **class** / **label** | Which of the 29 faults it is. "Class 2" = a voltage dip (sag). |
| **SNR**, measured in **dB** | How much static. **40 dB = very clean. 0 dB = the static is as loud as the signal.** Lower number = harder. |
| **feature** | One measured quantity. "How far did the voltage dip" is one feature. We use 191. |
| **training set** | The recordings the computer is allowed to study. |
| **test set** | The locked drawer. Used once, at the very end, to get an honest score. |
| **model** / **learner** | A method that learns the pattern. We use four different ones, because they make different mistakes. |
| **ensemble** / **voting** | Letting all four vote on each answer, hoping the majority is right more often than any one alone. |
| **macro-F1** | The score, from 0 to 1. It's an average of how well we do on *each* fault type, so getting one rare fault badly wrong hurts as much as a common one. Higher is better. Random guessing = 0.034. |
| **leakage** | Accidentally letting the computer see test answers during study. Makes your score look great and mean nothing. |
| **fold** / **cross-validation** | Splitting the study material 10 ways and rotating which part you hold back, so results don't depend on one lucky split. |

---

## Part 3 — What's in the folder

### Files you will actually type the name of (5 of them)

| File | What it does when you run it | Takes |
|---|---|---|
| `build_dataset.py` | Makes the 29,000 fake recordings and measures all 191 things | ~8 min |
| `pipeline.py` | Splits the data, teaches the four learners, gives the final score | ~25 min |
| `verify.py` | Runs 19 safety checks to prove we didn't cheat | ~1 min |
| `make_figures.py` | Draws the four result pictures | ~10 sec |
| `audit_leakage.py` | Shows how much your score inflates if you split the data wrongly | ~3 min |

### Files that do work behind the scenes — never run these directly

| File | What it is |
|---|---|
| `pqmodel.py` | The signal factory. This is your `pqmodel.m` translated from MATLAB into Python. |
| `features.py` | The measuring instruments — all 191 measurements live here. |

`build_dataset.py` reaches into these two automatically. You never touch them.

### Results that appear after you run things

| File | What it is |
|---|---|
| `results.json` | Every number, saved. Machine-readable. |
| `fig1_snr_degradation.png` | How the score falls as static increases |
| `fig2_class_snr_heatmap.png` | **The most useful picture.** Score for each of the 29 faults at each static level. Green = good. |
| `fig3_confusion.png` | What gets mistaken for what |
| `fig4_feature_importance.png` | Which of the 191 measurements mattered most |

### Reading material

| File | What it is |
|---|---|
| `REPORT.md` | The full write-up: results, the two discoveries, and recommendations |
| `REPLICATION_GUIDE.md` | Technical version of Part 5 below, with all expected numbers |
| `START_HERE.md` | This file |

### The supporting files — evidence, not leftovers

You won't run these every time, but each exists to answer a specific question
somebody will eventually ask you about this work. Two of them are arguably the
most important files in the folder.

#### The proof files — run these before you trust any number

| File | Question it answers | Time |
|---|---|---|
| `test_pqmodel.py` | Is the signal factory producing correct waveforms? | ~20 sec |
| `test_features.py` | Do the 191 measurements measure what they claim? | ~30 sec |

```
python test_pqmodel.py
python test_features.py
```

**These two are different in kind from everything else here.** They check against
answers that are true by mathematics and physics, not against my results.

For example: a pure 50 Hz sine wave of amplitude 1.0 *must* appear in the
Stockwell transform as exactly 0.5, at bin 10, unchanging over time. That's
provable on paper before you write a line of code. `test_features.py` checks it,
and also checks that a 50% voltage dip reads as 0.50, that the harmonics come out
between 0.05 and 0.15 as the model specifies, and that the added noise really
lands at the SNR it was asked for.

Why that matters: every other file in this project can only tell you the code is
*self-consistent*. If the signal factory had a bug, the whole pipeline would run
happily and produce confident, wrong numbers. These two would fail — no matter
whose computer ran them. They're your only independent check that the foundations
are sound.

#### The investigation files — the evidence behind the two discoveries

| File | What it establishes | Time |
|---|---|---|
| `exp_flicker.py` … `exp_flicker4.py` | Four rounds of detective work that found *why* the flicker faults were being confused | 2–5 min each |
| `exp_degeneracy.py` | Measures how much distinguishing evidence each fault pair actually contains | ~1 min |

`REPORT.md` makes two strong claims: that the Stockwell transform is blind to
flicker at the fundamental frequency, and that four fault pairs are
near-identical by construction. **When your examiner asks "how do you know
that?", these files are the answer.** Without them both claims are just
assertions.

Two are worth running to see the evidence directly:

```
python exp_flicker2.py      (~3 min)
python exp_degeneracy.py    (~1 min)
```

In `exp_flicker2.py`, look at the row labelled `pure / pure flicker (control)`.
The full-bandwidth measurements (`Hilbert`, `square-law`, `qcycle-RMS`) score
**1.000** — perfect separation — while the Stockwell measurement scores about
0.5, which is coin-flipping. Same detector, same signals; the only difference is
which measurement it reads. That one row is what proved the Stockwell transform
was destroying the flicker, rather than the detector being weak.

`exp_degeneracy.py` prints the size of the actual signal difference for each
fault pair. The four problem pairs come out at **0.013**, against **0.037** for
the pairs that work — roughly a third the size, which is about **eight times
less signal power** to work with (9 dB weaker). That is the whole argument of
Part 7 in a single line of output.

One more reason to keep them: `exp_flicker.py` and `exp_flicker2.py` both
**failed** — every detector scored at chance. That failure is what located the
real cause. A record of what didn't work, and why, is worth as much in a project
report as the thing that finally did.

#### The extra experiments — how much to trust the numbers

| File | Question it answers | Time |
|---|---|---|
| `multiseed.py` | How much would my score change if the data had been split differently? | ~5 min |
| `unseen_snr.py` | What happens at a noise level the system was never trained on? | ~5 min |

```
python multiseed.py --mode split --seeds 0 1 2 3 4
```

`multiseed.py` re-runs everything with five different random splits. The answer:
scores wobble by about **±0.005**. That's your margin of error, and it means any
difference smaller than about 0.01 is noise, not a finding — useful discipline
before you claim one method beat another.

```
python unseen_snr.py --only 0     (then --only 1, 2, 3, 4)
python unseen_snr.py --merge
```

`unseen_snr.py` trains on four noise levels and tests on the fifth, which it has
never seen. Result: the system copes fine with *less* noise than it trained on,
but collapses completely on *more*. Held out, the 0 dB level scores 0.051 — barely
above the 0.034 you'd get by guessing. Worth knowing before anyone asks how it
would behave on a noise level you didn't anticipate.

---

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
del results_oof_ckpt.npz results_base_ckpt.npz results.json results_preds.npz
```
If it says "Could not find" — that's fine, it just means they were already gone.

### Step 1 — Build the recordings (~8 minutes)

Six lines. Type one, press Enter, **wait for it to finish**, then type the next.
Each of the first five takes about 90 seconds.

```
python build_dataset.py --step 0 --n-base 200 --shard-dir data\shards
python build_dataset.py --step 1 --n-base 200 --shard-dir data\shards
python build_dataset.py --step 2 --n-base 200 --shard-dir data\shards
python build_dataset.py --step 3 --n-base 200 --shard-dir data\shards
python build_dataset.py --step 4 --n-base 200 --shard-dir data\shards
python build_dataset.py --merge --steps 0 1 2 3 4 --shard-dir data\shards --out data\dataset.npz
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
python build_dataset.py --step 5 --n-base 200 --shard-dir data\shards
```

Then pick which dataset you want. **These are two different experiments** — see
Part 7b for which to use when.

**A — clean data only** (the "best case" number):
```
python build_dataset.py --merge --steps 5 --shard-dir data\shards --out data\dataset_clean.npz
```

**B — clean added as a 6th level alongside the noisy ones:**
```
python build_dataset.py --merge --steps 0 1 2 3 4 5 --shard-dir data\shards --out data\dataset_all6.npz
```

⚠ **Once shard 5 exists, a plain `--merge` with no `--steps` picks up all six.**
If you want your original 5-level dataset back, say so explicitly:
```
python build_dataset.py --merge --steps 0 1 2 3 4 --shard-dir data\shards --out data\dataset.npz
```

### Step 2 — Teach and examine (~25 minutes)

One line. Then go make tea.
```
python pipeline.py --data data\dataset.npz --out results.json --folds 10
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
python pipeline.py --data data\dataset.npz --out results.json --folds 10 --max-new-folds 1
```
Then run the full line above once at the end.

**To run the clean experiment instead**, point the same command at the other
dataset and give the answers a different name (so your noisy results aren't
overwritten):
```
del results_clean_oof_ckpt.npz results_clean_base_ckpt.npz results_clean.json results_clean_preds.npz
python pipeline.py --data data\dataset_clean.npz --out results_clean.json --folds 10
python make_figures.py --results results_clean.json --prefix figclean
```
The clean run is much faster — about 4 minutes, because there are 5,800
recordings instead of 29,000.

### Step 3 — Prove we didn't cheat (~1 minute)

```
python verify.py --data data\dataset.npz
```
You want the last line to say **`19/19 checks passed`**. See Part 6 for why this
is the most important thing on the page.

### Step 4 — Draw the pictures (~10 seconds)

```
python make_figures.py
```
Four `.png` files appear in the folder. Double-click them to view. Start with
`fig2_class_snr_heatmap.png`.

### Step 5 — The leakage test (~3 minutes)

```
python audit_leakage.py --data data\dataset.npz
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
