# Standards assessment

How this classifier and its dataset measure up against IEC 61000-4-7,
IEC 61000-4-15, EN 50160, IEEE 1159 and IEEE 519.

Two things dominate everything below:

1. **The 200 ms / 5 Hz measurement window is exactly the IEC 61000-4-7
   aperture.** That was not designed for — `n=10` is the generator default and
   `fs=6400` was chosen for notch resolution — but it lands precisely on the
   standard's synchronised 10-cycle block. The harmonic features are therefore
   computed over a standards-legal aperture.
2. **The 29-way single-label formulation is not a standards taxonomy.**
   IEEE 1159 has no category for combined disturbances; a compliant monitor
   reports *concurrent* events. Re-framing the output as multi-label would both
   align with the standard and dissolve most of the residual error (§5).

| Standard | Verdict |
|---|---|
| IEC 61000-4-7 | **Aperture compliant**, estimator non-compliant (2 fixable deviations) |
| IEC 61000-4-15 | **Out of scope** — P_st needs 10 min, we have 0.2 s |
| IEEE 1159 | Magnitudes match exactly; **only 1 of 4 duration categories covered** |
| EN 50160 | Dataset is **non-compliant supply by construction** (~99% of harmonic classes exceed THD limit) |
| IEEE 519 | Same as EN 50160; no aggregation or percentile statistics |

---

## 1. IEC 61000-4-7 — harmonic measurement

### What matches

| Requirement | Standard | This pipeline |
|---|---|---|
| Measurement window | 10 cycles @ 50 Hz (200 ms) | 10 cycles, 200 ms ✔ |
| Frequency resolution | 5 Hz | 5 Hz ✔ |
| Window synchronisation | locked to fundamental | signals generated phase-locked ✔ |
| Anti-alias headroom | evaluate to 40th–50th order | f_s = 6400 Hz → Nyquist 3200 Hz = 64th order ✔ |

This is a real, quotable alignment: **the features are extracted over the same
aperture a Class A instrument would use.**

### Two deviations that are fixable

**(a) Wrong window function.** Feature group G in `features.py` applies
`np.hanning` before the FFT. IEC 61000-4-7 specifies a **rectangular** window on
the synchronised 10-cycle block — the synchronisation is what removes leakage,
so no taper is wanted. Hann widens the mainlobe to ≈2 bins and rescales
amplitudes, so `thd`, `harm_frac` and the band-fraction features are not
directly comparable to instrument readings.

**(b) No harmonic subgrouping.** We read the single 5 Hz bin at *h*·50 Hz.
Class A requires the **harmonic subgroup**, which folds in the two adjacent
interharmonic bins:

```
G²_sg,h  =  C²_(k−1)  +  C²_k  +  C²_(k+1)
```

Without this, a small frequency deviation spills harmonic energy into
neighbouring bins and the harmonic reads low. Our S-transform harmonic features
(`h3_ratio`, `h5_ratio`, …) are a third estimator again — S-transform amplitude
at one bin, neither a raw DFT bin nor a subgroup.

**Not implemented at all:** the 1.5 s smoothing filter, and the 3 s / 10 min /
2 h aggregation intervals. We produce one instantaneous snapshot.

---

## 2. IEC 61000-4-15 — flicker

### This is out of scope, and the reason matters

P_st is defined over **10 minutes**; P_lt over **2 hours**. Our observation
window is **0.2 seconds** — roughly 3000× short. No amount of feature
engineering recovers this: the quantity is not computable from the data we hold.

**The system is a flicker *detector*, not a flickermeter.** It answers "is there
a fluctuation in the 8–25 Hz band?" It cannot answer "is this flicker severe
enough to matter?", which is the question the standard exists to answer.

### What we accidentally got right

The square-law demodulator tested in `exp_flicker2.py` (`env_square`) is
precisely **block 2 of the IEC flickermeter chain**. The coherent 8–25 Hz
projection is a crude stand-in for blocks 3–4.

### What is missing, and why it flatters our results

The standard's **eye–brain weighting filter** (0.05–35 Hz bandpass, peak
sensitivity ≈ 8.8 Hz) is absent. Our detector weights 8 Hz and 25 Hz equally;
perceptually they are very different, and 25 Hz flicker is far less severe.

More importantly — **the flicker in this dataset is enormous by standards
terms:**

| | value |
|---|---|
| Generator flicker depth λ | 0.05 – 0.10 (peak-to-peak ΔV/V = 10–20%) |
| Perception threshold at 8.8 Hz | ≈ 0.3% for 50% of the population (P_st ≈ 1) |
| **Ratio** | **≈ 30–60× the threshold of irritability** |
| IEEE 1159 stated flicker range | 0.1 – 7% — our range exceeds the top |

⚠ **Read our flicker results with this in mind.** We detect flicker that is one
to two orders of magnitude above the level at which it becomes a compliance
problem. A genuine P_st = 1 event would be ~30–60× smaller and correspondingly
harder. **Nothing in these results demonstrates that the method could perform
standards-relevant flicker monitoring.**

That makes the gated-pair failure (§5, `REPORT.md` §5) sharper still: even at
30–60× the perceptibility threshold, gated flicker is undetectable — because the
problem is not amplitude, it is that the evidence is confined to the sag window.

---

## 3. IEEE 1159 — monitoring categories

### Magnitudes: exact matches

The generator's amplitude ranges align with IEEE 1159 to the decimal, which is
strong evidence the model was written against this standard:

| Disturbance | IEEE 1159 | Generator | |
|---|---|---|---|
| Sag | 0.1 – 0.9 pu | 1−α, α ∈ [0.1, 0.9] → 0.1 – 0.9 pu | ✔ exact |
| Swell | 1.1 – 1.8 pu | 1+β, β ∈ [0.1, 0.8] → 1.1 – 1.8 pu | ✔ exact |
| Interruption | < 0.1 pu | 1−ρ, ρ ∈ [0.9, 1.0] → 0 – 0.1 pu | ✔ exact |
| Oscillatory transient (LF) | < 5 kHz, 0 – 4 pu | 300 – 900 Hz, 0.1 – 0.8 pu | ✔ classic capacitor-switching band |

Note this also settles a boundary question: the sag/interruption split at 0.1 pu
follows **IEEE 1159**, not EN 50160 (which draws interruptions at ~1% of U_n).
The two standards disagree, and the generator follows IEEE.

### Durations: only one category of four

Events last `periodMin` to `periodMax` = **1 to 9 cycles (20–180 ms)**, inside a
10-cycle window.

| IEEE 1159 duration category | Range | Covered? |
|---|---|---|
| Instantaneous | 0.5 – 30 cycles | ✔ partially (1–9 cycles) |
| Momentary | 30 cycles – 3 s | ✘ impossible in a 200 ms window |
| Temporary | 3 s – 1 min | ✘ |
| Sustained | > 1 min | ✘ |

**The classifier is validated for instantaneous events only.** A momentary sag
would fill the entire window with no visible edges, and every duration-based
feature (`run_below_*`, `ev_frac`, `env_edge_ratio`) would misread it — likely
as a low-amplitude normal signal. This is a scope limit to state explicitly, not
a defect.

Two smaller mismatches:

- **Oscillatory transient duration.** Ours spans 10–60 ms; IEEE 1159 gives
  0.3–50 ms for low-frequency oscillatory transients. Slightly long at the top.
- **Impulsive transient magnitude.** Peak ≈ 0.05–0.26 pu. Real impulsive
  transients reach several pu. Ours is weak — which is exactly why class 5
  collapses fastest with noise (`REPORT.md`, `test_pqmodel.py`).

### The taxonomy problem

**IEEE 1159 defines no category for combined disturbances.** Of the 29 classes,
only 9 map to a standards category — 1 (normal), 2, 3, 4, 5, 6, 7, 10, 17. The
remaining **20 classes are combinations with no IEEE 1159 label at all.**

A compliant monitor would report concurrent events: *"instantaneous sag, 0.35 pu,
4 cycles; harmonic distortion, THD 14%; voltage fluctuation present."* It would
not choose one of 29 mutually exclusive labels.

---

## 4. EN 50160 and IEEE 519 — limits

Both set the same low-voltage limits: individual harmonics ≤ 5% (h3), ≤ 6% (h5),
≤ 5% (h7), and **THD ≤ 8%**.

### The dataset is non-compliant supply by construction

Generator harmonics are drawn from a3, a5, a7 ~ U(0.05, 0.15), i.e. **5–15% each**:

| Quantity | Generator range | Limit | Fraction exceeding |
|---|---|---|---|
| 3rd harmonic | 5 – 15% | 5% | ~100% |
| 5th harmonic | 5 – 15% | 6% | 90% |
| 7th harmonic | 5 – 15% | 5% | ~100% |
| THD, 3-harmonic classes (cls 7) | 8.7 – 26.0% | 8% | **100%** |
| THD, 2-harmonic classes (17 classes) | 7.1 – 21.2% | 8% | **99.2%** |

*(THD = √(Σa_h²); the 0.8% compliant fraction is the small corner of the
(a3, a5) square where a3² + a5² < 0.0064.)*

**Every harmonic-bearing class in this dataset represents a supply that violates
EN 50160 and IEEE 519 in essentially every draw.** The minimum possible
harmonic amplitude, 5%, already sits at the h3 and h7 limits.

### What that means

The classifier is not separating *compliant* from *non-compliant* supply — every
disturbed sample is non-compliant. It separates **types of violation**. For
compliance monitoring you would need a magnitude-threshold stage on top of the
classifier: classify the disturbance type, then compare measured THD / harmonic
subgroups / P_st against the limit.

Also absent for either standard: the 10-minute aggregation and the 95th-percentile
weekly evaluation both require. IEEE 519 explicitly defers to IEC 61000-4-7 for
measurement methodology, so the §1 deviations apply here too.

---

## 5. What this means for the reported performance

### The honest framing

> Macro-F1 0.886 (clean) / 0.807 (20 dB) across a 29-class research taxonomy,
> under a strict waveform-level split; 0.990 / 0.922 over the 21 classes the
> generator makes separable. Measurement aperture is IEC 61000-4-7 compliant
> (10 cycles, 5 Hz). Scope is limited to IEEE 1159 *instantaneous* events. No
> IEC 61000-4-15 flicker severity is computed, and the flicker present is
> 30–60× the perceptibility threshold, so flicker results are optimistic
> relative to compliance monitoring.

### The reframing that pays off

Because IEEE 1159 reports concurrent events rather than combined labels, the
natural standards-aligned output is **multi-label**:

```
sag ∈ {0,1}   swell ∈ {0,1}   interruption ∈ {0,1}   harmonics ∈ {0,1}
flicker ∈ {0,1}   oscillatory ∈ {0,1}   impulsive ∈ {0,1}   notch ∈ {0,1}
```

**This should largely dissolve the degenerate-pair penalty.** Classes 15 and 20
differ in exactly one bit — flicker. Under the current single-label scheme,
getting that bit wrong is a full misclassification and costs 0.103 macro-F1
across the four pairs. Under multi-label it costs one sub-label; the sag,
harmonics and oscillatory bits stay correct.

Expected outcome, from the confusion structure at 40 dB (where the *only*
remaining errors are 15↔20, 16↔21, 22↔28, 23↔29): per-primitive F1 near 1.0 for
sag, swell, interruption, harmonics, oscillatory, impulsive and notch, with
flicker alone around 0.85–0.92.

This is a prediction from the confusion matrix, not a measurement — I can write
the script to compute it exactly if useful.

---

## 6. Prioritised changes for standards alignment

**1. Switch group-G spectral features to a rectangular window.** One-line change
in `features.py`; makes `thd` and the band fractions comparable to instrument
readings. *(IEC 61000-4-7)*

**2. Add harmonic subgroup features** G²_sg,h = C²_(k−1) + C²_k + C²_(k+1) for
h = 1…13, alongside the existing single-bin ones. Cheap, and makes the harmonic
features quotable as standard quantities. *(IEC 61000-4-7, IEEE 519)*

**3. Switch the classifier head to multi-label.** Highest-value change on this
list: aligns with IEEE 1159's concurrent-event model *and* removes most of the
degeneracy penalty. Keep the 29-way output alongside for comparison with the
literature.

**4. Add explicit compliance flags** — THD > 8%, h3 > 5%, h5 > 6%, h7 > 5% — both
as features and as reported outputs. Turns a type classifier into something that
can support a compliance decision. *(EN 50160, IEEE 519)*

**5. Insert the IEC 61000-4-15 weighting filter** between the existing square-law
demodulator and the flicker detector: 0.05–35 Hz bandpass with the eye–brain
response, then squaring and a 300 ms sliding mean. Yields a P_inst-like
perceptually-weighted statistic instead of a flat 8–25 Hz sweep. *(IEC 61000-4-15)*

**6. Extend the window from 10 to 150 cycles (3 s).** This reaches IEEE 1159's
*momentary* category and IEC 61000-4-7's 3 s aggregation interval — **and it is
already recommendation #2 in `REPORT.md`** for a completely independent reason
(the gated-flicker evidence grows as √N). Standards alignment and the accuracy
fix point at the same change, which makes it the strongest single item here.

**7. Reduce flicker depth toward realistic levels** (λ ≈ 0.003–0.02 rather than
0.05–0.10) in a supplementary dataset, to test whether the method survives at
standards-relevant severity. Expect a substantial drop; that number is more
honest than the current one for any monitoring claim.

---

## Sources

- [EN 50160 voltage characteristics — harmonic limits (BS EN 50160:2007)](https://fs.gongkong.com/files/technicalData/201110/2011100922385600001.pdf)
- [EN 50160 overview — Power Quality Blog](https://powerquality.blog/2021/07/22/standard-en-50160-voltage-characteristics-of-public-distribution-systems/)
- [IEC 61000-4-15:2010 flickermeter — functional and design specifications](https://webstore.iec.ch/en/publication/4173)
- [Flicker perceptibility threshold and P_st — EnerNex](https://www.enernex.com/blog/electric-power-systems-flicker-analysis/)
- [IEEE 1159-2019 summary of power quality indices](https://assets.website-files.com/5eb2d3b23eb980aab682bf00/61b220a2f82003db5fc2fd89_Summary%20poster%20IEEE%201159%202019%20v1r0.pdf)
- [IEEE 1159 categories — sags and swells](https://powerquality.blog/2021/12/01/sags-and-swells/)
- [IEEE 1453 / IEC 61000-4-15 adoption](https://ieeexplore.ieee.org/document/6053977)
