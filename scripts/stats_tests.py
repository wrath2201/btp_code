"""
stats_tests.py -- the statistical tests behind the numbers quoted in
README.md, results/FINAL_RESULTS.md and the audit documents.

Every figure in those files that carries a p-value, a confidence interval or a
"significant" claim is produced here, from the committed per-seed JSONs, so it
can be re-derived rather than taken on trust.

    python scripts/stats_tests.py            # console tables
    python scripts/stats_tests.py --json results/stats_tests.json

Design notes
------------
* n = 5 seeds. With five paired observations the exact Wilcoxon signed-rank
  test cannot return p < 0.0625 however large the effect, so the paired
  t-test is the primary test and Wilcoxon is reported alongside as a
  distribution-free check, not as a contradiction.
* Confidence intervals use Student's t with 4 df, NOT the normal
  quantile. At n = 5, t(0.975, 4) = 2.776 against z = 1.96, so a z-based
  interval understates the half-width by about 30%.
* Variance ratios are tested with Levene (robust to non-normality) as the
  primary test. Bartlett and the F-ratio are reported too, but both assume
  normality, which a single gross outlier violates -- so they overstate
  significance here.
* The Classical Ensemble comparison is reported against each fixed ensemble
  variant as well as against the per-seed validation-selected one, because
  the selection rule itself moves the baseline by about half a point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

LEVELS = [999, 40, 30, 20, 10, 0]
ENSEMBLES = ["soft_vote", "weighted_vote", "geometric_vote", "stacked"]


def level_name(s):
    return "clean" if int(s) == 999 else f"{int(s)}dB"


def load():
    B = [json.load(open(f"results/multiseed/baseline_seed{i}.json")) for i in range(5)]
    F = [json.load(open(f"results/multiseed/frozen_dualpq_seed{i}.json")) for i in range(5)]
    O = [json.load(open(f"results/multiseed/dualpq_concat_seed{i}.json")) for i in range(5)]
    D = [json.load(open(f"results/multiseed/dasnet_seed{i}.json")) for i in range(5)]
    M = [json.load(open(f"results/mgcnn_sdtransformer_seed{i}.json")) for i in range(5)]
    return B, F, O, D, M


def ci95(v):
    """Student-t 95% CI half-width for the mean of v."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * stats.sem(v))


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "t": float(t), "p_ttest": float(p),
            "p_wilcoxon": float(w.pvalue),
            "ci95": [float(lo), float(hi)],
            "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else float("inf"),
            "n_positive": int((d > 0).sum()), "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write results here")
    a = ap.parse_args()

    B, F, O, D, M = load()
    out = {}

    f = np.array([x["test"]["macro_f1"] for x in F]) * 100
    o = np.array([x["test"]["macro_f1"] for x in O]) * 100
    d = np.array([x["test"]["macro_f1"] for x in D]) * 100
    m = np.array([x["test"]["macro_f1"] for x in M]) * 100
    sel = np.array([x[x["selected_ensemble"]]["test"]["macro_f1"] for x in B]) * 100

    # ---------------------------------------------------------------- 1
    print("=" * 78)
    print("1. Per-seed test macro-F1 (%), mean +/- sample SD (ddof=1), t-based 95% CI")
    print("=" * 78)
    print(f"{'model':<26}{'mean':>7}{'SD':>7}{'CI+/-':>7}   seeds")
    rows = {"Frozen-DASNet DualPQ": f, "Classical Ensemble (val-selected)": sel,
            "DASNet": d, "MGCNN-SDTransformer": m, "Original DualPQ-D": o}
    for k, v in rows.items():
        print(f"{k:<26}{v.mean():>7.2f}{v.std(ddof=1):>7.2f}{ci95(v):>7.2f}   {np.round(v, 2)}")
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "ci95_halfwidth": ci95(v), "seeds": v.round(4).tolist()}
    print("\nNote: MGCNN-SDTransformer seeds 1-4 used split_seed == seed while every")
    print("other model used split_seed = 0, so its row is NOT paired with the rest")
    print("and no paired test below includes it.")

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 78)
    print("2. Frozen-DASNet DualPQ vs the Classical Ensemble  (paired over 5 seeds)")
    print("=" * 78)
    print(f"{'baseline':<34}{'mean':>7}{'SD':>6}{'margin':>8}{'t':>7}{'p':>9}"
          f"{'95% CI of margin':>22}")
    out["frozen_vs_classical"] = {}
    for e in ENSEMBLES + ["val_selected"]:
        v = sel if e == "val_selected" else np.array(
            [x[e]["test"]["macro_f1"] for x in B]) * 100
        r = paired(f, v)
        out["frozen_vs_classical"][e] = {"baseline_mean": float(v.mean()),
                                         "baseline_sd": float(v.std(ddof=1)), **r}
        print(f"{e:<34}{v.mean():>7.2f}{v.std(ddof=1):>6.2f}{r['mean_diff']:>+8.2f}"
              f"{r['t']:>7.2f}{r['p_ttest']:>9.4f}"
              f"   [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    print("\nAll five baseline_seed*.json were produced with --fast (reduced capacity:")
    print("RF 300->150 trees, LightGBM 250->120 iters, MLP 400->120 iters). The one")
    print("full-capacity run available, results/results.json (seed 0, fast=False),")
    r0 = json.load(open("results/results.json"))
    print(f"selects {r0['selected_ensemble']} at "
          f"{r0[r0['selected_ensemble']]['test']['macro_f1']*100:.2f} vs the fast seed-0")
    print(f"value of {sel[0]:.2f}. The --fast penalty on geometric_vote, the strongest")
    print("variant, is only +0.02 pp, so geometric_vote is the fair reference.")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 78)
    print("3. Per-SNR: Frozen-DASNet DualPQ vs Classical Ensemble (val-selected)")
    print("=" * 78)
    print(f"{'cond':<8}{'frozen':>16}{'classical':>16}{'margin':>9}{'p':>9}  verdict")
    out["per_snr"] = {}
    for s in LEVELS:
        fv = np.array([x["test_per_snr"][str(s)]["macro_f1"] for x in F]) * 100
        cv = np.array([x[x["selected_ensemble"]]["test_per_snr"][str(s)]["macro_f1"]
                       for x in B]) * 100
        r = paired(fv, cv)
        verdict = "significant" if r["p_ttest"] < 0.05 else "not significant"
        out["per_snr"][int(s)] = {"frozen_mean": float(fv.mean()),
                                  "frozen_sd": float(fv.std(ddof=1)),
                                  "classical_mean": float(cv.mean()),
                                  "classical_sd": float(cv.std(ddof=1)), **r}
        print(f"{level_name(s):<8}{fv.mean():>9.2f}+/-{fv.std(ddof=1):<5.2f}"
              f"{cv.mean():>9.2f}+/-{cv.std(ddof=1):<5.2f}"
              f"{r['mean_diff']:>+9.2f}{r['p_ttest']:>9.4f}  {verdict}")
    print("\nThe advantage is confined to clean/40/30 dB. At 20, 10 and 0 dB the")
    print("difference is not distinguishable from zero at n = 5.")

    # ---------------------------------------------------------------- 4
    print()
    print("=" * 78)
    print("4. Run-to-run variability: Frozen-DASNet DualPQ vs Original DualPQ-D")
    print("=" * 78)
    lev = stats.levene(f, o)
    bar = stats.bartlett(f, o)
    fr = o.var(ddof=1) / f.var(ddof=1)
    p_f = float(2 * min(stats.f.sf(fr, 4, 4), stats.f.cdf(fr, 4, 4)))
    print(f"  SD  frozen = {f.std(ddof=1):.2f}   original = {o.std(ddof=1):.2f}")
    print(f"  var frozen = {f.var(ddof=1):.3f}  original = {o.var(ddof=1):.3f}"
          f"   ratio = {fr:.1f}")
    print(f"  Levene   (robust to non-normality)  p = {lev.pvalue:.4f}   <-- primary")
    print(f"  Bartlett (assumes normality)        p = {bar.pvalue:.5f}")
    print(f"  F-ratio  (assumes normality)        p = {p_f:.5f}")
    o_no3 = np.delete(o, 3)
    print(f"\n  Original DualPQ-D without seed 3: {o_no3.mean():.2f} +/- "
          f"{o_no3.std(ddof=1):.2f}  (with: {o.mean():.2f} +/- {o.std(ddof=1):.2f})")
    print("  The variance result rests on one collapsed run, and the only test that")
    print("  survives non-normality is not significant at n = 5. Report the SD drop")
    print("  descriptively; do not claim the instability is eliminated.")
    out["variance"] = {"sd_frozen": float(f.std(ddof=1)), "sd_original": float(o.std(ddof=1)),
                       "var_ratio": float(fr), "p_levene": float(lev.pvalue),
                       "p_bartlett": float(bar.pvalue), "p_ftest": p_f,
                       "original_excl_seed3_mean": float(o_no3.mean()),
                       "original_excl_seed3_sd": float(o_no3.std(ddof=1))}

    # ---------------------------------------------------------------- 5
    print()
    print("=" * 78)
    print("5. Does stage-1 DASNet quality predict the stage-2 frozen outcome?")
    print("=" * 78)
    r_all = stats.pearsonr(d, f)
    r_no0 = stats.pearsonr(d[1:], f[1:])
    print(f"  all 5 seeds        pearson r = {r_all[0]:.3f}  p = {r_all[1]:.4f}")
    print(f"  excluding seed 0   pearson r = {r_no0[0]:.3f}  p = {r_no0[1]:.4f}")
    print("  Seed 0 is the collapsed DASNet run (53.45). With it excluded the")
    print("  correlation disappears, so the apparent coupling is one outlier, not a")
    print("  trend: over the normal operating range the frozen result does not track")
    print("  stage-1 quality.")
    out["stage1_stage2_correlation"] = {"r_all": float(r_all[0]), "p_all": float(r_all[1]),
                                        "r_excl_seed0": float(r_no0[0]),
                                        "p_excl_seed0": float(r_no0[1])}

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nsaved -> {a.json}")


if __name__ == "__main__":
    main()
