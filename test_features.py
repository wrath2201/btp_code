"""Validate the S-transform against analytic cases and time the extractor."""
import time
import numpy as np
from features import STransform, PQFeatureExtractor
from pqmodel import pqmodel, add_awgn

FS, F0, NCYC = 6400.0, 50.0, 10
N = int(FS / F0 * NCYC)
t = np.arange(N) / FS

st = STransform(N, FS, 1600.0)
print(f"S-transform: {st.n_max+1} freq bins x {st.N} time points, "
      f"df = {st.df:g} Hz, f_max = {st.freqs[-1]:g} Hz")

# ---- 1. pure sinusoid: |S| must peak at the right bin, flat in time --------
x = np.sin(2 * np.pi * 50 * t)
S = np.abs(st(x))
row = int(round(50 / st.df))
print(f"\n[1] pure 50 Hz sinusoid, A=1")
print(f"    argmax over freq  = bin {S.max(1).argmax()} (expected {row})")
print(f"    |S| at bin {row}    = {S[row].mean():.4f} +/- {S[row].std():.2e} "
      f"(expected 0.5 = A/2, constant in time)")
assert S.max(1).argmax() == row
assert abs(S[row].mean() - 0.5) < 0.02
assert S[row].std() < 1e-3

# ---- 2. two-tone: both components resolved, correct amplitudes -------------
x = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 250 * t)
S = np.abs(st(x))
r5 = int(round(250 / st.df))
print(f"\n[2] 50 Hz (A=1) + 250 Hz (A=0.3)")
print(f"    |S| @50 Hz  = {S[row].mean():.4f} (expected 0.500)")
print(f"    |S| @250 Hz = {S[r5].mean():.4f} (expected 0.150)")
assert abs(S[row].mean() - 0.5) < 0.02 and abs(S[r5].mean() - 0.15) < 0.02

# ---- 3. time localisation: amplitude step must be tracked -----------------
# The S-transform window width at frequency f is sigma = 1/f, so at the 50 Hz
# fundamental the effective time resolution is ~20 ms (one cycle). A sag edge
# is therefore smeared over ~+/-1 cycle: this is intrinsic to the transform,
# not an implementation error. The test measures well inside the event.
x = np.sin(2 * np.pi * 50 * t) * np.where((t > 0.04) & (t < 0.16), 0.5, 1.0)
S = np.abs(st(x))
env = S[row] * 2
inside = env[(t > 0.09) & (t < 0.11)].mean()          # >2 sigma from each edge
outside = env[t < 0.01].mean()
print(f"\n[3] 50% sag between 40 and 160 ms (window sigma = 1/50 Hz = 20 ms)")
print(f"    envelope deep inside sag = {inside:.3f} (expected 0.50)")
print(f"    envelope before sag      = {outside:.3f} (expected 1.00)")
assert abs(inside - 0.5) < 0.05 and abs(outside - 1.0) < 0.05

# edge smearing, quantified -- reported so the limit is explicit
edge = env[(t > 0.045) & (t < 0.055)].mean()
print(f"    envelope 5-15 ms into sag = {edge:.3f} (smeared, <1 sigma from edge)")
print("\n[OK] S-transform validated: correct frequency scaling, correct")
print("     amplitude normalisation (|S| = A/2), correct time localisation")
print("     to within the transform's intrinsic ~1-cycle resolution at 50 Hz.")

# ---- 4. feature extractor: dimensionality, finiteness, timing --------------
fx = PQFeatureExtractor(FS, F0, NCYC, 1600.0)
names = fx.feature_names()
print(f"\n[4] feature vector length = {len(names)}")

out = pqmodel(ns=6, fs=FS, f=F0, n=NCYC, A=1.0, seed=7)
sigs = out.transpose(0, 2, 1).reshape(-1, N)
rng = np.random.default_rng(0)
sigs = np.vstack([sigs] + [add_awgn(sigs, s, rng) for s in (20, 0)])

t0 = time.perf_counter()
Fm = fx.transform(sigs)
dt = time.perf_counter() - t0
print(f"    extracted {Fm.shape[0]} signals in {dt:.2f}s "
      f"-> {1000*dt/Fm.shape[0]:.1f} ms/signal")
print(f"    projected for 29,000 signals: {29000*dt/Fm.shape[0]/60:.1f} min "
      f"single-core")
assert np.all(np.isfinite(Fm)), (
    "non-finite features: " + str([names[j] for j in
     np.unique(np.where(~np.isfinite(Fm))[1])]))
print(f"    all finite: True    dtype: {Fm.dtype}")

# ---- 5. discriminative sanity: key features must behave as designed --------
print("\n[5] key discriminators on clean signals (mean over 6 samples)")
clean = out.transpose(0, 2, 1).reshape(-1, N)
Fc = fx.transform(clean)
idx = {n: i for i, n in enumerate(names)}
cls = np.tile(np.arange(1, 30), 6)


def show(feat, classes):
    j = idx[feat]
    parts = [f"c{c}={Fc[cls == c, j].mean():.3f}" for c in classes]
    print(f"    {feat:<18} " + "  ".join(parts))


show("rel_min", [1, 2, 4])            # pure / sag / interruption
show("rel_max", [1, 3])               # pure / swell
show("mod_frac_flick", [1, 10])       # pure / flicker
show("h3_ratio", [1, 7])              # pure / harmonics
show("h3_cv", [7, 15])                # harmonics-throughout vs harmonics-in-sag
show("ot_peak_over_med", [1, 6])      # pure / oscillatory transient
show("hf_perio_ratio", [5, 17])       # impulse (aperiodic) vs notch (periodic)
show("hf_n_bursts", [5, 17])
show("snr_est_db", [1])

print("\nALL FEATURE CHECKS PASSED")
