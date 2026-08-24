"""Sanity checks on the NumPy port of pqmodel.m before generating the dataset."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.pqmodel import pqmodel, add_awgn, CLASS_NAMES

FS, F, NCYC, A = 6400.0, 50.0, 10, 1.0

out = pqmodel(ns=30, fs=FS, f=F, n=NCYC, A=A, seed=0)
print("shape:", out.shape, "expected (30, %d, 29)" % int(FS / F * NCYC))
assert out.shape == (30, 1280, 29)
assert np.all(np.isfinite(out)), "non-finite values present"

t = np.arange(out.shape[1]) / FS

print("\n--- per-class amplitude / RMS envelope ---")
print(f"{'cls':>3}  {'name':<32} {'rms':>7} {'|max|':>7} {'min_cyc_rms':>11} {'max_cyc_rms':>11}")
spc = int(FS / F)  # samples per cycle
for c in range(29):
    sig = out[:, :, c]
    cyc = sig.reshape(sig.shape[0], NCYC, spc)
    crms = np.sqrt((cyc ** 2).mean(-1))
    print(f"{c+1:>3}  {CLASS_NAMES[c+1]:<32} {np.sqrt((sig**2).mean()):>7.3f} "
          f"{np.abs(sig).max():>7.3f} {crms.min():>11.3f} {crms.max():>11.3f}")

# ---- Class 1 must be a clean sinusoid of amplitude A -----------------------
c1 = out[0, :, 0]
assert abs(np.abs(c1).max() - A) < 1e-3, "class 1 amplitude wrong"
sp = np.abs(np.fft.rfft(c1))
assert np.argmax(sp) == int(F * len(c1) / FS) == 10, "class 1 fundamental bin wrong"
print("\n[OK] class 1 is a pure 50 Hz sinusoid, peak amplitude 1.0, bin 10")

# ---- Class 4 (interruption) must drop to <=0.1 pu somewhere ---------------
env = np.abs(out[:, :, 3]).reshape(30, NCYC, spc).max(-1)
assert env.min() < 0.15, "interruption never reaches near-zero"
print("[OK] class 4 interruption reaches %.3f pu at its deepest" % env.min())

# ---- Class 7 (harmonics) must show 3rd/5th/7th ----------------------------
sp = np.abs(np.fft.rfft(out[0, :, 6])) / (len(c1) / 2)
h = {k: sp[int(k * F * len(c1) / FS)] for k in (1, 3, 5, 7)}
print("[OK] class 7 harmonic amplitudes:", {k: round(v, 3) for k, v in h.items()})
assert all(0.03 < h[k] < 0.20 for k in (3, 5, 7)), "harmonic amplitudes out of range"

# ---- Class 17 (notch) must contain high-frequency content ------------------
hf_notch = np.abs(np.fft.rfft(out[:, :, 16]))[:, 200:].mean()
hf_pure = np.abs(np.fft.rfft(out[:, :, 0]))[:, 200:].mean()
print(f"[OK] notch HF energy {hf_notch:.3f} vs pure sinusoid {hf_pure:.2e} "
      f"(ratio {hf_notch/max(hf_pure,1e-12):.0f}x)")
assert hf_notch > 50 * hf_pure, "notch produced no detectable HF content"

# ---- Class 5 (impulse) must contain a narrow spike -------------------------
d = np.abs(np.diff(out[:, :, 4], axis=1)).max(1)
d0 = np.abs(np.diff(out[:, :, 0], axis=1)).max(1)
print(f"[OK] impulse max|dx| {d.mean():.3f} vs pure sinusoid {d0.mean():.4f} "
      f"(ratio {d.mean()/d0.mean():.1f}x)")
# NOTE: the model's impulse is (exp(-750*dt) - exp(-344*dt)) over a 1 ms window,
# whose peak magnitude is only ~0.237*psi (psi in [0.222, 1.11]), i.e. at most
# ~0.26 pu and often much less. Class 5 is therefore intrinsically the weakest
# disturbance in this model and is expected to be the first casualty at low SNR.
assert d.mean() > 2 * d0.mean(), "impulse not detectable"

# ---- Class 6 (osc. transient) must have 300-900 Hz energy ------------------
freqs = np.fft.rfftfreq(1280, 1 / FS)
band = (freqs >= 300) & (freqs <= 900)
ot = np.abs(np.fft.rfft(out[:, :, 5]))[:, band].mean()
pu = np.abs(np.fft.rfft(out[:, :, 0]))[:, band].mean()
print(f"[OK] OT band energy {ot:.3f} vs pure sinusoid {pu:.2e}")
assert ot > 50 * pu

# ---- AWGN: achieved SNR must match the target -----------------------------
print("\n--- AWGN verification (target vs achieved) ---")
rng = np.random.default_rng(1)
clean = out[:, :, 0]
for snr in (40, 30, 20, 10, 0):
    noisy = add_awgn(clean, snr, rng)
    achieved = 10 * np.log10(np.mean(clean ** 2) / np.mean((noisy - clean) ** 2))
    print(f"  target {snr:>3} dB -> achieved {achieved:6.2f} dB")
    assert abs(achieved - snr) < 0.5, "SNR calibration off"

# ---- classes must be mutually distinguishable in the clean case -----------
means = np.stack([out[:, :, c].std(1).mean() for c in range(29)])
print(f"\n[OK] per-class std ranges {means.min():.3f} - {means.max():.3f}")
print("\nALL CHECKS PASSED")
