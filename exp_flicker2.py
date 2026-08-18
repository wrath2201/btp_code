"""
Round 2. Round 1 showed every detector at chance, including on near-clean
signals, which pointed at the measurement rather than the detector.

Root cause: the S-transform's Gaussian window at frequency f has time width
sigma_t = 1/f, hence frequency width sigma_f = f/(2*pi) ~ 8 Hz at 50 Hz. The
fundamental row of |S| is therefore an ~8 Hz low-pass of the true amplitude
envelope, and flicker at 8-25 Hz is attenuated by exp(-ff^2 / 2*sigma_f^2)
-- a factor of ~140 at 25 Hz. The flicker is destroyed before any detector
sees it.

Fix: demodulate at full bandwidth (Hilbert / square-law / cycle-RMS) instead of
reading the envelope off the S-matrix, then apply the prominence test.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import hilbert
from features import STransform
from pqmodel import pqmodel, add_awgn

FS, F0, NCYC = 6400.0, 50.0, 10
N = int(FS / F0 * NCYC)
NB = 120
st = STransform(N, FS, 1600.0)
B1 = int(round(50 / st.df))
t = np.arange(N) / FS

PAIRS = [(3, 12, "swell / swell+flicker"),
         (2, 11, "sag / sag+flicker"),
         (15, 20, "sag+harm / sag+harm+flicker"),
         (16, 21, "swell+harm / swell+harm+flicker"),
         (9, 19, "harm+swell / harm+swell+flicker"),
         (23, 29, "swell+harm+OT / +flicker"),
         (24, 26, "harm+sag+OT / +flicker"),
         (1, 10, "pure / pure flicker (control)")]

raw = pqmodel(ns=NB, fs=FS, f=F0, n=NCYC, A=1.0, seed=99)
rng = np.random.default_rng(3)


def auc(a, b):
    x = np.concatenate([a, b])
    r = np.argsort(np.argsort(x)) + 1.0
    na, nb = len(a), len(b)
    return (r[na:].sum() - nb * (nb + 1) / 2) / (na * nb)


# ---------------- envelope estimators ------------------------------------
def env_stransform(x):
    """What the current feature set uses -- ~8 Hz low-passed."""
    return np.abs(st(x))[B1] * 2.0


def env_hilbert(x):
    """Analytic-signal magnitude: full bandwidth."""
    return np.abs(hilbert(np.asarray(x, float)))


def env_square(x):
    """
    Square-law demodulation: x^2 = A^2/2 * (1 - cos(2*w0*t)).
    Low-pass at 60 Hz removes the 2*f0 term and every harmonic beat while
    passing the whole 8-25 Hz flicker band untouched.
    """
    y = np.asarray(x, float) ** 2
    Y = np.fft.rfft(y)
    fr = np.fft.rfftfreq(N, 1 / FS)
    Y[fr > 60] = 0
    return np.sqrt(np.maximum(2 * np.fft.irfft(Y, N), 0))


def env_qcycle(x):
    """RMS over each quarter cycle, then linearly resampled back to N."""
    q = int(round(FS / F0 / 4))
    v = np.asarray(x, float)[: (N // q) * q].reshape(-1, q)
    r = np.sqrt((v ** 2).mean(1)) * np.sqrt(2)
    return np.interp(np.arange(N), np.linspace(0, N - 1, len(r)), r)


ENVS = {"S-transform": env_stransform, "Hilbert": env_hilbert,
        "square-law": env_square, "qcycle-RMS": env_qcycle}


# ---------------- detector applied to an envelope -------------------------
def prominence(env, lo=7.0, hi=27.0, pad=4, win_hz=30.0):
    e = env / (np.median(env) + 1e-12)
    e = e - e.mean()
    n = int(pad * N)
    M = np.abs(np.fft.rfft(e, n=n))          # rectangular window: best resolution
    fr = np.fft.rfftfreq(n, 1 / FS)
    k = int(round(win_hz / fr[1])) | 1
    p = M / (median_filter(M, size=k, mode="nearest") + 1e-12)
    b = (fr >= lo) & (fr <= hi)
    return float(p[b].max())


def band_snr(env, lo=7.0, hi=27.0):
    """Peak in the flicker band relative to the 30-120 Hz modulation floor."""
    e = env / (np.median(env) + 1e-12)
    e = e - e.mean()
    M = np.abs(np.fft.rfft(e, n=4 * N))
    fr = np.fft.rfftfreq(4 * N, 1 / FS)
    b = (fr >= lo) & (fr <= hi)
    f = (fr >= 30) & (fr <= 120)
    return float(M[b].max() / (np.median(M[f]) + 1e-12))


DETS = {"prom": prominence, "bandSNR": band_snr}

for snr in (40, 20, 10, 0):
    print(f"\n{'='*104}\nAUC at SNR = {snr} dB     (1.00 = perfect separation, "
          f"0.50 = chance)\n{'='*104}")
    cols = [(e, d) for e in ENVS for d in DETS]
    print(f"{'pair':<34}" + "".join(f"{e[:6]+'/'+d[:4]:>13}" for e, d in cols))
    tot = {c: [] for c in cols}
    for a, b, lab in PAIRS:
        xa = add_awgn(raw[:, :, a - 1], snr, rng)
        xb = add_awgn(raw[:, :, b - 1], snr, rng)
        row = f"{lab:<34}"
        cache = {}
        for e in ENVS:
            cache[e] = ([ENVS[e](v) for v in xa], [ENVS[e](v) for v in xb])
        for c in cols:
            e, d = c
            ea, eb = cache[e]
            u = auc(np.array([DETS[d](v) for v in ea]),
                    np.array([DETS[d](v) for v in eb]))
            tot[c].append(u)
            row += f"{u:>13.3f}"
        print(row)
    print(f"{'MEAN':<34}" + "".join(f"{np.mean(tot[c]):>13.3f}" for c in cols))
