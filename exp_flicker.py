"""
Empirically choose a flicker detector that survives a co-occurring sag/swell.

The failure being fixed: flicker is a 5-10% sinusoidal AM at 8-25 Hz. When a
sag or swell is also present, the rectangular envelope step contributes far
more modulation energy than the flicker ripple, so any *energy-fraction*
feature is swamped. Discrimination must therefore be based on the SHAPE of the
modulation spectrum -- a rectangle gives a smooth 1/f sinc, a tone gives a
narrow peak -- not on energy.

Separability is scored with AUC between class pairs that differ ONLY by the
presence of flicker.
"""
import numpy as np
from scipy.ndimage import median_filter
from features import STransform
from pqmodel import pqmodel, add_awgn

FS, F0, NCYC = 6400.0, 50.0, 10
N = int(FS / F0 * NCYC)
NB = 120
st = STransform(N, FS, 1600.0)
B1 = int(round(50 / st.df))

# class pairs identical except for flicker
PAIRS = [(3, 12, "swell / swell+flicker"),
         (2, 11, "sag / sag+flicker"),
         (15, 20, "sag+harm / sag+harm+flicker"),
         (16, 21, "swell+harm / swell+harm+flicker"),
         (9, 19, "harm+swell / harm+swell+flicker"),
         (23, 29, "swell+harm+OT / +flicker"),
         (24, 26, "harm+sag+OT / +flicker")]

raw = pqmodel(ns=NB, fs=FS, f=F0, n=NCYC, A=1.0, seed=99)
rng = np.random.default_rng(3)


def auc(a, b):
    """AUC of separating b (positive) from a (negative); 0.5 = no signal."""
    x = np.concatenate([a, b])
    r = np.argsort(np.argsort(x)) + 1.0
    na, nb = len(a), len(b)
    return (r[na:].sum() - nb * (nb + 1) / 2) / (na * nb)


def envelope(x):
    return np.abs(st(x))[B1] * 2.0


# ------------------------------------------------------------------ #
# candidate detectors
# ------------------------------------------------------------------ #
def f_energy_frac(env):
    """CURRENT feature: fraction of modulation energy in 8-25 Hz."""
    ac = env / (np.median(env) + 1e-12)
    ac = ac - ac.mean()
    M = np.abs(np.fft.rfft(ac * np.hanning(N)))
    fr = np.fft.rfftfreq(N, 1 / FS)
    b = (fr >= 8) & (fr <= 25)
    return (M[b] ** 2).sum() / ((M ** 2).sum() + 1e-12)


def _prom(env, pad, win_hz, window):
    """Peak-to-local-baseline prominence in the 8-25 Hz modulation band."""
    ac = env / (np.median(env) + 1e-12)
    ac = ac - ac.mean()
    w = {"hann": np.hanning(N), "rect": np.ones(N),
         "hamm": np.hamming(N)}[window]
    n = int(pad * N)
    M = np.abs(np.fft.rfft(ac * w, n=n))
    fr = np.fft.rfftfreq(n, 1 / FS)
    df = fr[1]
    k = int(round(win_hz / df)) | 1
    base = median_filter(M, size=k, mode="nearest")
    p = M / (base + 1e-12)
    b = (fr >= 7) & (fr <= 27)
    return float(p[b].max())


def f_prom_hann(env):
    return _prom(env, 4, 40, "hann")


def f_prom_rect(env):
    return _prom(env, 4, 40, "rect")


def f_prom_rect_narrow(env):
    return _prom(env, 4, 20, "rect")


def f_prom_rect_wide(env):
    return _prom(env, 8, 60, "rect")


def f_sideband(x):
    """
    Interharmonic sidebands. Flicker at ff creates lines at k*f0 +/- ff around
    EVERY harmonic. A rectangular sag also leaks, but its leakage is smooth and
    monotonically decaying -- so again use local prominence, not raw energy.
    """
    n = 4 * N
    X = np.abs(np.fft.rfft(np.asarray(x) * np.hanning(N), n=n))
    fr = np.fft.rfftfreq(n, 1 / FS)
    df = fr[1]
    base = median_filter(X, size=int(round(40 / df)) | 1, mode="nearest")
    p = X / (base + 1e-12)
    best = 0.0
    for k in (1, 3, 5):
        c = k * F0
        for lo, hi in ((c - 27, c - 7), (c + 7, c + 27)):
            b = (fr >= lo) & (fr <= hi)
            if b.any():
                best = max(best, float(p[b].max()))
    return best


def f_sb_consistency(x):
    """
    Offset agreement across harmonics: with flicker, the sideband offset ff is
    the SAME around 50, 150 and 250 Hz. Sag leakage has no such structure.
    """
    n = 4 * N
    X = np.abs(np.fft.rfft(np.asarray(x) * np.hanning(N), n=n))
    fr = np.fft.rfftfreq(n, 1 / FS)
    df = fr[1]
    base = median_filter(X, size=int(round(40 / df)) | 1, mode="nearest")
    p = X / (base + 1e-12)
    offs = []
    for k in (1, 3, 5):
        c = k * F0
        b = (fr >= c + 7) & (fr <= c + 27)
        if b.any():
            offs.append(fr[b][np.argmax(p[b])] - c)
    return -float(np.std(offs)) if len(offs) == 3 else 0.0


DETECTORS_ENV = {
    "energy_frac (current)": f_energy_frac,
    "prom hann pad4 w40": f_prom_hann,
    "prom rect pad4 w40": f_prom_rect,
    "prom rect pad4 w20": f_prom_rect_narrow,
    "prom rect pad8 w60": f_prom_rect_wide,
}
DETECTORS_RAW = {
    "sideband prom": f_sideband,
    "sideband offset consistency": f_sb_consistency,
}

for snr in (40, 10):
    print(f"\n{'='*88}\nAUC of flicker detectors at SNR = {snr} dB "
          f"(1.00 = perfect, 0.50 = useless)\n{'='*88}")
    hdr = f"{'pair':<34}" + "".join(f"{k.split()[0][:8]:>10}" for k in DETECTORS_ENV)
    hdr += f"{'sbProm':>10}{'sbCons':>10}"
    print(hdr)
    tot = {k: [] for k in list(DETECTORS_ENV) + list(DETECTORS_RAW)}
    for a, b, lab in PAIRS:
        xa = add_awgn(raw[:, :, a - 1], snr, rng)
        xb = add_awgn(raw[:, :, b - 1], snr, rng)
        ea = [envelope(v) for v in xa]
        eb = [envelope(v) for v in xb]
        row = f"{lab:<34}"
        for k, fn in DETECTORS_ENV.items():
            u = auc(np.array([fn(e) for e in ea]), np.array([fn(e) for e in eb]))
            tot[k].append(u)
            row += f"{u:>10.3f}"
        for k, fn in DETECTORS_RAW.items():
            u = auc(np.array([fn(v) for v in xa]), np.array([fn(v) for v in xb]))
            tot[k].append(u)
            row += f"{u:>10.3f}"
        print(row)
    print(f"{'MEAN':<34}" + "".join(f"{np.mean(tot[k]):>10.3f}" for k in tot))
