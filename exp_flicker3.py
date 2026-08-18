"""
Round 3.

Round 2 established that the demodulators are correct (pure vs pure-flicker
gives AUC = 1.000) but that a co-occurring sag/swell defeats spectral
prominence: a rectangle of duration 20-180 ms has sinc structure on the same
5-50 Hz scale as the flicker line, so "is there a narrow peak in 8-25 Hz"
cannot separate them inside a 0.2 s window.

Key realisation: the two envelope estimators are complementary.

    S-transform fundamental row  ->  rectangle only  (flicker low-passed away
                                     by the sigma_f ~ 8 Hz Gaussian)
    Hilbert / square-law         ->  rectangle + flicker (full bandwidth)

So the S-transform's low-pass behaviour -- the very thing that broke the
original feature -- is exactly the right *baseline estimator*. Two ways to use
it:

    (A) detect the event from the S-envelope, mask it plus one cycle either
        side, and measure AC content of the Hilbert envelope on what remains
    (B) divide the Hilbert envelope by the S-envelope, cancelling the rectangle
        everywhere, then measure AC content (uses all samples, but edges need
        masking because the two estimators smear differently)
"""
import numpy as np
from scipy.signal import hilbert
from features import STransform
from pqmodel import pqmodel, add_awgn

FS, F0, NCYC = 6400.0, 50.0, 10
N = int(FS / F0 * NCYC)
SPC = int(FS / F0)
NB = 80
st = STransform(N, FS, 1600.0)
B1 = int(round(50 / st.df))

PAIRS = [(3, 12, "swell / swell+flicker"),
         (2, 11, "sag / sag+flicker"),
         (15, 20, "sag+harm / sag+harm+flicker"),
         (16, 21, "swell+harm / swell+harm+flicker"),
         (8, 18, "harm+sag / harm+sag+flicker"),
         (9, 19, "harm+swell / harm+swell+flicker"),
         (22, 28, "sag+harm+OT / +flicker"),
         (23, 29, "swell+harm+OT / +flicker"),
         (24, 26, "harm+sag+OT / +flicker"),
         (25, 27, "harm+swell+OT / +flicker"),
         (1, 10, "pure / pure flicker (control)")]

raw = pqmodel(ns=NB, fs=FS, f=F0, n=NCYC, A=1.0, seed=99)
rng = np.random.default_rng(3)


def auc(a, b):
    x = np.concatenate([a, b])
    r = np.argsort(np.argsort(x)) + 1.0
    na, nb = len(a), len(b)
    return (r[na:].sum() - nb * (nb + 1) / 2) / (na * nb)


_FF = np.arange(8.0, 25.5, 0.5)


def _sweep(v, t):
    """Max |projection| of v onto exp(i*2*pi*ff*t) over the flicker band."""
    W = 2 * np.pi * np.outer(_FF, t)
    return float(np.hypot(np.cos(W) @ v, np.sin(W) @ v).max() * 2 / len(t))


def _dilate(mask, k):
    """Binary dilation by k samples each side."""
    c = np.convolve(mask.astype(float), np.ones(2 * k + 1), mode="same")
    return c > 0


def envelopes(x):
    """(full-bandwidth Hilbert envelope, flicker-free S-transform envelope)."""
    return np.abs(hilbert(np.asarray(x, float))), np.abs(st(x))[B1] * 2.0


def clean_mask(s_env, thr=0.08, guard=SPC):
    """
    Samples that are outside any sag/swell/interruption event and at least one
    cycle from its edges. The S-envelope is used for detection precisely
    because it does not contain the flicker.
    """
    rel = s_env / (np.median(s_env) + 1e-12)
    ev = np.abs(rel - 1.0) > thr
    m = ~_dilate(ev, guard)
    return m


def ac_rms_outside(h_env, s_env):
    """(A) AC-RMS of the full-bandwidth envelope on event-free samples."""
    m = clean_mask(s_env)
    if m.sum() < SPC:
        return 0.0
    v = h_env[m]
    return float(v.std() / (np.abs(v).mean() + 1e-12))


def band_peak_outside(h_env, s_env):
    """(A') 8-25 Hz line strength on event-free samples, via Goertzel-style
    projection (robust to the irregular support left by masking)."""
    m = clean_mask(s_env)
    if m.sum() < 2 * SPC:
        return 0.0
    t = np.arange(N)[m] / FS
    v = h_env[m]
    v = (v - v.mean()) / (np.abs(h_env[m]).mean() + 1e-12)
    return _sweep(v, t)


def ratio_ac(h_env, s_env):
    """(B) cancel the rectangle by division, then AC-RMS away from edges."""
    rel = s_env / (np.median(s_env) + 1e-12)
    ev = np.abs(rel - 1.0) > 0.08
    edges = np.zeros(N, bool)
    d = np.diff(ev.astype(np.int8))
    for i in np.flatnonzero(d != 0):
        edges[max(0, i - SPC):min(N, i + SPC)] = True
    r = h_env / (s_env + 1e-9)
    m = ~edges
    if m.sum() < SPC:
        return 0.0
    v = r[m]
    return float(v.std() / (np.abs(v).mean() + 1e-12))


def ratio_band(h_env, s_env):
    """(B') 8-25 Hz line strength of the ratio signal."""
    rel = s_env / (np.median(s_env) + 1e-12)
    ev = np.abs(rel - 1.0) > 0.08
    edges = np.zeros(N, bool)
    d = np.diff(ev.astype(np.int8))
    for i in np.flatnonzero(d != 0):
        edges[max(0, i - SPC):min(N, i + SPC)] = True
    m = ~edges
    if m.sum() < 2 * SPC:
        return 0.0
    t = np.arange(N)[m] / FS
    v = h_env[m] / (s_env[m] + 1e-9)
    v = v - v.mean()
    return _sweep(v, t)


DETS = {"A:ac_out": ac_rms_outside, "A':band_out": band_peak_outside,
        "B:ratio_ac": ratio_ac, "B':ratio_band": ratio_band}

for snr in (40, 10):
    print(f"\n{'='*92}\nAUC at SNR = {snr} dB\n{'='*92}")
    print(f"{'pair':<36}" + "".join(f"{k:>14}" for k in DETS))
    tot = {k: [] for k in DETS}
    for a, b, lab in PAIRS:
        xa = add_awgn(raw[:, :, a - 1], snr, rng)
        xb = add_awgn(raw[:, :, b - 1], snr, rng)
        ea = [envelopes(v) for v in xa]
        eb = [envelopes(v) for v in xb]
        row = f"{lab:<36}"
        for k, fn in DETS.items():
            u = auc(np.array([fn(*e) for e in ea]),
                    np.array([fn(*e) for e in eb]))
            tot[k].append(u)
            row += f"{u:>14.3f}"
        print(row)
    print(f"{'MEAN':<36}" + "".join(f"{np.mean(tot[k]):>14.3f}" for k in DETS))
