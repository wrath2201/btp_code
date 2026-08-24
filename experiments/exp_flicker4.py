"""
Round 4 -- exploit the frequency-dependent bandwidth of the S-transform.

The S-transform window at frequency f has sigma_t = 1/f, hence sigma_f =
f/(2*pi). So the usable modulation bandwidth of row f is proportional to f:

    row  50 Hz  ->  sigma_f =  8 Hz   flicker (8-25 Hz) is DESTROYED
    row 150 Hz  ->  sigma_f = 24 Hz   flicker is PRESERVED
    row 250 Hz  ->  sigma_f = 40 Hz   flicker is PRESERVED

That maps exactly onto the two ways flicker enters the model:

    GLOBAL flicker modulates the full carrier   -> read it off the Hilbert
                                                   envelope (full bandwidth)
    GATED  flicker modulates only the sag-gated harmonic term -> read it off
                                                   the 150/250 Hz S-rows,
                                                   inside the event window

Detection is a coherent projection onto exp(i*2*pi*ff*t) swept over 8-25 Hz --
the practical stand-in for the matched filter, which the degeneracy analysis
showed has ample margin at >=20 dB.
"""
import numpy as np
from scipy.signal import hilbert
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.features import STransform
from src.pqmodel import pqmodel, add_awgn

FS, F0, NCYC = 6400.0, 50.0, 10
N = int(FS / F0 * NCYC)
SPC = int(FS / F0)
NB = 60
st = STransform(N, FS, 1600.0)
B = {k: int(round(k * F0 / st.df)) for k in (1, 3, 5)}
_FF = np.arange(8.0, 25.25, 0.25)

GATED = [(15, 20, "sag+harm -> +flicker"), (16, 21, "swell+harm -> +flicker"),
         (22, 28, "sag+harm+OT -> +flicker"), (23, 29, "swell+harm+OT -> +flicker")]
GLOBAL = [(2, 11, "sag -> +flicker"), (3, 12, "swell -> +flicker"),
          (8, 18, "harm+sag -> +flicker"), (9, 19, "harm+swell -> +flicker"),
          (24, 26, "harm+sag+OT -> +flicker"), (25, 27, "harm+swell+OT -> +flicker")]

raw = pqmodel(ns=NB, fs=FS, f=F0, n=NCYC, A=1.0, seed=99)
rng = np.random.default_rng(3)


def auc(a, b):
    x = np.concatenate([a, b])
    r = np.argsort(np.argsort(x)) + 1.0
    na, nb = len(a), len(b)
    return (r[na:].sum() - nb * (nb + 1) / 2) / (na * nb)


def sweep(v, t):
    """Max normalised coherent projection onto the 8-25 Hz band."""
    if len(t) < SPC:
        return 0.0
    v = v - v.mean()
    W = 2 * np.pi * np.outer(_FF, t)
    return float(np.hypot(np.cos(W) @ v, np.sin(W) @ v).max() * 2 / len(t))


def prep(x):
    S = np.abs(st(x))
    return {"h1": np.abs(hilbert(np.asarray(x, float))),
            "s1": S[B[1]] * 2, "s3": S[B[3]], "s5": S[B[5]]}


def event_mask(s1):
    rel = s1 / (np.median(s1) + 1e-12)
    return np.abs(rel - 1.0) > 0.08


def outside_mask(s1, guard=SPC):
    ev = event_mask(s1)
    c = np.convolve(ev.astype(float), np.ones(2 * guard + 1), mode="same")
    return c == 0


T = np.arange(N) / FS

DETS = {
    # global-flicker detectors
    "H1 all": lambda d: sweep(d["h1"] / (d["h1"].mean() + 1e-12), T),
    "H1 outside": lambda d: sweep(d["h1"][outside_mask(d["s1"])]
                                  / (d["h1"].mean() + 1e-12),
                                  T[outside_mask(d["s1"])]),
    # gated-flicker detectors: harmonic rows, inside the event
    "S3 all": lambda d: sweep(d["s3"] / (d["s3"].mean() + 1e-12), T),
    "S3 inside": lambda d: sweep(d["s3"][event_mask(d["s1"])]
                                 / (d["s3"].mean() + 1e-12),
                                 T[event_mask(d["s1"])]),
    "S5 inside": lambda d: sweep(d["s5"][event_mask(d["s1"])]
                                 / (d["s5"].mean() + 1e-12),
                                 T[event_mask(d["s1"])]),
    "S3+S5 inside": lambda d: sweep(
        (d["s3"] / (d["s3"].mean() + 1e-12)
         + d["s5"] / (d["s5"].mean() + 1e-12))[event_mask(d["s1"])],
        T[event_mask(d["s1"])]),
}

for snr in (40, 20, 10):
    print(f"\n{'='*104}\nAUC at SNR = {snr} dB\n{'='*104}")
    print(f"{'pair':<32}" + "".join(f"{k:>12}" for k in DETS))
    for tag, PAIRS in (("GATED", GATED), ("GLOBAL", GLOBAL)):
        tot = {k: [] for k in DETS}
        for a, b, lab in PAIRS:
            xa = add_awgn(raw[:, :, a - 1], snr, rng)
            xb = add_awgn(raw[:, :, b - 1], snr, rng)
            da = [prep(v) for v in xa]
            db = [prep(v) for v in xb]
            row = f"{tag[:1]} {lab:<30}"
            for k, fn in DETS.items():
                u = auc(np.array([fn(d) for d in da]),
                        np.array([fn(d) for d in db]))
                tot[k].append(u)
                row += f"{u:>12.3f}"
            print(row)
        print(f"  {tag+' MEAN':<30}" + "".join(f"{np.mean(tot[k]):>12.3f}" for k in DETS))
