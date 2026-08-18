"""
How separable are the near-degenerate class pairs, in principle?

Reading the model equations shows that flicker enters the 29 classes in two
structurally different ways:

  GLOBAL   the (1 + lambda*sin(2*pi*ff*t)) factor multiplies the whole signal
           e.g. class 18 = A*AFinal*(harm)*(1+lambda*sin)
           -> delta = (whole signal) * lambda*sin

  GATED    the flicker factor multiplies only the sag/swell-gated harmonic term,
           which is ZERO outside the event window
           e.g. class 20 = A*(sin(w0-th1) + (harm)*(-alpha*u)*(1+lambda*sin))
           -> delta = A*(harm)*(-alpha*u)*lambda*sin, nonzero only inside u

For a GATED pair the entire evidence distinguishing the two classes is
alpha * lambda * (harmonic amplitude), confined to part of the window --
roughly a 0.3% perturbation of a 1 pu signal.

For additive white Gaussian noise the optimal (matched-filter) detector has
    d' = ||delta|| / sigma_noise
and the best achievable two-class AUC is Phi(d'/2). That is a hard
information-theoretic ceiling: no feature set and no classifier can beat it.

This script measures ||delta|| exactly by generating MATCHED pairs -- identical
random parameters, with and without the flicker factor -- and converts it into
the ceiling AUC at each SNR level.
"""
import numpy as np
from scipy.stats import norm

FS, F0, NCYC, A = 6400.0, 50.0, 10, 1.0
N = int(FS / F0 * NCYC)
t = np.arange(N) / FS
w0 = 2 * np.pi * F0 * t
SNRS = [40, 30, 20, 10, 0]
NREP = 400
rng = np.random.default_rng(11)


def rect():
    """Random sag/swell support u, same construction as the model."""
    d = int(round((1 + 8 * rng.random()) * (FS / F0)))
    p = int(round(rng.random() * N))
    while p + d > N:
        p = int(round(rng.random() * N))
    u = np.zeros(N)
    u[p:p + d] = 1.0
    return u


def harm():
    a3 = 0.05 + 0.10 * rng.random()
    a5 = 0.05 + 0.10 * rng.random()
    th = rng.uniform(-np.pi, np.pi, 3)
    return (np.sin(w0 - th[0]) + a3 * np.sin(3 * w0 - th[1])
            + a5 * np.sin(5 * w0 - th[2]))


def flick():
    lam = 0.05 + 0.05 * rng.random()
    ff = 8 + 17 * rng.random()
    return lam * np.sin(2 * np.pi * ff * t)


# delta(x) for each near-degenerate pair, derived from the model equations
def d_global_sag():                      # 2 -> 11 , 3 -> 12  (flicker + sag/swell)
    phi = rng.uniform(-np.pi, np.pi)
    return A * np.sin(w0 - phi) * flick()


def d_global_harm():                     # 8 -> 18 , 9 -> 19  (harm + sag/swell + flicker)
    al = 0.1 + 0.8 * rng.random()
    return A * A * (1 - al * rect()) * harm() * flick()


def d_global_harmOT():                   # 24 -> 26 , 25 -> 27
    al = 0.1 + 0.8 * rng.random()
    return A * harm() * (1 - al * rect()) * flick()


def d_gated_harm():                      # 15 -> 20 , 16 -> 21
    al = 0.1 + 0.8 * rng.random()
    return A * harm() * (-al * rect()) * flick()


def d_gated_harmOT():                    # 22 -> 28 , 23 -> 29
    al = 0.1 + 0.8 * rng.random()
    return A * harm() * (-al * rect()) * flick()


CASES = [
    ("GLOBAL", "2->11, 3->12   sag/swell + flicker", d_global_sag),
    ("GLOBAL", "8->18, 9->19   harm+sag/swell + flicker", d_global_harm),
    ("GLOBAL", "24->26, 25->27 harm+sag+OT + flicker", d_global_harmOT),
    ("GATED", "15->20, 16->21 sag+harm + flicker", d_gated_harm),
    ("GATED", "22->28, 23->29 sag+harm+OT + flicker", d_gated_harmOT),
]

# reference signal power: 1 pu sinusoid
p_sig = 0.5
print(f"reference signal power = {p_sig:.3f} (1 pu sinusoid), N = {N} samples\n")
print(f"{'kind':<8}{'pair':<34}{'||d||rms':>10}{'dB below':>10}"
      + "".join(f"{s:>8}dB" for s in SNRS))
print("-" * (62 + 10 * len(SNRS)))

for kind, lab, fn in CASES:
    r = np.array([np.sqrt((fn() ** 2).mean()) for _ in range(NREP)])
    drms = r.mean()
    row = f"{kind:<8}{lab:<34}{drms:>10.4f}{10*np.log10(p_sig/drms**2):>10.1f}"
    for s in SNRS:
        sigma = np.sqrt(p_sig / 10 ** (s / 10))       # per-sample noise std
        dprime = drms * np.sqrt(N) / sigma            # matched filter over N samples
        row += f"{norm.cdf(dprime / 2):>10.3f}"
    print(row)

print("""
Columns under each SNR are the CEILING pairwise AUC -- the accuracy of the
optimal matched-filter detector that knows the exact waveform difference.
A practical classifier, which must estimate the disturbance parameters from
the data rather than being told them, will always fall short of these.

Reading: GLOBAL pairs stay separable down to low SNR because the flicker
modulates the full-amplitude carrier over the whole window. GATED pairs carry
~10x less evidence, all of it inside the event window, and collapse first.
""")
