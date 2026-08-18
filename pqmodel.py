"""
pqmodel.py -- faithful NumPy port of pqmodel.m

Implements the 29-class integral power-quality disturbance model of:
    R. Igual, C. Medrano, F.J. Arcega, G. Mantescu,
    "Integral mathematical model of power quality disturbances" (2017)

The original MATLAB/Octave source (pqmodel.m, GPLv3) is reproduced here in
NumPy. Every random-parameter range, every unit-step construction and every
signal equation matches the MATLAB source line-for-line. Differences are
limited to:

  * MATLAB `rand` is replaced by numpy.random.Generator.random, so the numeric
    stream differs -- the *distributions* are identical.
  * MATLAB `round` (half away from zero) is reproduced by `mround` rather than
    using np.round (which does banker's rounding).
  * Class 5's exp() terms are evaluated only inside the transient support, so
    the large-but-finite exponentials outside it cannot overflow.

Please cite the paper above if you use this.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 29

CLASS_NAMES = {
    1: "Pure sinusoidal",
    2: "Sag",
    3: "Swell",
    4: "Interruption",
    5: "Impulsive transient",
    6: "Oscillatory transient",
    7: "Harmonics",
    8: "Harmonics + Sag",
    9: "Harmonics + Swell",
    10: "Flicker",
    11: "Flicker + Sag",
    12: "Flicker + Swell",
    13: "Sag + Osc. transient",
    14: "Swell + Osc. transient",
    15: "Sag + Harmonics",
    16: "Swell + Harmonics",
    17: "Notch",
    18: "Harm + Sag + Flicker",
    19: "Harm + Swell + Flicker",
    20: "Sag + Harm + Flicker",
    21: "Swell + Harm + Flicker",
    22: "Sag + Harm + OT",
    23: "Swell + Harm + OT",
    24: "Harm + Sag + OT",
    25: "Harm + Swell + OT",
    26: "Harm + Sag + Flicker + OT",
    27: "Harm + Swell + Flicker + OT",
    28: "Sag + Harm + Flicker + OT",
    29: "Swell + Harm + Flicker + OT",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def mround(x):
    """MATLAB round(): half away from zero (numpy rounds half to even)."""
    return np.floor(np.abs(x) + 0.5) * np.sign(x)


def _iround(x) -> int:
    return int(mround(x))


def _u_window(N: int, start: int, length: int) -> np.ndarray:
    """
    Reproduces  u1 - u2  from the MATLAB source:
        u1 = [zeros(1,start)          ones(1,N-start)]
        u2 = [zeros(1,start+length)   ones(1,N-(start+length))]
    i.e. a unit rectangle that is 1 on [start, start+length) and 0 elsewhere.
    Indices are clamped to [0, N] exactly as MATLAB's zeros/ones concatenation
    degenerates when the count would go negative.
    """
    u = np.zeros(N)
    a = min(max(start, 0), N)
    b = min(max(start + length, 0), N)
    if b > a:
        u[a:b] = 1.0
    return u


def _select_interval(rng, fs, f, N, period_max, period_min):
    """selectInterval() -- sag / swell / interruption support."""
    t1t2 = period_min + (period_max - period_min) * rng.random()
    d = _iround(t1t2 * (fs / f))
    p1 = _iround(rng.random() * N)
    while (p1 + d) > N:
        p1 = _iround(rng.random() * N)
    return _u_window(N, p1, d)


def _select_amp_phase(rng, amp_min, amp_max, ph_min, ph_max):
    """selectAmpPhase()."""
    amp = amp_min + (amp_max - amp_min) * rng.random()
    ph = ph_min + (ph_max - ph_min) * rng.random()
    return amp, ph


def _select_ot_param(rng, b_min, b_max, fn_min, fn_max, tau_min, tau_max,
                     th_min, th_max):
    """selectOscTranParam()."""
    beta = b_min + (b_max - b_min) * rng.random()
    fn = fn_min + (fn_max - fn_min) * rng.random()
    tau = tau_min + (tau_max - tau_min) * rng.random()
    theta = th_min + (th_max - th_min) * rng.random()
    return beta, fn, tau, theta


def _select_ot_interval(rng, fs, f, N, pmin_ot, pmax_ot):
    """
    selectOscTranInterval().
    NOTE: the MATLAB source returns u = u2 - u1, i.e. the rectangle is
    NEGATIVE (-1) on the transient support. This is preserved verbatim.
    Because theta is drawn uniformly on (-pi, pi], a global sign flip is
    absorbed by the random phase, so the model is unaffected -- but we keep
    it identical so results are reproducible against the original code.
    """
    t1t2 = pmin_ot + (pmax_ot - pmin_ot) * rng.random()
    d = _iround(t1t2 * (fs / f))
    p1 = _iround(rng.random() * N)
    while (p1 + d) > N:
        p1 = _iround(rng.random() * N)
    t1 = p1 * (1.0 / fs)
    return -_u_window(N, p1, d), t1


def _select_harm_param(rng, a3_min, a3_max, a5_min, a5_max, th_min, th_max):
    """selectHarmParam()."""
    a3 = a3_min + (a3_max - a3_min) * rng.random()
    a5 = a5_min + (a5_max - a5_min) * rng.random()
    th1 = th_min + (th_max - th_min) * rng.random()
    th3 = th_min + (th_max - th_min) * rng.random()
    th5 = th_min + (th_max - th_min) * rng.random()
    return a3, a5, th1, th3, th5


def _select_flicker_param(rng, l_min, l_max, ff_min, ff_max, ph_min, ph_max):
    """selectFlickerParam()."""
    lam = l_min + (l_max - l_min) * rng.random()
    ff = ff_min + (ff_max - ff_min) * rng.random()
    phi = ph_min + (ph_max - ph_min) * rng.random()
    return lam, ff, phi


def _select_ot_in_sag_swell(rng, fs, f, N, period_min, period_max, pts_fifth):
    """
    selectOscTranInSagSwellInterval().
    Returns (u, utran, t1tran) with u = u1-u2 (positive rectangle for the
    sag/swell) and utran = u2-u1 (negative rectangle for the transient),
    matching the MATLAB source exactly.
    """
    t1t2 = period_min + (period_max - period_min) * rng.random()
    d = _iround(t1t2 * (fs / f))
    p1 = _iround(rng.random() * N)
    while (p1 + d) > N:
        p1 = _iround(rng.random() * N)
    u = _u_window(N, p1, d)

    d_tran = _iround(pts_fifth + (d - pts_fifth) * rng.random())
    p1_tran = _iround(p1 + ((p1 + d - d_tran) - p1) * rng.random())
    t1_tran = p1_tran * (1.0 / fs)
    utran = -_u_window(N, p1_tran, d_tran)
    return u, utran, t1_tran


def _damped_osc(A, beta, t, t1, tau, fn, theta, u):
    """
    A*beta*exp(-(t-t1)/tau) * sin(2*pi*fn*(t-t1)-theta) * u
    Evaluated only on supp(u) so the (finite but large) exponential outside
    the support cannot degrade numerically.
    """
    out = np.zeros_like(t)
    m = u != 0.0
    if not np.any(m):
        return out
    dt = t[m] - t1
    out[m] = (A * beta * np.exp(-dt / tau)
              * np.sin(2 * np.pi * fn * dt - theta) * u[m])
    return out


# --------------------------------------------------------------------------
# main generator
# --------------------------------------------------------------------------
def pqmodel(ns: int = 10, fs: float = 16000.0, f: float = 50.0,
            n: int = 10, A: float = 1.0, seed: int | None = None
            ) -> np.ndarray:
    """
    Generate `ns` random realisations of each of the 29 PQ disturbance classes.

    Parameters
    ----------
    ns   : samples per class            (1 .. 1_000_000)
    fs   : sampling frequency [Hz]      (200 .. 30_000)
    f    : fundamental frequency [Hz]   (40 .. 100)
    n    : cycles per sample            (3 .. 100)
    A    : nominal amplitude            (0.1 .. 400_000)
    seed : RNG seed for reproducibility

    Returns
    -------
    ndarray of shape (ns, PointsPerSignal, 29), float64.
    Axis 2 index k corresponds to model class k+1.
    """
    # ---- parameter validation (mirrors the MATLAB guards) ----------------
    if not (1 <= ns <= 1_000_000):
        raise ValueError("ns must be between 1 and 1,000,000")
    if not (200 <= fs <= 30_000):
        raise ValueError("fs must be between 200 Hz and 30 kHz")
    if not (40 <= f <= 100):
        raise ValueError("f must be between 40 Hz and 100 Hz")
    if not (3 <= n <= 100):
        raise ValueError("n must be between 3 and 100 cycles")
    if not (0.1 <= A <= 400_000):
        raise ValueError("A must be between 0.1 and 400,000")

    rng = np.random.default_rng(seed)

    # ---- random parameter ranges (verbatim from the MATLAB source) -------
    phi_min, phi_max = -np.pi, np.pi
    theta_min, theta_max = -np.pi, np.pi

    period_max, period_min = n - 1, 1
    alpha_min, alpha_max = 0.1, 0.9          # sag depth
    beta_min, beta_max = 0.1, 0.8            # swell / OT magnitude
    rho_min, rho_max = 0.9, 1.0              # interruption depth

    ta_period_min, ta_period_max = 1, n - 1
    psi_min, psi_max = 0.222, 1.11
    onems = _iround(0.001 / (1.0 / fs))      # samples in 1 ms

    ff_min, ff_max = 8.0, 25.0               # flicker frequency
    lambda_min, lambda_max = 0.05, 0.10      # flicker depth

    tau_min, tau_max = 0.008, 0.04
    fn_min, fn_max = 300.0, 900.0
    period_max_ot, period_min_ot = n / 3.33, 0.5
    pts_fifth = _iround(fs / (5 * f))

    alpha1 = 1.0
    a3_min, a3_max = 0.05, 0.15
    a5_min, a5_max = 0.05, 0.15
    a7_min, a7_max = 0.05, 0.15

    k_min, k_max = 0.1, 0.4                  # notch depth
    c_choices = np.array([1, 2, 4, 6])       # notches per cycle
    td_min, tc_min = 0.0, 0.0
    tdtc_min, tdtc_max = 0.01 * (1.0 / f), 0.05 * (1.0 / f)

    # ---- time base -------------------------------------------------------
    N = int(mround((fs / f) * n))
    t = np.arange(N) / fs
    out = np.zeros((ns, N, N_CLASSES), dtype=np.float64)

    w0 = 2 * np.pi * f * t

    def U(pm=period_max, pn=period_min):
        return _select_interval(rng, fs, f, N, pm, pn)

    def OTP():
        return _select_ot_param(rng, beta_min, beta_max, fn_min, fn_max,
                                tau_min, tau_max, theta_min, theta_max)

    def OTI():
        return _select_ot_interval(rng, fs, f, N, period_min_ot, period_max_ot)

    def HP():
        return _select_harm_param(rng, a3_min, a3_max, a5_min, a5_max,
                                  theta_min, theta_max)

    def SSOT():
        return _select_ot_in_sag_swell(rng, fs, f, N, period_min, period_max,
                                       pts_fifth)

    def uni(lo, hi):
        return lo + (hi - lo) * rng.random()

    def harm3(a3, a5, th1, th3, th5):
        """alpha1*sin(w0 t - th1) + a3*sin(3 w0 t - th3) + a5*sin(5 w0 t - th5)"""
        return (alpha1 * np.sin(w0 - th1)
                + a3 * np.sin(3 * w0 - th3)
                + a5 * np.sin(5 * w0 - th5))

    for i in range(ns):
        # --- Class 1: pure sinusoid ---------------------------------------
        phi = uni(phi_min, phi_max)
        out[i, :, 0] = A * np.sin(w0 - phi)

        # --- Class 2: sag --------------------------------------------------
        alpha, phi = _select_amp_phase(rng, alpha_min, alpha_max, phi_min, phi_max)
        u = U()
        out[i, :, 1] = (A * (1 - alpha * u)) * np.sin(w0 - phi)

        # --- Class 3: swell ------------------------------------------------
        beta, phi = _select_amp_phase(rng, beta_min, beta_max, phi_min, phi_max)
        u = U()
        out[i, :, 2] = (A * (1 + beta * u)) * np.sin(w0 - phi)

        # --- Class 4: interruption -----------------------------------------
        rho, phi = _select_amp_phase(rng, rho_min, rho_max, phi_min, phi_max)
        u = U()
        out[i, :, 3] = (A * (1 - rho * u)) * np.sin(w0 - phi)

        # --- Class 5: impulsive transient ----------------------------------
        phi = uni(phi_min, phi_max)
        psi = uni(psi_min, psi_max)
        ta_period = uni(ta_period_min, ta_period_max)
        ta = ta_period * (1.0 / f)
        pts = _iround(ta_period * (fs / f))
        u = _u_window(N, pts, onems)            # u1 - u2
        spike = np.zeros(N)
        m = u != 0.0
        if np.any(m):
            dt = t[m] - ta
            spike[m] = A * psi * (np.exp(-750 * dt) - np.exp(-344 * dt)) * u[m]
        out[i, :, 4] = -spike + A * np.sin(w0 - phi)

        # --- Class 6: oscillatory transient --------------------------------
        phi = uni(phi_min, phi_max)
        beta, fn, tau, theta = OTP()
        u, t1 = OTI()
        out[i, :, 5] = (A * np.sin(w0 - phi)
                        + _damped_osc(A, beta, t, t1, tau, fn, theta, u))

        # --- Class 7: harmonics (3rd, 5th, 7th) -----------------------------
        a3, a5, th1, th3, th5 = HP()
        a7 = uni(a7_min, a7_max)
        th7 = uni(theta_min, theta_max)
        out[i, :, 6] = A * (harm3(a3, a5, th1, th3, th5) + a7 * np.sin(7 * w0 - th7))

        # --- Class 8: harmonics + sag ---------------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        out[i, :, 7] = A * (A * (1 - alpha * u)) * harm3(a3, a5, th1, th3, th5)

        # --- Class 9: harmonics + swell -------------------------------------
        beta = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        out[i, :, 8] = A * (A * (1 + beta * u)) * harm3(a3, a5, th1, th3, th5)

        # --- Class 10: flicker ----------------------------------------------
        lam, ff, phi = _select_flicker_param(rng, lambda_min, lambda_max,
                                             ff_min, ff_max, phi_min, phi_max)
        out[i, :, 9] = A * np.sin(w0 - phi) * (1 + lam * np.sin(2 * np.pi * ff * t))

        # --- Class 11: flicker + sag -----------------------------------------
        alpha = uni(alpha_min, alpha_max)
        lam, ff, phi = _select_flicker_param(rng, lambda_min, lambda_max,
                                             ff_min, ff_max, phi_min, phi_max)
        u = U()
        out[i, :, 10] = (A * np.sin(w0 - phi)) * (lam * np.sin(2 * np.pi * ff * t)
                                                  + (1 - alpha * u))

        # --- Class 12: flicker + swell ---------------------------------------
        beta = uni(beta_min, beta_max)
        lam, ff, phi = _select_flicker_param(rng, lambda_min, lambda_max,
                                             ff_min, ff_max, phi_min, phi_max)
        u = U()
        out[i, :, 11] = (A * np.sin(w0 - phi)) * (lam * np.sin(2 * np.pi * ff * t)
                                                  + (1 + beta * u))

        # --- Class 13: sag + oscillatory transient ----------------------------
        alpha, phi = _select_amp_phase(rng, alpha_min, alpha_max, phi_min, phi_max)
        beta, fn, tau, theta = OTP()
        u, utran, t1tr = SSOT()
        out[i, :, 12] = (A * np.sin(w0 - phi) * (1 - alpha * u)
                         + _damped_osc(A, beta, t, t1tr, tau, fn, theta, utran))

        # --- Class 14: swell + oscillatory transient --------------------------
        beta1, phi = _select_amp_phase(rng, beta_min, beta_max, phi_min, phi_max)
        beta2, fn, tau, theta = OTP()
        u, utran, t1tr = SSOT()
        out[i, :, 13] = (A * np.sin(w0 - phi) * (1 + beta1 * u)
                         + _damped_osc(A, beta2, t, t1tr, tau, fn, theta, utran))

        # --- Class 15: sag + harmonics ----------------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        out[i, :, 14] = A * (alpha1 * np.sin(w0 - th1)
                             + harm3(a3, a5, th1, th3, th5) * (-alpha * u))

        # --- Class 16: swell + harmonics --------------------------------------
        beta = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        out[i, :, 15] = A * (alpha1 * np.sin(w0 - th1)
                             + harm3(a3, a5, th1, th3, th5) * (beta * u))

        # --- Class 17: notch ---------------------------------------------------
        phi = uni(phi_min, phi_max)
        notch_number = int(c_choices[rng.integers(len(c_choices))])
        td_max = 1.0 / (notch_number * f)
        factor = 1.0 / (notch_number * f)
        k = uni(k_min, k_max)
        tdtc = uni(tdtc_min, tdtc_max)
        td = uni(td_min, td_max)
        tc = td - tdtc
        while tc < tc_min:
            td = uni(td_min, td_max)
            tc = td - tdtc
        ut = np.zeros(N)
        for nn in range(n * notch_number):
            p1 = _iround((tc + factor * nn) * fs)
            p2 = _iround((td + factor * nn) * fs)
            ut += k * _u_window(N, p1, p2 - p1)
        s = np.sin(w0 - phi)
        out[i, :, 16] = A * (s - np.sign(s) * ut)

        # --- Class 18: harmonics + sag + flicker --------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        u = U()
        out[i, :, 17] = (A * (A * (1 - alpha * u)) * harm3(a3, a5, th1, th3, th5)
                         * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 19: harmonics + swell + flicker ------------------------------
        beta = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        u = U()
        out[i, :, 18] = (A * (A * (1 + beta * u)) * harm3(a3, a5, th1, th3, th5)
                         * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 20: sag + harmonics + flicker --------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        u = U()
        out[i, :, 19] = A * (alpha1 * np.sin(w0 - th1)
                             + harm3(a3, a5, th1, th3, th5) * (-alpha * u)
                             * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 21: swell + harmonics + flicker ------------------------------
        beta = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        u = U()
        out[i, :, 20] = A * (alpha1 * np.sin(w0 - th1)
                             + harm3(a3, a5, th1, th3, th5) * (beta * u)
                             * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 22: sag + harmonics + OT --------------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        beta, fn, tau, theta = OTP()
        u, utran, t1tr = SSOT()
        out[i, :, 21] = (A * (alpha1 * np.sin(w0 - th1)
                              + harm3(a3, a5, th1, th3, th5) * (-alpha * u))
                         + _damped_osc(A, beta, t, t1tr, tau, fn, theta, utran))

        # --- Class 23: swell + harmonics + OT ------------------------------------
        beta1 = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        beta2, fn, tau, theta = OTP()
        u, utran, t1tr = SSOT()
        out[i, :, 22] = (A * (alpha1 * np.sin(w0 - th1)
                              + harm3(a3, a5, th1, th3, th5) * (beta1 * u))
                         + _damped_osc(A, beta2, t, t1tr, tau, fn, theta, utran))

        # --- Class 24: harmonics + sag + OT ---------------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        utran, t1tr = OTI()
        beta, fn, tau, theta = OTP()
        out[i, :, 23] = (A * harm3(a3, a5, th1, th3, th5) * (1 - alpha * u)
                         + _damped_osc(A, beta, t, t1tr, tau, fn, theta, utran))

        # --- Class 25: harmonics + swell + OT -------------------------------------
        beta1 = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        utran, t1tr = OTI()
        beta2, fn, tau, theta = OTP()
        out[i, :, 24] = (A * harm3(a3, a5, th1, th3, th5) * (1 + beta1 * u)
                         + _damped_osc(A, beta2, t, t1tr, tau, fn, theta, utran))

        # --- Class 26: harmonics + sag + flicker + OT ------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        utran, t1tr = OTI()
        beta, fn, tau, theta = OTP()
        out[i, :, 25] = ((A * harm3(a3, a5, th1, th3, th5) * (1 - alpha * u)
                          + _damped_osc(A, beta, t, t1tr, tau, fn, theta, utran))
                         * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 27: harmonics + swell + flicker + OT ----------------------------
        beta1 = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        u = U()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        utran, t1tr = OTI()
        beta2, fn, tau, theta = OTP()
        out[i, :, 26] = ((A * harm3(a3, a5, th1, th3, th5) * (1 + beta1 * u)
                          + _damped_osc(A, beta2, t, t1tr, tau, fn, theta, utran))
                         * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 28: sag + harmonics + flicker + OT ------------------------------
        alpha = uni(alpha_min, alpha_max)
        a3, a5, th1, th3, th5 = HP()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        u, utran, t1tr = SSOT()
        beta, fn, tau, theta = OTP()
        out[i, :, 27] = A * (alpha1 * np.sin(w0 - th1)
                             + (harm3(a3, a5, th1, th3, th5) * (-alpha * u)
                                + _damped_osc(1.0, beta, t, t1tr, tau, fn, theta, utran))
                             * (1 + lam * np.sin(2 * np.pi * ff * t)))

        # --- Class 29: swell + harmonics + flicker + OT ----------------------------
        beta1 = uni(beta_min, beta_max)
        a3, a5, th1, th3, th5 = HP()
        lam = uni(lambda_min, lambda_max)
        ff = uni(ff_min, ff_max)
        u, utran, t1tr = SSOT()
        beta2, fn, tau, theta = OTP()
        out[i, :, 28] = A * (alpha1 * np.sin(w0 - th1)
                             + (harm3(a3, a5, th1, th3, th5) * (beta1 * u)
                                + _damped_osc(1.0, beta2, t, t1tr, tau, fn, theta, utran))
                             * (1 + lam * np.sin(2 * np.pi * ff * t)))

    return out


# --------------------------------------------------------------------------
# AWGN augmentation
# --------------------------------------------------------------------------
def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add white Gaussian noise at a target SNR.

    SNR_dB = 10*log10(P_signal / P_noise), with P_signal = mean(x**2)
    computed per-signal along the last axis.
    """
    p_sig = np.mean(x ** 2, axis=-1, keepdims=True)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return x + rng.standard_normal(x.shape) * np.sqrt(p_noise)
