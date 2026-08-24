"""
features.py -- Stockwell (S-transform) + time-domain + frequency-domain
feature extraction for 29-class power-quality disturbance classification.

Design principle
----------------
The 29 classes of the Igual model are combinations of six *primitive*
disturbances. The feature set is built so that each primitive has a dedicated,
physically-motivated group of descriptors, and so that the model can tell apart
the two harmonic families that most papers confuse:

  primitive                   physical signature            feature group
  --------------------------  ----------------------------  -------------
  sag / swell / interruption  rectangular AM of the 50 Hz    (A) fundamental
                              carrier                            envelope
  flicker                     sustained 8-25 Hz AM           (B) modulation
  harmonics (3rd/5th/7th)     energy at 150/250/350 Hz       (C) harmonic
  oscillatory transient       damped burst, 300-900 Hz       (D) OT band
  notch                       periodic HF bursts at c*f      (E) HF band
  impulse                     one 1 ms HF burst              (E) HF band

Group (C) additionally carries the *time profile* of the 3rd harmonic. This is
what separates "harmonics present throughout" (classes 7/8/9/18/19/24/25) from
"harmonics present only inside the sag/swell window" (15/16/20/21/22/23/28/29)
-- the single most important discrimination in this taxonomy.

An explicit estimated-SNR feature is included so downstream models can adapt
their decision rule to the noise level (this is the "SNR as a feature" input
required by the stacking design).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

EPS = 1e-12


# ==========================================================================
# Stockwell transform
# ==========================================================================
class STransform:
    """
    Band-limited discrete Stockwell transform.

    S[n, m] = sum_k  X[(k+n) mod N] * exp(-2*pi^2*k^2/n^2) * exp(i*2*pi*k*m/N)

    (Stockwell, Mansinha & Lowe 1996). Row n corresponds to frequency
    n*fs/N Hz; row 0 is the DC/mean row.

    Only rows [0, n_max] are computed. Restricting the band is what makes the
    transform affordable: the full N x N matrix is O(N^2) in memory, while the
    band-limited version is O(n_max * N). Windows and index maps are
    precomputed once and reused for every signal.
    """

    def __init__(self, n_samples: int, fs: float, f_max: float):
        self.N = int(n_samples)
        self.fs = float(fs)
        self.df = self.fs / self.N
        self.n_max = int(round(f_max / self.df))
        self.bins = np.arange(self.n_max + 1)
        self.freqs = self.bins * self.df

        # periodic Gaussian windows in the frequency domain, one row per bin
        k = np.fft.fftfreq(self.N) * self.N               # 0..N/2-1, -N/2..-1
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.exp(-2.0 * np.pi ** 2 * (k[None, :] ** 2)
                       / (self.bins[:, None].astype(float) ** 2))
        W[0, :] = 0.0                                     # DC handled separately
        self.W = W

        # index map for the circular shift X[(k+n) mod N]
        self.idx = (np.arange(self.N)[None, :] + self.bins[:, None]) % self.N

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the complex S-matrix, shape (n_max+1, N)."""
        X = np.fft.fft(x)
        M = X[self.idx] * self.W
        S = np.fft.ifft(M, axis=1)
        S[0, :] = x.mean()
        return S


# ==========================================================================
# small numeric helpers
# ==========================================================================
def _moments(v: np.ndarray) -> tuple:
    """(mean, std, skewness, kurtosis) -- excess kurtosis."""
    m = v.mean()
    s = v.std()
    if s < EPS:
        return m, s, 0.0, 0.0
    z = (v - m) / s
    return m, s, float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a boolean array."""
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def _spec_entropy(p: np.ndarray) -> float:
    p = p / (p.sum() + EPS)
    return float(-(p * np.log(p + EPS)).sum() / np.log(len(p)))


def _peak_in_band(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float):
    """(peak magnitude, peak frequency) inside [lo, hi]."""
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return 0.0, 0.0
    sub = mag[m]
    j = int(np.argmax(sub))
    return float(sub[j]), float(freqs[m][j])


# ==========================================================================
# feature extractor
# ==========================================================================
class PQFeatureExtractor:
    """Compute the full feature vector for one signal."""

    def __init__(self, fs: float = 6400.0, f0: float = 50.0, n_cycles: int = 10,
                 f_max: float = 1600.0):
        self.fs, self.f0, self.n_cycles = fs, f0, n_cycles
        self.N = int(round(fs / f0 * n_cycles))
        self.spc = int(round(fs / f0))                    # samples per cycle
        self.st = STransform(self.N, fs, f_max)

        df = self.st.df
        b = lambda hz: int(round(hz / df))
        self.b_h = {k: b(k * f0) for k in (1, 3, 5, 7, 9, 11, 13)}
        self.sl_ot = slice(b(300), b(900) + 1)            # oscillatory transient
        self.sl_hf = slice(b(900), b(1600) + 1)           # notch / impulse
        self.sl_sub = slice(b(5), b(45) + 1)              # sub-fundamental
        self.sl_inter = slice(b(60), b(140) + 1)          # interharmonic probe

        # FFT axis for the raw signal
        self.rfreqs = np.fft.rfftfreq(self.N, 1.0 / fs)
        self.harm_bins = np.array([int(round(k * f0 / (fs / self.N)))
                                   for k in range(1, int(fs / 2 / f0))])
        self.harm_bins = self.harm_bins[self.harm_bins < len(self.rfreqs)]

        # modulation-domain axis (FFT of the fundamental envelope)
        self.mfreqs = np.fft.rfftfreq(self.N, 1.0 / fs)

        # ---- precomputed projection basis for coherent flicker detection ----
        # The S-transform window at frequency f has sigma_t = 1/f, hence
        # sigma_f = f/(2*pi). So row 50 Hz has only ~8 Hz of modulation
        # bandwidth and ANNIHILATES 8-25 Hz flicker, while rows 150 Hz (24 Hz)
        # and 250 Hz (40 Hz) preserve it. Flicker must therefore be read off
        # the Hilbert envelope (when it modulates the carrier) or off the
        # harmonic rows (when it modulates only the sag-gated harmonic term).
        self.ff_grid = np.arange(8.0, 25.25, 0.25)
        w = 2 * np.pi * np.outer(self.ff_grid, np.arange(self.N) / fs)
        self._cos, self._sin = np.cos(w), np.sin(w)

        self.names: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _sweep(self, v, mask=None):
        """
        Max normalised coherent projection of v onto exp(i*2*pi*ff*t) over the
        8-25 Hz flicker band -- the practical stand-in for a matched filter.
        Operating coherently (rather than on a periodogram) is what recovers
        the ~30 dB of processing gain available over a 1280-sample window.
        Returns (amplitude, frequency).
        """
        if mask is None:
            c, s, vv = self._cos, self._sin, v
        else:
            n = int(mask.sum())
            if n < self.spc:
                return 0.0, 0.0
            c, s, vv = self._cos[:, mask], self._sin[:, mask], v[mask]
        vv = vv - vv.mean()
        mag = np.hypot(c @ vv, s @ vv) * 2.0 / vv.size
        j = int(np.argmax(mag))
        return float(mag[j]), float(self.ff_grid[j])

    # ---------------------------------------------------------------- #
    def __call__(self, x: np.ndarray) -> np.ndarray:
        f: dict[str, float] = {}
        S = self.st(np.asarray(x, dtype=np.float64))
        Sm = np.abs(S)
        t = np.arange(self.N) / self.fs

        # =============================================================
        # (A) fundamental amplitude envelope -- sag / swell / interruption
        # =============================================================
        env = Sm[self.b_h[1], :] * 2.0          # ~ instantaneous amplitude, pu
        med = float(np.median(env)) + EPS
        rel = env / med                         # amplitude-normalised envelope

        m, s, sk, ku = _moments(env)
        f.update(env_mean=m, env_std=s, env_skew=sk, env_kurt=ku)
        f["env_min"] = float(env.min())
        f["env_max"] = float(env.max())
        f["env_med"] = med
        f["env_range"] = float(env.max() - env.min())
        f["env_p05"] = float(np.percentile(env, 5))
        f["env_p95"] = float(np.percentile(env, 95))
        f["env_iqr"] = float(np.percentile(env, 75) - np.percentile(env, 25))
        f["env_cv"] = s / (m + EPS)

        f["rel_min"] = float(rel.min())          # sag depth
        f["rel_max"] = float(rel.max())          # swell height
        f["rel_range"] = f["rel_max"] - f["rel_min"]
        for thr in (0.95, 0.90, 0.80, 0.50, 0.20, 0.10):
            f[f"frac_below_{thr}"] = float((rel < thr).mean())
            f[f"run_below_{thr}"] = _longest_run(rel < thr) / self.N
        for thr in (1.05, 1.10, 1.20, 1.40):
            f[f"frac_above_{thr}"] = float((rel > thr).mean())
            f[f"run_above_{thr}"] = _longest_run(rel > thr) / self.N

        d1 = np.diff(env)
        f["env_d1_std"] = float(d1.std())
        f["env_d1_absmax"] = float(np.abs(d1).max())
        f["env_d1_absmean"] = float(np.abs(d1).mean())
        # rectangular (sag/swell) envelopes are edge-dominated; sinusoidal
        # (flicker) envelopes are smooth -> this ratio separates them
        f["env_edge_ratio"] = f["env_d1_absmax"] / (f["env_d1_absmean"] + EPS)

        # bimodality: a sag/swell envelope has two plateaus, flicker has none
        h, _ = np.histogram(rel, bins=20, range=(0.0, 2.0), density=True)
        f["env_hist_entropy"] = _spec_entropy(h)
        f["env_hist_peak"] = float(h.max()) / (h.sum() + EPS)

        # =============================================================
        # (B) flicker -- coherent detection at full bandwidth
        # =============================================================
        # Modulation spectrum of the S-envelope. Kept because it separates
        # flicker-only classes, but note it CANNOT see flicker on its own:
        # the 50 Hz S-row is an ~8 Hz low-pass, so it attenuates the 8-25 Hz
        # band by up to ~140x. The coherent detectors below do the real work.
        ac = rel - rel.mean()
        M = np.abs(np.fft.rfft(ac * np.hanning(self.N))) / self.N
        mf = self.mfreqs
        e_tot = float((M ** 2).sum()) + EPS
        f["mod_energy"] = e_tot
        for lo, hi, nm in ((0.0, 8.0, "lo"), (8.0, 25.0, "flick"),
                           (25.0, 60.0, "mid"), (60.0, 400.0, "hi")):
            bm = (mf >= lo) & (mf < hi)
            f[f"mod_frac_{nm}"] = float((M[bm] ** 2).sum()) / e_tot
        pk, pf = _peak_in_band(M, mf, 8.0, 25.0)
        f["mod_flick_peak"] = pk
        f["mod_flick_freq"] = pf
        f["mod_flick_prom"] = pk / (M[(mf >= 0) & (mf < 400)].mean() + EPS)
        f["env_zcr"] = float((np.diff(np.sign(ac)) != 0).mean())

        # ---- event geometry, from the (flicker-free) S-envelope -----------
        # Using the S-envelope for event detection is deliberate: because it
        # has already low-passed the flicker away, it isolates the rectangular
        # sag/swell component cleanly.
        ev = np.abs(rel - 1.0) > 0.08
        g = self.spc
        ev_dil = np.convolve(ev.astype(float), np.ones(2 * g + 1), "same") > 0
        ev_ero = np.convolve(ev.astype(float), np.ones(g + 1), "same") > g
        outside = ~ev_dil
        f["ev_frac"] = float(ev.mean())
        f["ev_core_frac"] = float(ev_ero.mean())
        f["ev_n"] = float(np.count_nonzero(np.diff(ev.astype(np.int8)) == 1))

        # ---- full-bandwidth amplitude envelope ---------------------------
        h_env = np.abs(hilbert(np.asarray(x, dtype=np.float64)))
        h_norm = h_env / (h_env.mean() + EPS)

        # GLOBAL flicker: modulates the whole carrier, so it is visible on the
        # Hilbert envelope everywhere -- including outside the sag/swell, where
        # the rectangle contributes nothing to confuse the detector.
        a, fr_ = self._sweep(h_norm)
        f["flk_h1_all"], f["flk_h1_all_f"] = a, fr_
        a_out, fr_out = self._sweep(h_norm, outside)
        f["flk_h1_out"], f["flk_h1_out_f"] = a_out, fr_out
        f["flk_h1_out_avail"] = float(outside.mean())

        # rectangle cancelled by division: h_env carries rectangle+flicker,
        # the S-envelope carries the rectangle alone, so the ratio is ~flicker
        ratio = h_env / (env + EPS)
        a_r, fr_r = self._sweep(ratio / (np.abs(ratio).mean() + EPS), outside)
        f["flk_ratio"], f["flk_ratio_f"] = a_r, fr_r

        # GATED flicker: modulates only the sag-gated harmonic term, so it is
        # invisible outside the event and invisible on the fundamental. It has
        # to be read off the harmonic rows INSIDE the event core (eroded to
        # keep the S-transform's edge smearing out of the measurement).
        for k in (3, 5):
            hk = Sm[self.b_h[k], :]
            hk_n = hk / (hk.mean() + EPS)
            a_all, f_all = self._sweep(hk_n)
            a_in, f_in = self._sweep(hk_n, ev_ero)
            f[f"flk_h{k}_all"], f[f"flk_h{k}_all_f"] = a_all, f_all
            f[f"flk_h{k}_in"], f[f"flk_h{k}_in_f"] = a_in, f_in

        # frequency agreement: a real flicker line sits at the SAME ff on the
        # fundamental and on the harmonics; noise peaks do not agree
        f["flk_agree_13"] = -abs(f["flk_h1_all_f"] - f["flk_h3_all_f"])
        f["flk_agree_35"] = -abs(f["flk_h3_all_f"] - f["flk_h5_all_f"])
        f["flk_agree_in"] = -abs(f["flk_h3_in_f"] - f["flk_h5_in_f"])
        f["flk_max"] = max(f["flk_h1_all"], f["flk_h1_out"],
                           f["flk_h3_in"], f["flk_h5_in"])

        # =============================================================
        # (C) harmonics -- magnitude AND time profile
        # =============================================================
        h1_t = Sm[self.b_h[1], :]
        h1 = float(h1_t.mean()) + EPS
        for k in (3, 5, 7, 9, 11, 13):
            hk_t = Sm[self.b_h[k], :]
            f[f"h{k}_mean"] = float(hk_t.mean())
            f[f"h{k}_ratio"] = float(hk_t.mean()) / h1
            f[f"h{k}_max"] = float(hk_t.max())
            f[f"h{k}_std"] = float(hk_t.std())
            # coefficient of variation over time: ~0 if the harmonic is present
            # throughout, large if it only exists inside the sag/swell window
            f[f"h{k}_cv"] = float(hk_t.std()) / (float(hk_t.mean()) + EPS)
        f["h1_mean"] = h1

        h3_t = Sm[self.b_h[3], :]
        h5_t = Sm[self.b_h[5], :]
        # correlation of the 3rd-harmonic envelope with the fundamental envelope:
        # negative when harmonics appear only during a sag, positive during a swell
        for nm, v in (("h3", h3_t), ("h5", h5_t)):
            a, bb = v - v.mean(), h1_t - h1_t.mean()
            f[f"corr_{nm}_h1"] = float((a * bb).sum()
                                       / (np.sqrt((a ** 2).sum() * (bb ** 2).sum()) + EPS))
        f["h3_gated"] = float((h3_t > 0.5 * h3_t.max()).mean())
        f["h5_gated"] = float((h5_t > 0.5 * h5_t.max()).mean())

        # odd-harmonic total from the S-matrix
        odd = [self.b_h[k] for k in (3, 5, 7, 9, 11, 13)]
        f["harm_total"] = float(Sm[odd, :].mean())
        f["harm_total_ratio"] = f["harm_total"] / h1

        # =============================================================
        # (D) oscillatory transient band, 300-900 Hz
        # =============================================================
        ot = (Sm[self.sl_ot, :] ** 2).sum(0)
        ot_med = float(np.median(ot)) + EPS
        f["ot_mean"] = float(ot.mean())
        f["ot_max"] = float(ot.max())
        f["ot_med"] = ot_med
        f["ot_crest"] = float(ot.max()) / (float(ot.mean()) + EPS)
        f["ot_peak_over_med"] = float(ot.max()) / ot_med
        _, _, f["ot_skew"], f["ot_kurt"] = _moments(ot)
        f["ot_argmax"] = float(np.argmax(ot)) / self.N
        for thr in (3.0, 10.0, 30.0):
            f[f"ot_run_{int(thr)}"] = _longest_run(ot > thr * ot_med) / self.N
            f[f"ot_frac_{int(thr)}"] = float((ot > thr * ot_med).mean())
        # dominant transient frequency and its sharpness
        prof = Sm[self.sl_ot, :].max(1)
        j = int(np.argmax(prof))
        f["ot_dom_freq"] = float(self.st.freqs[self.sl_ot][j])
        f["ot_dom_mag"] = float(prof[j])
        f["ot_dom_sharp"] = float(prof[j]) / (float(prof.mean()) + EPS)
        # exponential decay rate after the peak (proxy for 1/tau)
        i0 = int(np.argmax(ot))
        seg = ot[i0:i0 + self.spc]
        if len(seg) > 8 and seg[0] > 10 * ot_med:
            y = np.log(seg + EPS)
            xx = np.arange(len(seg)) / self.fs
            f["ot_decay"] = float(-np.polyfit(xx, y, 1)[0])
        else:
            f["ot_decay"] = 0.0

        # =============================================================
        # (E) high-frequency band, 900-1600 Hz -- notch and impulse
        # =============================================================
        hf = (Sm[self.sl_hf, :] ** 2).sum(0)
        hf_med = float(np.median(hf)) + EPS
        f["hf_mean"] = float(hf.mean())
        f["hf_max"] = float(hf.max())
        f["hf_crest"] = float(hf.max()) / (float(hf.mean()) + EPS)
        f["hf_peak_over_med"] = float(hf.max()) / hf_med
        _, _, f["hf_skew"], f["hf_kurt"] = _moments(hf)
        for thr in (3.0, 10.0):
            f[f"hf_frac_{int(thr)}"] = float((hf > thr * hf_med).mean())
            f[f"hf_run_{int(thr)}"] = _longest_run(hf > thr * hf_med) / self.N
        # periodicity of the HF bursts: notches repeat at c*f0 (50/100/200/300 Hz),
        # an impulse is a single aperiodic event -> flat modulation spectrum
        hfa = hf - hf.mean()
        HF = np.abs(np.fft.rfft(hfa * np.hanning(self.N)))
        hf_tot = float((HF ** 2).sum()) + EPS
        pk, pf = _peak_in_band(HF, mf, 40.0, 350.0)
        f["hf_perio_peak"] = pk / (np.sqrt(hf_tot) + EPS)
        f["hf_perio_freq"] = pf
        f["hf_perio_ratio"] = pf / self.f0                # ~= notches per cycle
        f["hf_perio_prom"] = pk / (HF[(mf > 0) & (mf < 400)].mean() + EPS)
        # number of distinct HF bursts
        f["hf_n_bursts"] = float(np.count_nonzero(
            np.diff((hf > 5 * hf_med).astype(np.int8)) == 1))

        # =============================================================
        # (F) time-domain features on the raw waveform
        # =============================================================
        rms = float(np.sqrt((x ** 2).mean()))
        pk_ = float(np.abs(x).max())
        f["td_rms"] = rms
        f["td_peak"] = pk_
        f["td_crest"] = pk_ / (rms + EPS)
        f["td_form"] = rms / (float(np.abs(x).mean()) + EPS)
        _, _, f["td_skew"], f["td_kurt"] = _moments(np.asarray(x))
        f["td_zcr"] = float((np.diff(np.sign(x)) != 0).mean())
        f["td_energy"] = float((x ** 2).sum()) / self.N
        f["td_ptp"] = float(x.max() - x.min())

        cyc = np.asarray(x)[: self.n_cycles * self.spc].reshape(self.n_cycles, self.spc)
        crms = np.sqrt((cyc ** 2).mean(1))
        f["cyc_rms_min"] = float(crms.min())
        f["cyc_rms_max"] = float(crms.max())
        f["cyc_rms_std"] = float(crms.std())
        f["cyc_rms_range"] = float(crms.max() - crms.min())
        f["cyc_rms_minrat"] = float(crms.min()) / (float(crms.mean()) + EPS)
        f["cyc_rms_maxrat"] = float(crms.max()) / (float(crms.mean()) + EPS)
        cpk = np.abs(cyc).max(1)
        f["cyc_pk_std"] = float(cpk.std())
        f["cyc_pk_minrat"] = float(cpk.min()) / (float(cpk.mean()) + EPS)
        f["cyc_pk_maxrat"] = float(cpk.max()) / (float(cpk.mean()) + EPS)

        for order in (1, 2):
            d = np.diff(x, n=order)
            f[f"d{order}_std"] = float(d.std())
            f[f"d{order}_absmax"] = float(np.abs(d).max())
            _, _, f[f"d{order}_skew"], f[f"d{order}_kurt"] = _moments(d)
            f[f"d{order}_crest"] = float(np.abs(d).max()) / (d.std() + EPS)

        hb, _ = np.histogram(x, bins=32, density=True)
        f["td_hist_entropy"] = _spec_entropy(hb)

        # =============================================================
        # (G) frequency-domain features on the raw waveform
        # =============================================================
        F_ = np.abs(np.fft.rfft(np.asarray(x) * np.hanning(self.N))) / (self.N / 2)
        P = F_ ** 2
        Pt = float(P.sum()) + EPS
        fr = self.rfreqs
        cen = float((fr * P).sum()) / Pt
        f["sp_centroid"] = cen
        f["sp_spread"] = float(np.sqrt(((fr - cen) ** 2 * P).sum() / Pt))
        f["sp_skew"] = float(((fr - cen) ** 3 * P).sum() / Pt) / (f["sp_spread"] ** 3 + EPS)
        f["sp_kurt"] = float(((fr - cen) ** 4 * P).sum() / Pt) / (f["sp_spread"] ** 4 + EPS)
        cs = np.cumsum(P) / Pt
        f["sp_rolloff85"] = float(fr[np.searchsorted(cs, 0.85)])
        f["sp_rolloff95"] = float(fr[np.searchsorted(cs, 0.95)])
        f["sp_flatness"] = float(np.exp(np.log(P + EPS).mean()) / (P.mean() + EPS))
        f["sp_entropy"] = _spec_entropy(P)

        f1 = float(F_[self.harm_bins[0]]) + EPS
        hpow = P[self.harm_bins]
        f["thd"] = float(np.sqrt(hpow[1:].sum())) / f1
        f["harm_frac"] = float(hpow.sum()) / Pt
        f["interharm_frac"] = 1.0 - f["harm_frac"]

        for lo, hi, nm in ((0, 45, "dc"), (45, 55, "fund"), (55, 290, "loh"),
                           (290, 910, "ot"), (910, 1600, "hf"), (1600, 3200, "vhf")):
            bm = (fr >= lo) & (fr < hi)
            f[f"bandfrac_{nm}"] = float(P[bm].sum()) / Pt

        # =============================================================
        # (H) estimated SNR -- lets downstream models adapt to noise level
        # =============================================================
        # signal = energy at the odd harmonics; noise = median power elsewhere,
        # scaled by the number of bins
        mask = np.ones(len(P), bool)
        for bi in self.harm_bins:
            mask[max(bi - 2, 0): bi + 3] = False
        noise_psd = float(np.median(P[mask])) + EPS
        sig_pow = float(hpow.sum())
        f["snr_est_db"] = 10.0 * np.log10(sig_pow / (noise_psd * mask.sum()) + EPS)
        f["noise_floor_db"] = 10.0 * np.log10(noise_psd)
        f["noise_flatness"] = float(np.exp(np.log(P[mask] + EPS).mean())
                                    / (P[mask].mean() + EPS))

        if self.names is None:
            self.names = list(f.keys())
        return np.array([f[k] for k in self.names], dtype=np.float32)

    # ---------------------------------------------------------------- #
    def feature_names(self) -> list[str]:
        if self.names is None:
            self(np.sin(2 * np.pi * self.f0 * np.arange(self.N) / self.fs))
        return list(self.names)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract features for a stack of signals, shape (n, N)."""
        self.feature_names()
        out = np.empty((X.shape[0], len(self.names)), dtype=np.float32)
        for i in range(X.shape[0]):
            out[i] = self(X[i])
        return out
