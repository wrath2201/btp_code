"""
dst.py -- Differentiable Stockwell Transform (DST) layer.

The classical S-transform uses a frequency-dependent Gaussian window with a
FIXED width law, sigma_t(f) = 1/f. That law is exactly why the transform is
blind to flicker at the fundamental (see experiments/exp_flicker*.py): the
50 Hz row has only ~8 Hz of modulation bandwidth and annihilates 8-25 Hz AM.

This layer generalizes the window law and learns it END-TO-END by gradient
descent:

    sigma_t(f) = softplus(c) / f^p * exp(delta_f)

  * c, p     -- global power-law parameters (classical ST: c=1, p=1)
  * delta_f  -- small per-frequency-row residual (L2-regularized), which lets
                individual bands deviate from the power law, e.g. the 50 Hz
                row can SHRINK sigma_t to widen its modulation bandwidth and
                recover flicker visibility.

Implementation is the standard frequency-domain form of the discrete ST
(Stockwell, Mansinha & Lowe 1996):

    S[n, :] = IFFT_k( X[(k + n) mod N] * exp(-2*pi^2*k_hz^2*sigma_t(f_n)^2) )

which is exactly the fixed transform in src/features.py when c=1, p=1,
delta=0 (verified in tests/test_dst.py), and is differentiable w.r.t.
(c, p, delta) because they only enter through the Gaussian window.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSTransform(nn.Module):
    """
    Band-limited Stockwell transform with a learnable window-width law.

    Parameters
    ----------
    n_samples : signal length N
    fs        : sampling frequency [Hz]
    f_max     : highest analysed frequency [Hz] (rows 1 .. f_max/df)
    learnable : if False, the window law is frozen at the classical
                sigma_t = 1/f (ablation baseline)

    forward(x) : (B, N) real -> (B, n_rows, N) magnitude
    """

    def __init__(self, n_samples: int, fs: float, f_max: float = 1600.0,
                 learnable: bool = True, delta_clamp: float = 2.0):
        super().__init__()
        self.N = int(n_samples)
        self.fs = float(fs)
        self.df = self.fs / self.N
        self.n_max = int(round(f_max / self.df))
        self.learnable = learnable
        self.delta_clamp = float(delta_clamp)

        bins = torch.arange(1, self.n_max + 1)                # skip DC row
        self.register_buffer("freqs_hz", bins.to(torch.float32) * self.df)

        # circular-shift index map for X[(k + n) mod N]
        idx = (torch.arange(self.N)[None, :] + bins[:, None]) % self.N
        self.register_buffer("idx", idx)

        # frequency variable of the window's Fourier transform, in Hz
        k_hz = torch.fft.fftfreq(self.N) * self.fs
        self.register_buffer("k_hz", k_hz.to(torch.float32))

        # learnable window law; init = classical ST (c=1, p=1, delta=0)
        c0 = math.log(math.e - 1.0)                           # softplus^-1(1)
        self.c_raw = nn.Parameter(torch.tensor(c0), requires_grad=learnable)
        self.p_raw = nn.Parameter(torch.tensor(0.0), requires_grad=learnable)
        self.delta = nn.Parameter(torch.zeros(self.n_max),
                                  requires_grad=learnable)

    # ------------------------------------------------------------------ #
    def sigma_t(self) -> torch.Tensor:
        """Per-row window width sigma_t(f) in seconds, shape (n_rows,)."""
        c = F.softplus(self.c_raw)
        p = 2.0 * torch.sigmoid(self.p_raw)                   # p in (0, 2)
        d = torch.clamp(self.delta, -self.delta_clamp, self.delta_clamp)
        return c * self.freqs_hz.pow(-p) * torch.exp(d)

    def window(self) -> torch.Tensor:
        """Frequency-domain Gaussian windows, shape (n_rows, N)."""
        sig = self.sigma_t()                                  # (R,)
        return torch.exp(-2.0 * math.pi ** 2
                         * (self.k_hz[None, :] ** 2) * (sig[:, None] ** 2))

    def regularizer(self) -> torch.Tensor:
        """L2 penalty keeping per-row deviations close to the power law."""
        return (self.delta ** 2).mean()

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT in float32/complex64 regardless of autocast
        X = torch.fft.fft(x.float(), dim=-1)                  # (B, N)
        M = X[:, self.idx] * self.window().to(torch.complex64)
        S = torch.fft.ifft(M, dim=-1)                         # (B, R, N)
        return S.abs()

    # ------------------------------------------------------------------ #
    def law_summary(self) -> dict:
        """Learned law, for logging / the paper's sigma(f) figure."""
        with torch.no_grad():
            return {
                "c": float(F.softplus(self.c_raw)),
                "p": float(2.0 * torch.sigmoid(self.p_raw)),
                "freqs_hz": self.freqs_hz.cpu().numpy().tolist(),
                "sigma_t": self.sigma_t().cpu().numpy().tolist(),
                "sigma_t_classical": (1.0 / self.freqs_hz.cpu()
                                      .numpy()).tolist(),
            }


# ---------------------------------------------------------------------- #
def numpy_reference(x: np.ndarray, fs: float, f_max: float) -> np.ndarray:
    """Fixed ST via src.features.STransform, rows 1.. (for tests)."""
    from src.features import STransform
    st = STransform(len(x), fs, f_max)
    return np.abs(st(x))[1:, :]
