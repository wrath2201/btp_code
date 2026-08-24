"""
dasnet.py -- Differentiable Adaptive Stockwell Network.

Architecture (waveform -> class), three components, each tied to a documented
failure mode of the fixed-feature ensemble in this repo:

 1. DST front-end (src/dst.py): learnable Stockwell window law. Motivated by
    the S-transform flicker blindness (experiments/exp_flicker*.py).
 2. SNR-conditioned CNN: a compact 2D CNN over the log TF map whose blocks are
    modulated by FiLM (Perez et al. 2018) from a *measured* SNR estimate --
    the same snr_est_db definition as features.py group (H), recomputed from
    the input waveform so it stays consistent under noise augmentation.
    Motivated by the 0.21 F1 unseen-SNR extrapolation gap (unseen_snr.json).
 3. Head: standard softmax (label smoothing) or an evidential Dirichlet head
    (Sensoy et al. 2018) whose uncertainty can be checked against the
    matched-filter ceilings of the four near-degenerate class pairs
    (experiments/exp_degeneracy.py).

Input pipeline inside the model:
    (B, 1280) waveform
      -> DST magnitude (B, 320, 1280)
      -> log compression, avg+max time pooling by 8 -> (B, 2, 320, 160)
      -> + constant frequency-coordinate channel    -> (B, 3, 320, 160)
      -> 4 FiLM-conditioned conv stages -> GAP -> head -> 29 classes
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dst import DifferentiableSTransform

N_CLASSES = 29


# ========================================================================== #
# measured SNR (torch port of features.py group (H))
# ========================================================================== #
class SNREstimator(nn.Module):
    """
    snr_est_db from the waveform itself: harmonic power over the median
    off-harmonic noise floor. No oracle information -- identical definition to
    the `snr_est_db` feature used by the stacking ensemble.
    """

    def __init__(self, n_samples: int, fs: float, f0: float):
        super().__init__()
        self.N = int(n_samples)
        n_r = self.N // 2 + 1
        df = fs / self.N

        harm = torch.tensor([int(round(k * f0 / df))
                             for k in range(1, int(fs / 2 / f0))])
        harm = harm[harm < n_r]
        mask = torch.ones(n_r, dtype=torch.bool)
        for b in harm.tolist():
            mask[max(b - 2, 0): b + 3] = False
        self.register_buffer("harm", harm)
        self.register_buffer("mask", mask)
        self.register_buffer("hann", torch.hann_window(self.N, periodic=False))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Fm = torch.fft.rfft(x.float() * self.hann, dim=-1).abs() / (self.N / 2)
        P = Fm ** 2
        noise_psd = P[:, self.mask].median(dim=-1).values + 1e-12
        sig = P[:, self.harm].sum(dim=-1)
        n_bins = int(self.mask.sum())
        return 10.0 * torch.log10(sig / (noise_psd * n_bins) + 1e-12)


# ========================================================================== #
# FiLM-conditioned convolutional stage
# ========================================================================== #
class FiLMStage(nn.Module):
    """conv-GN-FiLM-SiLU x2 with residual; stride-2 downsampling."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int, stride=(2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1,
                               bias=False)
        self.gn1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
        self.film = nn.Linear(cond_dim, 2 * c_out)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.film(cond).chunk(2, dim=-1)               # (B, C) each
        g = g[:, :, None, None]
        b = b[:, :, None, None]
        h = F.silu((1.0 + g) * self.gn1(self.conv1(x)) + b)
        h = F.silu((1.0 + g) * self.gn2(self.conv2(h)) + b)
        return h + self.skip(x)


# ========================================================================== #
# DASNet
# ========================================================================== #
class DASNet(nn.Module):
    def __init__(self, n_samples: int = 1280, fs: float = 6400.0,
                 f0: float = 50.0, f_max: float = 1600.0,
                 channels=(32, 64, 128, 256), time_pool: int = 8,
                 cond_dim: int = 64, head: str = "softmax",
                 learnable_dst: bool = True, film: bool = True,
                 dropout: float = 0.15):
        super().__init__()
        assert head in ("softmax", "evidential")
        self.head_type = head
        self.film_on = film
        self.time_pool = time_pool

        self.dst = DifferentiableSTransform(n_samples, fs, f_max,
                                            learnable=learnable_dst)
        self.snr_est = SNREstimator(n_samples, fs, f0)

        # frequency-coordinate channel (CNNs need absolute row position:
        # a harmonic at 150 Hz is not the same event as one at 750 Hz)
        rows = self.dst.n_max
        coord = (torch.arange(rows, dtype=torch.float32) / rows) * 2.0 - 1.0
        self.register_buffer("coord", coord[None, None, :, None])

        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU())

        stages, c_in = [], 3
        for c_out in channels:
            stages.append(FiLMStage(c_in, c_out, cond_dim))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, N_CLASSES)

    # ------------------------------------------------------------------ #
    def _tf_image(self, x: torch.Tensor) -> torch.Tensor:
        mag = self.dst(x)                                     # (B, R, N)
        z = torch.log1p(20.0 * mag)[:, None]                  # (B, 1, R, N)
        za = F.avg_pool2d(z, (1, self.time_pool))
        zm = F.max_pool2d(z, (1, self.time_pool))
        img = torch.cat([za, zm], dim=1)                      # (B, 2, R, T')
        c = self.coord.expand(img.shape[0], 1, -1, img.shape[-1])
        return torch.cat([img, c], dim=1)                     # (B, 3, R, T')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snr = self.snr_est(x)[:, None] / 40.0                 # ~[-0.5, 1.2]
        cond = self.cond_mlp(snr)
        if not self.film_on:
            cond = torch.zeros_like(cond)
        h = self._tf_image(x)
        for st in self.stages:
            h = st(h, cond)
        h = h.mean(dim=(2, 3))                                # GAP
        return self.fc(self.dropout(h))                       # (B, 29) logits

    # ------------------------------------------------------------------ #
    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.head_type == "softmax":
            return logits.softmax(-1)
        alpha = F.softplus(logits) + 1.0
        return alpha / alpha.sum(-1, keepdim=True)

    def uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Evidential vacuity u = K / sum(alpha); ~1 = 'I do not know'."""
        alpha = F.softplus(logits) + 1.0
        return N_CLASSES / alpha.sum(-1)


# ========================================================================== #
# losses
# ========================================================================== #
def evidential_loss(logits: torch.Tensor, target: torch.Tensor,
                    epoch: int, anneal_epochs: int = 10) -> torch.Tensor:
    """
    Sensoy et al. 2018, type-II MLE form with annealed KL regularizer.
    target: int64 class indices 0..K-1.
    """
    alpha = F.softplus(logits) + 1.0                          # (B, K)
    S = alpha.sum(-1, keepdim=True)
    y = F.one_hot(target, N_CLASSES).float()
    nll = (y * (torch.log(S) - torch.log(alpha))).sum(-1)

    lam = min(1.0, epoch / max(anneal_epochs, 1))
    alpha_t = y + (1.0 - y) * alpha                           # misleading part
    S_t = alpha_t.sum(-1, keepdim=True)
    K = float(N_CLASSES)
    kl = (torch.lgamma(S_t.squeeze(-1)) - torch.lgamma(torch.tensor(K))
          - torch.lgamma(alpha_t).sum(-1)
          + ((alpha_t - 1.0)
             * (torch.digamma(alpha_t) - torch.digamma(S_t))).sum(-1))
    return (nll + 0.1 * lam * kl).mean()


def classification_loss(model: DASNet, logits: torch.Tensor,
                        target: torch.Tensor, epoch: int,
                        delta_reg: float = 1e-3) -> torch.Tensor:
    if model.head_type == "softmax":
        loss = F.cross_entropy(logits, target, label_smoothing=0.05)
    else:
        loss = evidential_loss(logits, target, epoch)
    return loss + delta_reg * model.dst.regularizer()
