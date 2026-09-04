"""
dualpq.py -- DualPQ-Net: Deep Learning Hybrid Architecture

Implements the Dual-Path Power Quality Network which combines:
  1. A Deep Expert (Differentiable Stockwell Transform + CNN)
  2. A Classical Expert (Multi-Layer Perceptron on 191 Handcrafted Features)
  3. An SNR-Conditioned learned routing mechanism.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.dasnet import DASNet, SNREstimator
from src.dst import DifferentiableSTransform

N_CLASSES = 29


# ========================================================================== #
# Dataset Loader
# ========================================================================== #
class DualWaveDataset(Dataset):
    """
    Loads both Raw Waveforms (W) and 191 Handcrafted Features (X) 
    with perfect row-alignment.
    """
    def __init__(self, W, X, y, snr, group, clean_row_of_group=None,
                 augment=False, p_aug=0.5):
        self.W = W                      # (n, 1280) float32
        self.X = X                      # (n, 191) float32, normalized!
        self.y = y.astype(np.int64) - 1 # 0-indexed
        self.snr = snr
        self.group = group
        self.clean_row = clean_row_of_group
        self.augment = augment
        self.p_aug = p_aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w = self.W[i]
        x_feat = self.X[i]
        
        # Continuous SNR augmentation (same as DASNet, applied only to waveforms)
        # Note: We do NOT augment the classical features dynamically here because 
        # that requires re-running the heavy signal processing transforms. 
        # This is scientifically fair because the CNN is penalized with harder
        # waveforms while the features remain frozen at their discrete SNR level.
        if self.augment:
            if not hasattr(self, '_rng'):
                info = torch.utils.data.get_worker_info()
                self._rng = np.random.default_rng(info.seed if info else None)

            if self._rng.random() < self.p_aug:
                wc = self.clean_row[int(self.group[i])]
                snr_db = self._rng.uniform(0.0, 40.0)
                pw = float(np.mean(wc.astype(np.float64) ** 2))
                sd = math.sqrt(pw / (10.0 ** (snr_db / 10.0)))
                w = (wc + self._rng.normal(0.0, sd, wc.shape)).astype(np.float32)
            if self._rng.random() < 0.5:
                w = -w
                
        w_tensor = torch.from_numpy(np.ascontiguousarray(w))
        x_tensor = torch.from_numpy(np.ascontiguousarray(x_feat))
        
        return w_tensor, x_tensor, self.y[i]


# ========================================================================== #
# Network Components
# ========================================================================== #

class DeepExpert(nn.Module):
    """
    Path 1: Uses the Differentiable Stockwell Transform and FiLM ResNet.
    Outputs a 256-dimensional embedding.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Instantiate the full DASNet but we will bypass its final FC layer
        self.dasnet = DASNet(**kwargs)
        self.out_dim = 256
        
    def forward(self, x):
        snr = self.dasnet.snr_est(x)[:, None] / 40.0
        cond = self.dasnet.cond_mlp(snr)
        if not self.dasnet.film_on:
            cond = torch.zeros_like(cond)
            
        h = self.dasnet._tf_image(x)
        for st in self.dasnet.stages:
            h = st(h, cond)
            
        h = h.mean(dim=(2, 3)) # Global Average Pooling -> (B, 256)
        # We apply dropout but DO NOT pass through self.dasnet.fc
        return self.dasnet.dropout(h)


class ClassicalExpert(nn.Module):
    """
    Path 2: Uses a Multi-Layer Perceptron on the 191 handcrafted features.
    Outputs a 256-dimensional embedding.
    """
    def __init__(self, in_features=191, hidden_dim=512, out_dim=256, dropout=0.15):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.mlp(x)


class SNRGate(nn.Module):
    """
    Learns to output a weight between [0, 1] based on measured SNR.
    1 -> Trust Deep Expert
    0 -> Trust Classical Expert
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, snr):
        return self.net(snr)


class FeatureGate(nn.Module):
    """
    Alternative Gate for Experiment G: Learns weight based on features, 
    without explicit SNR.
    """
    def __init__(self, in_dim=512, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z_concat):
        return self.net(z_concat)


# ========================================================================== #
# DualPQ-Net Master Architecture
# ========================================================================== #

class DualPQNet(nn.Module):
    def __init__(self, gate_type="snr_learned", n_samples=1280, fs=6400.0, f0=50.0, **kwargs):
        """
        gate_type can be:
          - "concat"          : Experiment D (Simple concatenation)
          - "snr_learned"     : Experiment E (Proposed: SNR-conditioned learned gate)
          - "snr_hard"        : Experiment F (Hard heuristic: g=1 if SNR>20 else g=0)
          - "feature_learned" : Experiment G (Learned gate without explicit SNR)
        """
        super().__init__()
        self.gate_type = gate_type
        
        self.deep_expert = DeepExpert(**kwargs)
        self.classical_expert = ClassicalExpert(in_features=191, out_dim=256)
        
        # SNREstimator from the waveform (to be used by the gate)
        self.snr_est = SNREstimator(n_samples, fs, f0)
        
        if gate_type == "snr_learned":
            self.gate = SNRGate()
            final_dim = 256
        elif gate_type == "feature_learned":
            self.gate = FeatureGate(in_dim=512) # Concat of 256 + 256
            final_dim = 256
        elif gate_type == "concat":
            self.gate = None
            final_dim = 512
        elif gate_type == "snr_hard":
            self.gate = None
            final_dim = 256
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
            
        self.fc = nn.Linear(final_dim, N_CLASSES)

    def forward(self, w, x_feat, classical_only=False):
        # 1. Get representations
        if classical_only:
            # Skip the heavy CNN forward pass entirely
            z_deep = torch.zeros(w.shape[0], 256, device=w.device)
        else:
            z_deep = self.deep_expert(w)          # (B, 256)
            
        z_class = self.classical_expert(x_feat) # (B, 256)
        
        # 2. Gate & Fuse
        g_val = None
        
        if self.gate_type == "concat":
            z_fused = torch.cat([z_deep, z_class], dim=-1)
            
        elif self.gate_type == "snr_hard":
            # Estimated SNR in dB
            measured_snr_db = self.snr_est(w)[:, None]
            # Hard routing: 1 if SNR > 20, else 0
            g_val = (measured_snr_db > 20.0).float()
            z_fused = (g_val * z_deep) + ((1.0 - g_val) * z_class)
            
        elif self.gate_type == "snr_learned":
            # Pass normalized SNR to gate for better gradient flow
            measured_snr_db = self.snr_est(w)[:, None]
            norm_snr = measured_snr_db / 40.0
            g_val = self.gate(norm_snr)
            z_fused = (g_val * z_deep) + ((1.0 - g_val) * z_class)
            
        elif self.gate_type == "feature_learned":
            z_concat = torch.cat([z_deep, z_class], dim=-1)
            g_val = self.gate(z_concat)
            z_fused = (g_val * z_deep) + ((1.0 - g_val) * z_class)
            
        # 3. Classify
        logits = self.fc(z_fused)
        
        return logits, g_val

    def probs(self, logits):
        return logits.softmax(-1)
