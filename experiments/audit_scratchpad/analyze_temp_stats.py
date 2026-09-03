import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from src.dualpq import DualPQNet
from src.pipeline import grouped_stratified_split

# Load data
W = np.load("data/waveforms.npz")["W"]
d_feat = np.load("data/dataset.npz")
X = d_feat["X"]
y = d_feat["y"] - 1
groups = d_feat["group"]

# Split
(i_tr, i_va, i_te), _ = grouped_stratified_split(y, groups, seed=0)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr = scaler.fit_transform(X[i_tr])

# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualPQNet(gate_type="concat", n_samples=1280).to(device)

# Load pretrained deep expert
state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device, weights_only=True)
model.deep_expert.dasnet.load_state_dict(state)
model.eval()

# Get a batch
w_batch = torch.from_numpy(W[i_tr][:256]).float().to(device)
x_batch = torch.from_numpy(X_tr[:256]).float().to(device)

with torch.no_grad():
    z_deep = model.deep_expert(w_batch)
    z_class = model.classical_expert(x_batch)

    print("z_deep  - mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}".format(
        z_deep.mean().item(), z_deep.std().item(), z_deep.min().item(), z_deep.max().item()
    ))
    print("z_class - mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}".format(
        z_class.mean().item(), z_class.std().item(), z_class.min().item(), z_class.max().item()
    ))
    print("z_deep L2 norm per sample: {:.4f} +- {:.4f}".format(
        torch.norm(z_deep, dim=1).mean().item(), torch.norm(z_deep, dim=1).std().item()
    ))
    print("z_class L2 norm per sample: {:.4f} +- {:.4f}".format(
        torch.norm(z_class, dim=1).mean().item(), torch.norm(z_class, dim=1).std().item()
    ))
    
    # Also per-dimension variance
    print("z_deep per-dim variance: mean {:.4f}, max {:.4f}".format(
        z_deep.var(dim=0).mean().item(), z_deep.var(dim=0).max().item()
    ))
    print("z_class per-dim variance: mean {:.4f}, max {:.4f}".format(
        z_class.var(dim=0).mean().item(), z_class.var(dim=0).max().item()
    ))
