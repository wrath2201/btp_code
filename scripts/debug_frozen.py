import torch
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dualpq import DualPQNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    W = np.load("data/waveforms.npz")["W"]
    d_feat = np.load("data/dataset.npz")
    X = d_feat["X"]
    y = d_feat["y"] - 1
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    W = torch.from_numpy(W[:64]).float().to(device)
    X = torch.from_numpy(X[:64]).float().to(device)
    
    print("W has NaNs:", torch.isnan(W).any().item())
    print("X has NaNs:", torch.isnan(X).any().item())
    
    model = DualPQNet(gate_type="concat", n_samples=W.shape[1]).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.deep_expert.dasnet.load_state_dict(state)
    
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            z_deep = model.deep_expert(W)
            print("z_deep has NaNs:", torch.isnan(z_deep).any().item())
            
            z_class = model.classical_expert(X)
            print("z_class has NaNs:", torch.isnan(z_class).any().item())
            
            logits, _ = model(W, X)
            print("logits has NaNs:", torch.isnan(logits).any().item())

if __name__ == "__main__":
    main()
