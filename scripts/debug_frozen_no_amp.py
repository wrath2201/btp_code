import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dualpq import DualPQNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = np.load("data/waveforms.npz")["W"]
    d_feat = np.load("data/dataset.npz")
    X = d_feat["X"]
    y = d_feat["y"] - 1
    
    W = torch.from_numpy(W[:64]).float().to(device)
    X = torch.from_numpy(X[:64]).float().to(device)
    y = torch.from_numpy(y[:64]).long().to(device)
    
    model = DualPQNet(gate_type="concat", n_samples=W.shape[1]).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.deep_expert.dasnet.load_state_dict(state)
    
    for param in model.deep_expert.parameters():
        param.requires_grad = False
        
    model.train()
    model.deep_expert.eval()
    
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
    
    for i in range(5):
        opt.zero_grad()
        # NO AUTOCAST HERE
        logits, _ = model(W, X)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        print(f"Iter {i} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")

if __name__ == "__main__":
    main()
