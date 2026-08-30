import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dasnet import DASNet
from scripts.run_dasnet import load_data, grouped_stratified_split

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, y, group, snr = load_data("data/waveforms.npz")
    (i_tr, i_va, i_te), _ = grouped_stratified_split(y, group, (0.70, 0.15, 0.15), 0)
    
    Wte = W[i_te]
    
    model = DASNet(n_samples=W.shape[1], head="softmax", learnable_dst=True, film=True).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    nans = 0
    with torch.no_grad():
        for a in range(0, len(Wte), 64):
            x = torch.from_numpy(Wte[a:a + 64]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                logits = model(x)
            if torch.isnan(logits).any():
                nans += 1
                
    print(f"NaN batches: {nans} / {len(range(0, len(Wte), 64))}")

if __name__ == "__main__":
    main()
