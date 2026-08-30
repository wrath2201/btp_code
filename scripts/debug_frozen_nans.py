import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dualpq import DualPQNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = np.load("data/waveforms.npz")["W"]
    W = torch.from_numpy(W[:64]).float().to(device)
    
    model = DualPQNet(gate_type="concat", n_samples=W.shape[1]).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.deep_expert.dasnet.load_state_dict(state)
    model.deep_expert.eval()
    
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
        dasnet = model.deep_expert.dasnet
        
        snr = dasnet.snr_est(W)[:, None] / 40.0
        print("SNR has NaNs:", torch.isnan(snr).any().item())
        
        cond = dasnet.cond_mlp(snr)
        print("cond has NaNs:", torch.isnan(cond).any().item())
        
        h = dasnet._tf_image(W)
        print("TF image has NaNs:", torch.isnan(h).any().item())
        
        for i, st in enumerate(dasnet.stages):
            h = st(h, cond)
            print(f"Stage {i} has NaNs:", torch.isnan(h).any().item())
            
        h_gap = h.mean(dim=(2, 3))
        print("GAP has NaNs:", torch.isnan(h_gap).any().item())

if __name__ == "__main__":
    main()
