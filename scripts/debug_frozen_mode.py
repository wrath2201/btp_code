import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dasnet import DASNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = np.load("data/waveforms.npz")["W"]
    W = torch.from_numpy(W[:64]).float().to(device)
    
    model = DASNet(n_samples=W.shape[1], head="softmax", learnable_dst=True, film=True).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.load_state_dict(state)
    
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
        model.train()
        logits_train = model(W)
        print("Train mode NaNs:", torch.isnan(logits_train).any().item())
        
        model.eval()
        logits_eval = model(W)
        print("Eval mode NaNs:", torch.isnan(logits_eval).any().item())
        
if __name__ == "__main__":
    main()
