import sys, os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_dasnet import predict
from src.dasnet import DASNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = np.load("data/waveforms.npz")["W"]
    Wte = W[-64:]
    
    model = DASNet(n_samples=W.shape[1], head="softmax", learnable_dst=True, film=True).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.load_state_dict(state)
    
    p = predict(model, Wte, device, batch=64, amp=True)
    print("Has NaNs:", np.isnan(p).any())

if __name__ == "__main__":
    main()
