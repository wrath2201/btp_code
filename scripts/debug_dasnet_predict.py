import torch
import numpy as np
import sys
import os
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dasnet import DASNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = np.load("data/waveforms.npz")["W"]
    y = np.load("data/dataset.npz")["y"] - 1
    
    Wte = W[-5000:]
    yte = y[-5000:]
    
    model = DASNet(n_samples=W.shape[1], head="softmax", learnable_dst=True, film=True).to(device)
    state = torch.load("results/multiseed/dasnet_seed0_best.pt", map_location=device)
    model.load_state_dict(state)
    
    model.eval()
    out = []
    for a in range(0, len(Wte), 64):
        x = torch.from_numpy(Wte[a:a + 64]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            logits = model(x)
            
        print("Logits has NaNs:", torch.isnan(logits).any().item())
        out.append(model.probs(logits.float()).cpu().numpy())
    
    yp = np.vstack(out).argmax(1)
    f1 = f1_score(yte, yp, average="macro")
    print("F1:", f1)

if __name__ == "__main__":
    main()
