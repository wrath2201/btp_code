import torch
import numpy as np
import sys
import os
import json
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dasnet import DASNet
from src.pipeline import grouped_stratified_split

def run_control():
    print("--- 1. VERIFY DASNET CHECKPOINT AND CONTROL EXPERIMENT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wave_path = "data/waveforms.npz"
    feat_path = "data/dataset.npz"
    
    dw = np.load(wave_path, allow_pickle=True)
    df = np.load(feat_path, allow_pickle=True)
    W = dw["W"]
    y = dw["y"].astype(int) - 1 # 0-indexed
    group = dw["group"].astype(int)
    snr = dw["snr"].astype(int)
    
    # Same split as seed 0
    split_seed = 0
    (i_tr, i_va, i_te), _ = grouped_stratified_split(y, group, (0.70, 0.15, 0.15), split_seed)
    
    Wte = W[i_te]
    yte = y[i_te]
    
    model = DASNet(n_samples=W.shape[1], head="softmax", learnable_dst=True, film=True).to(device)
    
    ckpt_path = "results/multiseed/dasnet_seed0_best.pt"
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    
    print(f"Loaded {ckpt_path}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    model.eval()
    
    batch = 64
    out = []
    with torch.no_grad():
        for a in range(0, len(Wte), batch):
            x = torch.from_numpy(Wte[a:a + batch]).to(device)
            logits = model(x)
            out.append(logits.argmax(1).cpu().numpy())
            
    y_pred = np.concatenate(out)
    f1 = f1_score(yte, y_pred, average="macro")
    print(f"Frozen DASNet ALONE (Seed 0) Test F1: {f1:.4f}")
    
    print("\n--- 4. EXACT REPRESENTATION (TENSOR SHAPES) ---")
    x_dummy = torch.from_numpy(Wte[:2]).to(device)
    with torch.no_grad():
        # Inside DASNet
        h_tf = model._tf_image(x_dummy)
        print(f"TF Image shape: {h_tf.shape} (Expected: B, 3, 320, 160)")
        
        snr_est = model.snr_est(x_dummy)[:, None] / 40.0
        cond = model.cond_mlp(snr_est)
        
        h = h_tf
        for st in model.stages:
            h = st(h, cond)
        print(f"Post-CNN shape: {h.shape} (Expected: B, 256, H, W)")
        
        h_gap = h.mean(dim=(2, 3))
        print(f"GAP shape (z_deep): {h_gap.shape} (Expected: B, 256)")
        
        logits = model.fc(model.dropout(h_gap))
        print(f"Logits shape: {logits.shape} (Expected: B, 29)")
        
    print("\n--- 8. CHECK PREDICTIONS OF FROZEN DUALPQ ---")
    frozen_preds_path = "results/multiseed/frozen_dualpq_seed0_preds.npz"
    dp = np.load(frozen_preds_path)
    yp = dp["yp"]
    yte = dp["yte"]
    
    unique, counts = np.unique(yp, return_counts=True)
    print("Unique classes predicted by Frozen-DASNet DualPQ:")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} times")
    
    total_preds = len(yp)
    print(f"Total predictions: {total_preds}")
    
    # Are we just predicting one class?
    if len(unique) < 5:
        print("Model is predicting very few classes (severe bias).")
    elif max(counts) > total_preds * 0.5:
        print("Model is overwhelmingly predicting a single class.")
    else:
        print("Model predictions are somewhat distributed, but highly inaccurate.")
        
    print("DONE.")

if __name__ == '__main__':
    run_control()
