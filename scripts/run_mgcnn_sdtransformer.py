"""
run_mgcnn_sdtransformer.py -- train MGCNN-SDTransformer (Jiang et al., 2025).
Implementation of the paper's architecture with strict benchmarking on our 29-class dataset.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mgcnn_sdtransformer import MGCNNSDTransformer
from src.pipeline import grouped_stratified_split
import subprocess

class ThermalGovernor:
    def __init__(self, target_c=74, poll=8, max_pause=2.0, enabled=True):
        self.target = target_c
        self.poll = poll
        self.max_pause = max_pause
        self.enabled = enabled and torch.cuda.is_available()
        self.pause = 0.15 if self.enabled else 0.0
        self.temp = None
        self._n = 0

    @staticmethod
    def read_temp():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def step(self):
        if not self.enabled: return
        self._n += 1
        if self._n % self.poll == 0:
            t = self.read_temp()
            if t is not None:
                self.temp = t
                if t > self.target:
                    self.pause = min(self.max_pause, self.pause + 0.03 * (t - self.target))
                elif t < self.target - 3:
                    self.pause = max(0.0, self.pause - 0.015)
        if self.pause > 0:
            torch.cuda.synchronize()
            time.sleep(self.pause)

CLEAN = 999
N_CLASSES = 29

class PQDatasetRaw(Dataset):
    def __init__(self, w, y):
        self.w = w
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(np.ascontiguousarray(self.w[i])).float(),
                int(self.y[i]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-wave", default="data/waveforms.npz")
    parser.add_argument("--data-feat", default="data/dataset.npz")
    parser.add_argument("--out", required=True, help="Path to save JSON results")
    parser.add_argument("--save-ckpt", default=None, help="Path to save best checkpoint")
    
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)  # Paper used 64
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-temp", type=int, default=74)
    parser.add_argument("--no-throttle", action="store_true")
    
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    W = np.load(args.data_wave)["W"]
    d_feat = np.load(args.data_feat)
    y = d_feat["y"] - 1
    groups = d_feat["group"]
    snr = d_feat["snr"]

    (i_tr, i_va, i_te), _ = grouped_stratified_split(y, groups, seed=args.split_seed)
    
    ds_tr = PQDatasetRaw(W[i_tr], y[i_tr])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, 
                       num_workers=args.workers, pin_memory=True, drop_last=True)
                       
    ds_va = PQDatasetRaw(W[i_va], y[i_va])
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    print("Initializing MGCNN-SDTransformer...")
    model = MGCNNSDTransformer(num_classes=N_CLASSES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params:,}")

    gov = ThermalGovernor(target_c=args.max_temp, enabled=not args.no_throttle)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
         
    steps_per_epoch = max(len(dl_tr), 1)
    warm = args.warmup * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm: return (step + 1) / warm
        t = (step - warm) / max(total - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # Using autocast for faster training
    scaler_amp = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        for w, t in dl_tr:
            w, t = w.to(device, non_blocking=True), t.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(w)
                loss = TF.cross_entropy(logits, t)
                
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            sched.step()
            gov.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for w, t in dl_va:
                w = w.to(device)
                logits = model(w)
                preds.append(logits.argmax(1).cpu().numpy())
                trues.append(t.numpy())
                
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_f1 = f1_score(trues, preds, labels=np.arange(29), average="macro")

        print(f"Epoch {epoch:02d} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
            if args.save_ckpt:
                torch.save(best_state, args.save_ckpt)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test Evaluation
    print("Evaluating on test set...")
    model.load_state_dict(best_state)
    model.eval()
    
    ds_te = PQDatasetRaw(W[i_te], y[i_te])
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    
    preds, trues = [], []
    with torch.no_grad():
        for w, t in dl_te:
            w = w.to(device)
            logits = model(w)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(t.numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    test_f1 = f1_score(trues, preds, labels=np.arange(29), average="macro")
    
    # Per-SNR Evaluation
    res_snr = {}
    st = snr[i_te]
    for snr_val in [999, 40, 30, 20, 10, 0]:
        mask = (st == snr_val)
        if np.any(mask):
            f1_snr = f1_score(trues[mask], preds[mask], labels=np.arange(29), average="macro")
            res_snr[str(snr_val)] = {"macro_f1": float(f1_snr)}

    res = {
        "model": "mgcnn_sdtransformer",
        "config": vars(args),
        "val": {"macro_f1": float(best_f1)},
        "test": {"macro_f1": float(test_f1)},
        "test_per_snr": res_snr
    }
    res["config"]["best_epoch"] = best_epoch
    
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
        
    # Save predictions array.
    # Labels are stored in 1..29 to match src/pipeline.py and
    # scripts/run_dasnet.py, which both emit `argmax + 1`. Training uses 0..28
    # because cross_entropy requires it, so +1 is applied only on the way out.
    # Mixing the two conventions shifts every per-class attribution by one
    # class -- that is what corrupted results/per_class/summary.json.
    np.savez_compressed(
        args.out.replace(".json", "_preds.npz"),
        yte=trues + 1,
        yp=preds + 1,
        ste=st
    )
        
    print(f"Done! Test F1 = {test_f1:.4f}")

if __name__ == "__main__":
    main()
