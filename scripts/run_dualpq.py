"""
run_dualpq.py -- train and evaluate DualPQ-Net

Implements the 4 core experiments:
  D) Simple Concatenation
  E) SNR-Conditioned Learned Gate (DualPQ-Net)
  F) Hard SNR Routing
  G) Feature-Conditioned Gate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.dualpq import DualPQNet, DualWaveDataset
from src.dasnet import classification_loss
from src.pipeline import grouped_stratified_split, level_name, levels_of

CLEAN = 999
N_CLASSES = 29


# ========================================================================== #
# Data Loader
# ========================================================================== #
def load_dual_data(wave_path, feat_path):
    if not os.path.exists(wave_path):
        raise FileNotFoundError(f"Missing {wave_path}")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing {feat_path}. Run build_dataset.py first.")
        
    dw = np.load(wave_path, allow_pickle=True)
    df = np.load(feat_path, allow_pickle=True)
    
    W = dw["W"]
    X = df["X"]
    y = dw["y"].astype(int)
    group = dw["group"].astype(int)
    snr = dw["snr"].astype(int)
    
    assert W.shape[0] == X.shape[0], "Row count mismatch between W and X"
    assert np.all(y == df["y"].astype(int)), "Label mismatch"
    assert np.all(group == df["group"].astype(int)), "Group mismatch"
    assert np.all(snr == df["snr"].astype(int)), "SNR mismatch"
    
    return W, X, y, group, snr


# ========================================================================== #
# Thermal Governor
# ========================================================================== #
class ThermalGovernor:
    def __init__(self, target_c=74, poll=8, max_pause=2.0, enabled=True):
        self.target = target_c
        self.poll = poll
        self.max_pause = max_pause
        import torch
        self.enabled = enabled and torch.cuda.is_available()
        self.pause = 0.15 if self.enabled else 0.0
        self.temp = None
        self._n = 0

    @staticmethod
    def check_memory():
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            mem_total = next(int(line.split()[1]) for line in lines if line.startswith('MemTotal:'))
            mem_avail = next(int(line.split()[1]) for line in lines if line.startswith('MemAvailable:'))
            return mem_avail / mem_total
        except Exception:
            return 1.0

    @staticmethod
    def read_temp():
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def step(self):
        if not self.enabled:
            return
        self._n += 1
        if self._n % self.poll == 0:
            while self.check_memory() < 0.10:
                import time
                print("WARNING: RAM available < 10%. Pausing for 60s to prevent system hang...", flush=True)
                time.sleep(60.0)
            t = self.read_temp()
            if t is not None:
                self.temp = t
                if t > self.target:
                    self.pause = min(self.max_pause,
                                     self.pause + 0.03 * (t - self.target))
                elif t < self.target - 3:
                    self.pause = max(0.0, self.pause - 0.015)
        if self.pause > 0:
            import torch
            import time
            torch.cuda.synchronize()
            time.sleep(self.pause)


# ========================================================================== #
# Evaluation
# ========================================================================== #
@torch.no_grad()
def predict(model, W, X, device, batch=64, amp=True, gov=None):
    model.eval()
    out_probs = []
    out_gates = []
    
    for a in range(0, len(W), batch):
        w = torch.from_numpy(W[a:a + batch]).to(device)
        x = torch.from_numpy(X[a:a + batch]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits, g_val = model(w, x, classical_only=getattr(model, "classical_only_mode", False))
            probs = model.probs(logits.float()).cpu().numpy()
            
            # g_val is None for "concat" gate type
            if g_val is not None:
                g_val = g_val.float().cpu().numpy()
            else:
                g_val = np.zeros((w.shape[0], 1))
                
        out_probs.append(probs)
        out_gates.append(g_val)
        if gov is not None:
            gov.step()
            
    return np.vstack(out_probs), np.vstack(out_gates)


def scores(y_true, y_pred):
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float((y_true == y_pred).mean())}


def per_snr(y_true, y_pred, snr):
    return {int(s): scores(y_true[snr == s], y_pred[snr == s])
            for s in sorted(set(snr.tolist()), reverse=True)}


# ========================================================================== #
# Main
# ========================================================================== #
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}")

    W, X, y, group, snr = load_dual_data(args.data_wave, args.data_feat)
    n_samples = W.shape[1]
    
    # ---- Identical Split ------------------------------------------------- #
    (i_tr, i_va, i_te), (g_tr, g_va, g_te) = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), args.split_seed)
        
    if args.pilot:
        rng = np.random.default_rng(0)
        keep = []
        for part in (g_tr, g_va, g_te):
            cls = {}
            for g in part:
                cls.setdefault(int(y[group == g][0]), []).append(g)
            for c, gs in cls.items():
                keep += list(rng.choice(gs, min(args.limit_groups, len(gs)),
                                        replace=False))
        keep = set(int(k) for k in keep)
        f = lambda ix: ix[np.isin(group[ix], list(keep))]
        i_tr, i_va, i_te = f(i_tr), f(i_va), f(i_te)
        args.epochs = min(args.epochs, 8)

    print(f"[split] rows train={len(i_tr)} val={len(i_va)} test={len(i_te)}")

    # ---- Strict Normalization (No Leakage) ------------------------------- #
    scaler = StandardScaler()
    X[i_tr] = scaler.fit_transform(X[i_tr])
    X[i_va] = scaler.transform(X[i_va])
    X[i_te] = scaler.transform(X[i_te])

    clean_row = {int(g): W[i] for i, g in enumerate(group) if snr[i] == CLEAN}

    ds_tr = DualWaveDataset(W[i_tr], X[i_tr], y[i_tr], snr[i_tr], group[i_tr],
                            clean_row, augment=not args.no_aug, p_aug=args.p_aug)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       drop_last=True, persistent_workers=args.workers > 0)

    # ---- Model ----------------------------------------------------------- #
    model = DualPQNet(gate_type=args.gate, n_samples=n_samples, 
                      head=args.head, learnable_dst=True, film=True).to(device)
    model.classical_only_mode = getattr(args, "classical_only", False)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DualPQNet gate={args.gate} ({n_par/1e6:.2f}M params)")

    gov = ThermalGovernor(target_c=args.max_temp, enabled=not args.no_throttle)
    
    # We will use the same learning rate for the classical MLP and CNN backbone
    dst_params = list(model.deep_expert.dasnet.dst.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("deep_expert.dasnet.dst.")]
    opt = torch.optim.AdamW(
        [{"params": [p for p in dst_params if p.requires_grad],
          "lr": args.lr_dst, "weight_decay": 0.0},
         {"params": other, "lr": args.lr, "weight_decay": 1e-4}])
         
    steps_per_epoch = max(len(dl_tr), 1)
    warm = args.warmup * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(total - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler_amp = torch.amp.GradScaler(enabled=not args.no_amp and device.type == "cuda")

    # ---- Train ----------------------------------------------------------- #
    Wva, Xva, yva, sva = W[i_va], X[i_va], y[i_va], snr[i_va]
    Wte, Xte, yte, ste = W[i_te], X[i_te], y[i_te], snr[i_te]
    best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history = []
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        t0, tot_loss, n_seen = time.perf_counter(), 0.0, 0
        for w, x, t in dl_tr:
            w = w.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=not args.no_amp and device.type == "cuda"):
                logits, _ = model(w, x, classical_only=args.classical_only)
                # Hack: classification_loss expects a DASNet instance for the DST regularizer
                loss = classification_loss(model.deep_expert.dasnet, logits.float(), t, epoch)
                
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            sched.step()
            tot_loss += float(loss) * len(t)
            n_seen += len(t)
            gov.step()

        pva, _ = predict(model, Wva, Xva, device, args.batch, amp=not args.no_amp, gov=gov)
        f1v = f1_score(yva, pva.argmax(1) + 1, average="macro")
        law = model.deep_expert.dasnet.dst.law_summary()
        dt = time.perf_counter() - t0
        
        gtxt = f"  gpu={gov.temp}C pause={gov.pause*1e3:.0f}ms" if gov.enabled and gov.temp is not None else ""
        print(f"epoch {epoch+1:>3}/{args.epochs}  loss={tot_loss/max(n_seen,1):.4f}  val F1={f1v:.4f}  "
              f"dst(c={law['c']:.3f}, p={law['p']:.3f})  [{dt:.0f}s]{gtxt}", flush=True)
              
        history.append({"epoch": epoch + 1, "val_macro_f1": float(f1v)})

        if f1v > best_f1:
            best_f1, best_epoch, patience = f1v, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at epoch {epoch+1} (best val F1 {best_f1:.4f})")
                break

    # ---- Test ------------------------------------------------------------- #
    model.load_state_dict(best_state)
    pva, gva = predict(model, Wva, Xva, device, args.batch, amp=not args.no_amp, gov=gov)
    pte, gte = predict(model, Wte, Xte, device, args.batch, amp=not args.no_amp, gov=gov)
    ypv, ypt = pva.argmax(1) + 1, pte.argmax(1) + 1

    # Compute gate statistics by SNR
    gate_stats = {}
    if args.gate != "concat":
        for s in set(ste.tolist()):
            mask = ste == s
            g_snr = gte[mask]
            gate_stats[int(s)] = {
                "mean": float(np.mean(g_snr)),
                "median": float(np.median(g_snr)),
                "std": float(np.std(g_snr))
            }
        print("\n[gate] Learned Deep Expert Weight (g) by SNR:")
        for s in sorted(gate_stats.keys(), reverse=True):
            print(f"       SNR {level_name(s):>5}: mean={gate_stats[s]['mean']:.3f} ± {gate_stats[s]['std']:.3f}")

    results = {
        "model": "dualpq",
        "config": vars(args) | {"n_params": int(n_par), "best_epoch": best_epoch,
                                "wall_min": (time.perf_counter()-t_start)/60},
        "val": scores(yva, ypv),
        "test": scores(yte, ypt),
        "test_per_snr": per_snr(yte, ypt, ste),
        "gate_stats": gate_stats
    }

    LV = levels_of(ste)
    print("\n[eval] test macro-F1 by SNR level")
    print("       model          " + "".join(f"{level_name(s):>10}" for s in LV) + "     all")
    row = "".join(f"{results['test_per_snr'][s]['macro_f1']:>10.4f}" for s in LV)
    print(f"       dualpq        {row}{results['test']['macro_f1']:>8.4f}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    
    np.savez_compressed(args.out.replace(".json", "_preds.npz"),
                        yte=yte, ste=ste, yp=ypt, P=pte, gte=gte)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["concat", "snr_learned", "snr_hard", "feature_learned"])
    ap.add_argument("--data-wave", default="data/waveforms.npz")
    ap.add_argument("--data-feat", default="data/dataset.npz")
    ap.add_argument("--out", default="results/dualpq_results.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-dst", type=float, default=5e-3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--head", default="softmax")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--p-aug", type=float, default=0.5)
    ap.add_argument("--limit-groups", type=int, default=30)
    ap.add_argument("--max-temp", type=int, default=74)
    ap.add_argument("--no-throttle", action="store_true")
    ap.add_argument("--classical-only", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    
    if args.pilot:
        args.out = args.out.replace(".json", "_pilot.json")
        
    run(args)
