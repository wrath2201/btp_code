"""
run_dasnet.py -- train and evaluate DASNet under the SAME leakage-free
protocol as the feature ensemble (src/pipeline.py):

  * identical 70/15/15 group-stratified split (same function, same seed), so
    DASNet trains/validates/tests on exactly the same base waveforms and the
    same frozen noise realizations as the ensemble;
  * model selection (early stopping) on the validation partition only;
  * the test partition is touched exactly once, at the end;
  * same metrics: macro-F1 (primary), balanced accuracy, accuracy, per-SNR.

Training augmentation (training groups only -- no leakage):
  * fresh AWGN at a CONTINUOUS uniform SNR in [0, 40] dB drawn per sample from
    the group's clean waveform (attacks the unseen-SNR extrapolation gap);
  * random polarity flip.

Usage
-----
  python scripts/build_waveforms.py                      # once
  python scripts/run_dasnet.py --epochs 50               # full run
  python scripts/run_dasnet.py --pilot                   # quick smoke test
Ablations:
  --no-learnable-dst   freeze sigma_t = 1/f (classical ST front-end)
  --no-film            disable SNR conditioning
  --head evidential    Dirichlet evidential head
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.dasnet import DASNet, classification_loss
from src.pipeline import grouped_stratified_split, level_name, levels_of

CLEAN = 999
N_CLASSES = 29


# ========================================================================== #
# data
# ========================================================================== #
class WaveDataset(Dataset):
    def __init__(self, W, y, snr, group, clean_row_of_group=None,
                 augment=False, p_aug=0.5):
        self.W = W                      # (n, N) float32, canonical rows
        self.y = y.astype(np.int64) - 1
        self.snr = snr
        self.group = group
        self.clean_row = clean_row_of_group   # dict group -> row into W_all
        self.augment = augment
        self.p_aug = p_aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w = self.W[i]
        if self.augment:
            if not hasattr(self, '_rng'):
                info = torch.utils.data.get_worker_info()
                self._rng = np.random.default_rng(info.seed if info else None)
            
            if self._rng.random() < self.p_aug:
                wc = self.clean_row[int(self.group[i])]
                snr_db = self._rng.uniform(0.0, 40.0)
                pw = float(np.mean(wc.astype(np.float64) ** 2))
                sd = math.sqrt(pw / (10.0 ** (snr_db / 10.0)))
                w = (wc + self._rng.normal(0.0, sd, wc.shape)).astype(np.float32)
            if self._rng.random() < 0.5:
                w = -w
        return torch.from_numpy(np.ascontiguousarray(w)), self.y[i]


def load_data(path):
    d = np.load(path, allow_pickle=True)
    return (d["W"], d["y"].astype(int), d["group"].astype(int),
            d["snr"].astype(int))


# ========================================================================== #
# thermal governor -- keeps the GPU at a safe temperature by inserting
# short idle pauses between batches. No data is skipped and nothing about
# the optimization changes; training simply proceeds more slowly when hot.
# ========================================================================== #
class ThermalGovernor:
    """
    Proportional controller: every `poll` steps read the GPU temperature and
    adjust a per-step pause so the temperature settles at `target_c`.

      temp > target      -> lengthen the pause (0.03 s per degree over)
      temp < target - 3  -> shorten it gradually

    The pause is applied after cuda.synchronize(), so the GPU is genuinely
    idle (not just the host) during the pause.
    """

    def __init__(self, target_c=74, poll=8, max_pause=2.0, enabled=True):
        self.target = target_c
        self.poll = poll
        self.max_pause = max_pause
        self.enabled = enabled and torch.cuda.is_available()
        self.pause = 0.15 if self.enabled else 0.0   # start gently
        self.temp = None
        self._n = 0

    @staticmethod
    def read_temp():
        try:
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
            t = self.read_temp()
            if t is not None:
                self.temp = t
                if t > self.target:
                    self.pause = min(self.max_pause,
                                     self.pause + 0.03 * (t - self.target))
                elif t < self.target - 3:
                    self.pause = max(0.0, self.pause - 0.015)
        if self.pause > 0:
            torch.cuda.synchronize()
            time.sleep(self.pause)


# ========================================================================== #
# evaluation
# ========================================================================== #
@torch.no_grad()
def predict(model, W, device, batch=64, amp=True, gov=None):
    model.eval()
    out = []
    for a in range(0, len(W), batch):
        x = torch.from_numpy(W[a:a + batch]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(x)
        out.append(model.probs(logits.float()).cpu().numpy())
        if gov is not None:
            gov.step()
    return np.vstack(out)


def scores(y_true, y_pred):
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float((y_true == y_pred).mean())}


def per_snr(y_true, y_pred, snr):
    return {int(s): scores(y_true[snr == s], y_pred[snr == s])
            for s in sorted(set(snr.tolist()), reverse=True)}


# ========================================================================== #
# main
# ========================================================================== #
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))

    W, y, group, snr = load_data(args.data)
    n_samples = W.shape[1]
    print(f"data: {W.shape[0]} waveforms x {n_samples} samples, "
          f"{len(np.unique(group))} groups, "
          f"levels [{', '.join(level_name(s) for s in levels_of(snr))}]")

    # ---- identical split to the ensemble run ---------------------------- #
    (i_tr, i_va, i_te), (g_tr, g_va, g_te) = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), args.split_seed)
    print(f"[split] groups train={len(g_tr)} val={len(g_va)} test={len(g_te)}"
          f"   rows train={len(i_tr)} val={len(i_va)} test={len(i_te)}")

    if args.limit_groups:                                     # pilot mode
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
        print(f"[pilot] rows train={len(i_tr)} val={len(i_va)} "
              f"test={len(i_te)}")

    clean_row = {int(g): W[i] for i, g in enumerate(group)
                 if snr[i] == CLEAN}

    ds_tr = WaveDataset(W[i_tr], y[i_tr], snr[i_tr], group[i_tr],
                        clean_row, augment=not args.no_aug, p_aug=args.p_aug)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       drop_last=True, persistent_workers=args.workers > 0)

    # ---- model ----------------------------------------------------------- #
    model = DASNet(n_samples=n_samples, head=args.head,
                   learnable_dst=not args.no_learnable_dst,
                   film=not args.no_film).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DASNet head={args.head} "
          f"learnable_dst={not args.no_learnable_dst} "
          f"film={not args.no_film}  ({n_par/1e6:.2f}M params)")

    if args.init_from and os.path.exists(args.init_from):
        model.load_state_dict(torch.load(args.init_from,
                                         map_location=device))
        print(f"[model] warm-started weights from {args.init_from}")

    gov = ThermalGovernor(target_c=args.max_temp, enabled=not args.no_throttle)
    if gov.enabled:
        print(f"[gov] thermal governor ON: target {args.max_temp}C, "
              f"current {gov.read_temp()}C")

    dst_params = list(model.dst.parameters())
    other = [p for n, p in model.named_parameters()
             if not n.startswith("dst.")]
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
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # ---- train ------------------------------------------------------------ #
    Wva, yva, sva = W[i_va], y[i_va], snr[i_va]
    Wte, yte, ste = W[i_te], y[i_te], snr[i_te]
    best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history = []
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        t0, tot_loss, n_seen = time.perf_counter(), 0.0, 0
        for x, t in dl_tr:
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(x)
                loss = classification_loss(model, logits.float(), t, epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += float(loss) * len(t)
            n_seen += len(t)
            gov.step()

        pv = predict(model, Wva, device, args.batch, gov=gov)
        f1v = f1_score(yva, pv.argmax(1) + 1, average="macro")
        law = model.dst.law_summary()
        dt = time.perf_counter() - t0
        gtxt = (f"  gpu={gov.temp}C pause={gov.pause*1e3:.0f}ms"
                if gov.enabled and gov.temp is not None else "")
        print(f"epoch {epoch+1:>3}/{args.epochs}  "
              f"loss={tot_loss/max(n_seen,1):.4f}  val F1={f1v:.4f}  "
              f"dst(c={law['c']:.3f}, p={law['p']:.3f})  [{dt:.0f}s]{gtxt}",
              flush=True)
        history.append({"epoch": epoch + 1,
                        "train_loss": tot_loss / max(n_seen, 1),
                        "val_macro_f1": float(f1v),
                        "dst_c": law["c"], "dst_p": law["p"]})

        if f1v > best_f1:
            best_f1, best_epoch, patience = f1v, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(best_state, args.out.replace(".json", "_best.pt"))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at epoch {epoch+1} "
                      f"(best val F1 {best_f1:.4f} @ {best_epoch})")
                break

    # ---- final evaluation: test touched once ------------------------------ #
    model.load_state_dict(best_state)
    pva = predict(model, Wva, device, args.batch, gov=gov)
    pte = predict(model, Wte, device, args.batch, gov=gov)
    ypv, ypt = pva.argmax(1) + 1, pte.argmax(1) + 1

    results = {
        "model": "dasnet",
        "config": vars(args) | {"n_params": int(n_par),
                                "best_epoch": best_epoch,
                                "wall_min": (time.perf_counter()-t_start)/60},
        "val": scores(yva, ypv),
        "test": scores(yte, ypt),
        "test_per_snr": per_snr(yte, ypt, ste),
        "val_per_snr": per_snr(yva, ypv, sva),
        "confusion_all": confusion_matrix(
            yte, ypt, labels=np.arange(1, 30)).tolist(),
        "dst_law": model.dst.law_summary(),
        "history": history,
    }

    LV = levels_of(ste)
    print("\n[eval] test macro-F1 by SNR level")
    print("       model          " +
          "".join(f"{level_name(s):>10}" for s in LV) + "     all")
    row = "".join(f"{results['test_per_snr'][s]['macro_f1']:>10.4f}"
                  for s in LV)
    print(f"       dasnet        {row}{results['test']['macro_f1']:>8.4f}")

    # side-by-side with the ensemble baseline, if available
    base = args.baseline
    if os.path.exists(base):
        with open(base) as fh:
            R = json.load(fh)
        sel = R.get("selected_ensemble")
        if sel and sel in R:
            r = R[sel]["test_per_snr"]
            row = "".join(f"{r[str(s)]['macro_f1']:>10.4f}"
                          if str(s) in r else f"{'-':>10}" for s in LV)
            print(f"       {sel:<14}{row}"
                  f"{R[sel]['test']['macro_f1']:>8.4f}   (baseline)")
            results["baseline"] = {"name": sel,
                                   "test": R[sel]["test"],
                                   "test_per_snr": r}

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    np.savez_compressed(args.out.replace(".json", "_preds.npz"),
                        yte=yte, ste=ste, yp=ypt, P=pte,
                        yva=yva, sva=sva, Pva=pva)
    print(f"\nsaved -> {args.out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/waveforms.npz")
    ap.add_argument("--out", default="results/dasnet_results.json")
    ap.add_argument("--baseline", default="results/results.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-dst", type=float, default=5e-3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--head", choices=["softmax", "evidential"],
                    default="softmax")
    ap.add_argument("--no-learnable-dst", action="store_true")
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--p-aug", type=float, default=0.5)
    ap.add_argument("--limit-groups", type=int, default=None)
    ap.add_argument("--max-temp", type=int, default=74,
                    help="GPU temperature target [C] for the governor")
    ap.add_argument("--no-throttle", action="store_true",
                    help="disable the thermal governor (full speed)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights from a checkpoint (.pt)")
    ap.add_argument("--pilot", action="store_true",
                    help="shortcut: --limit-groups 30 --epochs 8")
    args = ap.parse_args()
    if args.pilot:
        args.limit_groups = args.limit_groups or 30
        args.epochs = min(args.epochs, 8)
        args.out = args.out.replace(".json", "_pilot.json")
    run(args)
