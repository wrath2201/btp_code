"""
repro_frozen_dualpq.py -- reproduce Frozen-DASNet DualPQ predictions.

Why this exists: run_frozen_dualpq.py never calls torch.save, and
`*_preds.npz` is gitignored, so neither the weights nor the predictions of the
final proposed model are in the repository. Per-class metrics for it therefore
cannot be computed from committed artefacts -- the stage-2 head has to be
retrained.

This script is equivalent to run_frozen_dualpq.py with one optimisation that
changes nothing about the model: because the deep expert is frozen and kept in
eval() mode, and stage 2 uses NO waveform augmentation (PQDataset has no
augment path), z_deep is a constant per row. It is therefore computed once and
cached, after which stage-2 training costs milliseconds per epoch instead of a
full DST forward pass per batch per epoch.

Same as the original: gate=concat, AdamW(lr=1e-3, wd=1e-4), warmup 3 +
cosine, 40 epochs, patience 12, batch 32, CrossEntropy, StandardScaler fit on
train only, val-macro-F1 model selection, test read once.

    python scripts/repro_frozen_dualpq.py --seed 0            # one seed
    python scripts/repro_frozen_dualpq.py --seed 0 --embed-only   # just cache

Caveat: CPU RNG differs from the CUDA RNG used for the published runs, so the
head's initialisation and batch order differ. Expect the same distribution of
outcomes, not the published value to the decimal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.dasnet import DASNet
from src.dualpq import ClassicalExpert
from src.pipeline import grouped_stratified_split

N_CLASSES = 29
EMB_DIR = "data/frozen_emb"


# ------------------------------------------------------------------ stage 1
@torch.no_grad()
def embed(seed, W, batch, threads):
    """z_deep for every row, from the committed stage-1 DASNet checkpoint."""
    os.makedirs(EMB_DIR, exist_ok=True)
    cache = os.path.join(EMB_DIR, f"z_deep_seed{seed}.npy")
    if os.path.exists(cache):
        Z = np.load(cache)
        if len(Z) == len(W):
            print(f"[embed] cache hit {cache}")
            return Z
    torch.set_num_threads(threads)
    ck = f"results/multiseed/dasnet_seed{seed}_best.pt"
    net = DASNet(n_samples=W.shape[1])
    net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    net.eval()
    print(f"[embed] {ck} -> {len(W)} rows")

    out, t0 = [], time.perf_counter()
    for a in range(0, len(W), batch):
        x = torch.from_numpy(np.ascontiguousarray(W[a:a + batch])).float()
        snr = net.snr_est(x)[:, None] / 40.0
        cond = net.cond_mlp(snr)
        if not net.film_on:
            cond = torch.zeros_like(cond)
        h = net._tf_image(x)
        for st in net.stages:
            h = st(h, cond)
        out.append(h.mean(dim=(2, 3)).numpy().astype(np.float32))
        if (a // batch) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"    {a}/{len(W)}  [{el:.0f}s, ~{el/max(a+batch,1)*(len(W)-a):.0f}s left]",
                  flush=True)
    Z = np.vstack(out)
    np.save(cache, Z)
    print(f"[embed] saved {cache}  {Z.shape}")
    return Z


# ------------------------------------------------------------------ stage 2
class Stage2(nn.Module):
    """ClassicalExpert(191 -> 256) + concat with frozen z_deep(256) -> fc."""

    def __init__(self, n_feat=191, dropout=0.15):
        super().__init__()
        self.classical_expert = ClassicalExpert(in_features=n_feat, out_dim=256,
                                                dropout=dropout)
        self.deep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, N_CLASSES)

    def forward(self, z_deep, x_feat):
        z_c = self.classical_expert(x_feat)
        return self.fc(torch.cat([self.deep_dropout(z_deep), z_c], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-feat", default="data/dataset.npz")
    ap.add_argument("--data-wave", default="data/waveforms.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.data_feat, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"].astype(int)
    group, snr = d["group"], d["snr"].astype(int)
    W = np.load(a.data_wave)["W"]
    assert len(W) == len(y), "waveforms.npz / dataset.npz misaligned"

    Z = embed(a.seed, W, a.embed_batch, a.threads)
    if a.embed_only:
        return

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    (i_tr, i_va, i_te), _ = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), a.split_seed)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[i_tr]).astype(np.float32)
    Xva = sc.transform(X[i_va]).astype(np.float32)
    Xte = sc.transform(X[i_te]).astype(np.float32)

    T = lambda v: torch.from_numpy(np.ascontiguousarray(v))
    ztr, zva, zte = T(Z[i_tr]), T(Z[i_va]), T(Z[i_te])
    xtr, xva, xte = T(Xtr), T(Xva), T(Xte)
    ytr = T(y[i_tr] - 1).long()
    yva, yte = y[i_va], y[i_te]

    model = Stage2(n_feat=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(len(ytr) // a.batch, 1)
    warm, total = a.warmup * steps, a.epochs * steps

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        t = (s - warm) / max(total - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    from sklearn.metrics import f1_score

    best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
    g = torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        for b in range(steps):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            opt.zero_grad(set_to_none=True)
            loss = TF.cross_entropy(model(ztr[idx], xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(zva, xva).argmax(1).numpy() + 1
        f1v = f1_score(yva, pv, average="macro")
        print(f"epoch {epoch:02d} | val F1 {f1v:.4f}")
        if f1v > best_f1:
            best_f1, best_epoch, patience = f1v, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= a.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yp = model(zte, xte).argmax(1).numpy() + 1
    f1t = f1_score(yte, yp, average="macro")
    print(f"best_epoch={best_epoch}  val F1={best_f1:.4f}  TEST macro-F1={f1t:.4f}")

    os.makedirs("results/preds", exist_ok=True)
    out = a.out or f"results/preds/frozen_dualpq_repro_seed{a.seed}_preds.npz"
    np.savez_compressed(out, yte=yte, yp=yp, ste=snr[i_te])
    torch.save(best_state, out.replace("_preds.npz", "_stage2.pt"))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
