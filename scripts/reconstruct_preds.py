"""
reconstruct_preds.py -- regenerate the test-set prediction arrays that
`.gitignore` excludes (`*_preds.npz`), so per-class metrics can be recomputed
from the committed artefacts.

Reconstructable from what is in the repo:
  classical  refit the 4 base learners (seed 0, fast=False) and rebuild
             `weighted_vote` with the vote weights stored in results.json
  dasnet     forward pass from results/multiseed/dasnet_seed{k}_best.pt
  mgcnn      forward pass from results/multiseed/mgcnn_sdtransformer_seed{k}_best.pt
             NOTE: those runs used --split-seed == --seed, so the split must
             follow the seed to land on the same test rows.

NOT reconstructable: frozen_dualpq and dualpq_concat -- run_frozen_dualpq.py
and run_dualpq.py never call torch.save, so no weights were kept.

Output: results/preds/{model}_seed{k}_preds.npz with yte, yp, ste (labels 1..29).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.pipeline import grouped_stratified_split

OUT_DIR = "results/preds"


def _save(name, yte, yp, ste):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"{name}_preds.npz")
    np.savez_compressed(p, yte=np.asarray(yte), yp=np.asarray(yp),
                        ste=np.asarray(ste))
    print(f"saved -> {p}   n={len(yte)}")
    return p


# ---------------------------------------------------------------- classical
def classical(args):
    from src.pipeline import make_models, predict_proba

    d = np.load(args.data_feat, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    group, snr = d["group"], d["snr"].astype(int)

    (i_tr, i_va, i_te), _ = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), args.split_seed)
    print(f"[split] rows train={len(i_tr)} val={len(i_va)} test={len(i_te)}")

    ref = json.load(open(args.reference))
    W = ref["vote_weights"]
    names = ["rf", "lgbm", "svm", "mlp"]
    print(f"[ens] reusing published vote weights: {W}")

    P = {}
    for m in names:
        t0 = time.perf_counter()
        mdl = make_models(args.seed, args.n_jobs, fast=args.fast)[m]
        mdl.fit(X[i_tr], y[i_tr])
        P[m] = predict_proba(m, mdl, X[i_te])
        del mdl
        print(f"  fit {m:<5} [{time.perf_counter()-t0:.0f}s]", flush=True)

    w = np.array([W[m] for m in names])
    wv = np.tensordot(w, np.stack([P[m] for m in names]), 1)
    yp = wv.argmax(1) + 1

    for m in names:
        _save(f"classical_{m}_seed{args.seed}", y[i_te], P[m].argmax(1) + 1, snr[i_te])
    sv = np.mean([P[m] for m in names], 0)
    _save(f"classical_soft_vote_seed{args.seed}", y[i_te], sv.argmax(1) + 1, snr[i_te])
    lg = lambda Q: np.log(np.clip(Q, 1e-9, 1))
    gm = np.exp(np.tensordot(w, np.stack([lg(P[m]) for m in names]), 1))
    _save(f"classical_geometric_vote_seed{args.seed}", y[i_te], gm.argmax(1) + 1, snr[i_te])
    return _save(f"classical_weighted_vote_seed{args.seed}", y[i_te], yp, snr[i_te])


# ---------------------------------------------------------------- deep nets
def _torch_infer(model, W, batch, threads):
    import torch
    torch.set_num_threads(threads)
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(W), batch):
            x = torch.from_numpy(np.ascontiguousarray(W[a:a + batch])).float()
            out.append(model(x).argmax(1).numpy())
            if (a // batch) % 20 == 0:
                print(f"    {a}/{len(W)}", flush=True)
    return np.concatenate(out)


def dasnet(args):
    import torch
    from src.dasnet import DASNet

    dw = np.load(args.data_wave)
    Wv = dw["W"]
    d = np.load(args.data_feat, allow_pickle=True)
    y, group, snr = d["y"].astype(int), d["group"], d["snr"].astype(int)
    assert len(Wv) == len(y), "waveforms.npz and dataset.npz are misaligned"

    (i_tr, i_va, i_te), _ = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), args.split_seed)

    ck = f"results/multiseed/dasnet_seed{args.seed}_best.pt"
    model = DASNet(n_samples=Wv.shape[1])
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    print(f"[model] loaded {ck}")

    yp = _torch_infer(model, Wv[i_te], args.batch, args.threads) + 1
    return _save(f"dasnet_seed{args.seed}", y[i_te], yp, snr[i_te])


def mgcnn(args):
    import torch
    from src.mgcnn_sdtransformer import MGCNNSDTransformer

    dw = np.load(args.data_wave)
    Wv = dw["W"]
    d = np.load(args.data_feat, allow_pickle=True)
    y, group, snr = d["y"].astype(int), d["group"], d["snr"].astype(int)

    # these runs used --split-seed == --seed
    split_seed = args.seed if args.split_seed is None else args.split_seed
    (i_tr, i_va, i_te), _ = grouped_stratified_split(
        y, group, (0.70, 0.15, 0.15), split_seed)
    print(f"[split] split_seed={split_seed} (matches the published config)")

    ck = f"results/multiseed/mgcnn_sdtransformer_seed{args.seed}_best.pt"
    model = MGCNNSDTransformer(num_classes=29)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    print(f"[model] loaded {ck}")

    yp = _torch_infer(model, Wv[i_te], args.batch, args.threads) + 1
    return _save(f"mgcnn_seed{args.seed}", y[i_te], yp, snr[i_te])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["classical", "dasnet", "mgcnn"])
    ap.add_argument("--data-feat", default="data/dataset.npz")
    ap.add_argument("--data-wave", default="data/waveforms.npz")
    ap.add_argument("--reference", default="results/results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.model != "mgcnn" and a.split_seed is None:
        a.split_seed = 0
    {"classical": classical, "dasnet": dasnet, "mgcnn": mgcnn}[a.model](a)
