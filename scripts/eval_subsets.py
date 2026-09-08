import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.subset_utils import SUBSETS

def evaluate_npz(npz_path, subset_name):
    """
    Evaluates a model's npz file containing `yte` and `yp` (both in 1..29 label space)
    over the specific subset of classes.
    """
    if not os.path.exists(npz_path):
        return None
        
    data = np.load(npz_path)
    yte = data["yte"]
    yp = data["yp"]
    ste = data["ste"]
    
    subset = SUBSETS[subset_name]
    
    # Filter the test set to only rows where true label is in subset
    mask = np.isin(yte, subset)
    yte_sub = yte[mask]
    yp_sub = yp[mask]
    ste_sub = ste[mask]
    
    if len(yte_sub) == 0:
        return None
        
    # We can compute macro F1 by restricting the sklearn classes parameter to the subset
    macro_f1_clean = 0.0
    macro_f1_40db = 0.0
    
    # Clean Data
    mask_clean = (ste_sub == 999)
    if np.any(mask_clean):
        macro_f1_clean = f1_score(yte_sub[mask_clean], yp_sub[mask_clean], labels=subset, average="macro")
        
    # 40dB Data
    mask_40db = (ste_sub == 40)
    if np.any(mask_40db):
        macro_f1_40db = f1_score(yte_sub[mask_40db], yp_sub[mask_40db], labels=subset, average="macro")
        
    return {
        "Clean": macro_f1_clean * 100.0,
        "40dB": macro_f1_40db * 100.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=SUBSETS.keys())
    args = parser.parse_args()
    
    models = {
        "Classical Ensemble (weighted_vote)": f"results/preds/classical_weighted_vote_seed0_preds.npz",
        "Classical Ensemble (geometric_vote)": f"results/preds/classical_geometric_vote_seed0_preds.npz",
        "DASNet": f"results/subset_{args.subset}/dasnet_seed0_results_preds.npz",
        "MGCNN-SDTransformer": f"results/subset_{args.subset}/mgcnn_seed0_results_preds.npz",
        "DualPQ (End-to-End)": f"results/subset_{args.subset}/dualpq_seed0_results_preds.npz",
        "Frozen-DASNet DualPQ": f"results/subset_{args.subset}/frozen_dualpq_seed0_results_preds.npz",
    }
    
    print(f"=== Results for {args.subset.upper()} ({len(SUBSETS[args.subset])} classes) ===")
    
    for model_name, path_pattern in models.items():
        # Handle wildcards if needed (though we'll use seed0 explicitly for now)
        paths = glob.glob(path_pattern)
        if not paths:
            print(f"{model_name}:\n  [Missing predictions file: {path_pattern}]")
            continue
            
        path = paths[0]
        res = evaluate_npz(path, args.subset)
        if res is None:
            print(f"{model_name}:\n  [Error evaluating {path}]")
            continue
            
        print(f"{model_name}")
        print(f"  Clean Data: {res['Clean']:.2f}%")
        print(f"  40 dB Noise: {res['40dB']:.2f}%")
        
if __name__ == "__main__":
    main()
