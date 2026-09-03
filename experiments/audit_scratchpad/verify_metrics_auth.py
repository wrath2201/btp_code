import json
import numpy as np
import glob
from sklearn.metrics import f1_score

def calculate_from_npz(name, pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for {name}")
        return
    f1s = []
    for f in files:
        data = np.load(f)
        y_true = data['y_true'] if 'y_true' in data else data['yte']
        y_pred = data['y_pred'] if 'y_pred' in data else data['yp']
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        f1s.append(f1)
    
    mean_val = np.mean(f1s)
    # Using sample standard deviation (ddof=1) since n=5 seeds are independent samples
    std_val = np.std(f1s, ddof=1)
    
    print(f"[{name}]")
    print(f"  Seeds: {', '.join(['{:.2f}'.format(x) for x in f1s])}")
    print(f"  Mean: {mean_val:.2f}%")
    print(f"  Sample SD (ddof=1): {std_val:.2f}%")
    print(f"  Population SD (ddof=0): {np.std(f1s, ddof=0):.2f}%")
    print("-" * 40)

calculate_from_npz("Frozen-DASNet DualPQ", "results/multiseed/fixed_frozen_dualpq_seed*_preds.npz")
calculate_from_npz("Classical Ensemble", "results/multiseed/baseline_seed*_preds.npz")
calculate_from_npz("DASNet", "results/multiseed/dasnet_seed*_preds.npz")
calculate_from_npz("MGCNN-SDTransformer", "results/mgcnn_sdtransformer_seed*_preds.npz")
calculate_from_npz("Original DualPQ-D", "results/multiseed/dualpq_concat_seed*_preds.npz")
