import numpy as np
import glob
from sklearn.metrics import f1_score

def calculate_snr(name, pattern):
    files = sorted(glob.glob(pattern))
    if not files: return
    
    snrs = [999, 40, 30, 20, 10, 0]
    snr_f1s = {s: [] for s in snrs}
    
    for f in files:
        data = np.load(f)
        y_true = data['y_true'] if 'y_true' in data else data['yte']
        y_pred = data['y_pred'] if 'y_pred' in data else data['yp']
        st = data['ste'] if 'ste' in data else data['snr'] if 'snr' in data else None
        if st is None: continue
        
        for s in snrs:
            mask = (st == s)
            if np.any(mask):
                f1 = f1_score(y_true[mask], y_pred[mask], average='macro') * 100
                snr_f1s[s].append(f1)
                
    means = {s: np.mean(snr_f1s[s]) if snr_f1s[s] else 0.0 for s in snrs}
    print(f"[{name}] SNR:")
    for s in snrs:
        print(f"  {s}dB: {means[s]:.2f}")
    print("-" * 40)

calculate_snr("Frozen-DASNet DualPQ", "results/multiseed/fixed_frozen_dualpq_seed*_preds.npz")
calculate_snr("Classical Ensemble", "results/multiseed/baseline_seed*_preds.npz")
calculate_snr("DASNet", "results/multiseed/dasnet_seed*_preds.npz")
calculate_snr("MGCNN-SDTransformer", "results/mgcnn_sdtransformer_seed*_preds.npz")
calculate_snr("Original DualPQ-D", "results/multiseed/dualpq_concat_seed*_preds.npz")
