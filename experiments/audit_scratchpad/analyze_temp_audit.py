import json
import numpy as np
import os
from glob import glob
from sklearn.metrics import f1_score, confusion_matrix

RESULTS_DIR = '/home/tsaini/Desktop/btp_code/btp-pq-ensemble/results'
MULTISEED_DIR = os.path.join(RESULTS_DIR, 'multiseed')

models = {
    'Classical': {
        'json': os.path.join(MULTISEED_DIR, 'baseline_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'baseline_seed{}_preds.npz')
    },
    'DASNet': {
        'json': os.path.join(MULTISEED_DIR, 'dasnet_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'dasnet_seed{}_preds.npz')
    },
    'Original DualPQ': {
        'json': os.path.join(MULTISEED_DIR, 'dualpq_concat_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'dualpq_concat_seed{}_preds.npz')
    },
    'Frozen-DASNet': {
        'json': os.path.join(MULTISEED_DIR, 'fixed_frozen_dualpq_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'fixed_frozen_dualpq_seed{}_preds.npz')
    },
    'MGCNN': {
        'json': os.path.join(RESULTS_DIR, 'mgcnn_sdtransformer_seed{}.json'),
        'preds': os.path.join(RESULTS_DIR, 'mgcnn_sdtransformer_seed{}_preds.npz')
    }
}

# Part 1: Verify every number
print("="*50)
print("PART 1: VERIFYING NUMBERS")
print("="*50)

all_results = {}
for name, paths in models.items():
    print(f"\n--- {name} ---")
    seeds_f1 = []
    per_snr_f1s = {k: [] for k in ['Clean', '40dB', '30dB', '20dB', '10dB', '0dB']}
    preds_all = []
    trues_all = []
    snrs_all = []
    
    for i in range(5):
        j_path = paths['json'].format(i)
        if not os.path.exists(j_path):
            print(f"Seed {i} missing JSON: {j_path}")
            continue
        with open(j_path, 'r') as f:
            data = json.load(f)
        
        # Test Macro-F1
        f1 = data.get('test_macro_f1', None)
        if f1 is None: f1 = data.get('test_f1', None)
        seeds_f1.append(f1)
        
        # Load npz for exact per-snr and per-class if needed
        p_path = paths['preds'].format(i)
        if os.path.exists(p_path):
            arr = np.load(p_path)
            if 'preds' in arr and 'labels' in arr and 'snrs' in arr:
                preds_all.append(arr['preds'])
                trues_all.append(arr['labels'])
                snrs_all.append(arr['snrs'])
            elif 'y_pred' in arr and 'y_true' in arr and 'snrs' in arr:
                preds_all.append(arr['y_pred'])
                trues_all.append(arr['y_true'])
                snrs_all.append(arr['snrs'])
    
    if len(seeds_f1) > 0:
        mean_f1 = np.mean(seeds_f1)
        std_f1 = np.std(seeds_f1, ddof=1)
        ci = 1.96 * std_f1 / np.sqrt(len(seeds_f1))
        print(f"Seeds Macro-F1: {[f'{x:.4f}' for x in seeds_f1]}")
        print(f"Overall: {mean_f1*100:.2f} ± {std_f1*100:.2f}% (95% CI: ± {ci*100:.2f}%)")
    
    all_results[name] = {
        'seeds_f1': seeds_f1,
        'preds': preds_all,
        'trues': trues_all,
        'snrs': snrs_all
    }

print("\n" + "="*50)
print("PART 2: STATISTICAL SIGNIFICANCE (Paired Bootstrap)")
print("="*50)
# We will do bootstrap over the test samples for Seed 0 or concatenate all seeds?
# The prompt says "paired bootstrap over test samples for Macro-F1". Let's do it for Seed 0 since models were trained independently.
# Or aggregate predictions? Usually, statistical significance is computed on a single test set run, e.g. seed 0. Let's do it for Seed 0.
def bootstrap_macro_f1(y_true, y_pred1, y_pred2, n_bootstraps=1000):
    n = len(y_true)
    diffs = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        f1_1 = f1_score(y_true[indices], y_pred1[indices], average='macro', zero_division=0)
        f1_2 = f1_score(y_true[indices], y_pred2[indices], average='macro', zero_division=0)
        diffs.append(f1_1 - f1_2)
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_val = np.sum(diffs <= 0) / n_bootstraps
    return mean_diff, ci_lower, ci_upper, p_val

fd_preds = all_results['Frozen-DASNet']['preds']
if len(fd_preds) > 0:
    y_true = all_results['Frozen-DASNet']['trues'][0]
    fd_pred = fd_preds[0]
    
    comparisons = ['Classical', 'DASNet', 'MGCNN', 'Original DualPQ']
    for comp in comparisons:
        if len(all_results[comp]['preds']) > 0:
            comp_pred = all_results[comp]['preds'][0]
            md, cil, ciu, p = bootstrap_macro_f1(y_true, fd_pred, comp_pred)
            print(f"Frozen-DASNet vs {comp}:")
            print(f"  Observed Diff: {md*100:.2f}%")
            print(f"  95% CI: [{cil*100:.2f}%, {ciu*100:.2f}%]")
            print(f"  p-value: {p:.4f}")
            print(f"  Significant? {'Yes' if p < 0.05 else 'No'}")

print("\n" + "="*50)
print("PART 3: PER-SNR ANALYSIS")
print("="*50)

snr_map = {0: 'Clean', 40: '40dB', 30: '30dB', 20: '20dB', 10: '10dB', 1: '0dB', -1: '0dB'} # Need to check how snrs are stored

for name in models.keys():
    if len(all_results[name]['preds']) == 0: continue
    
    # We will average over seeds
    snr_f1s = {s: [] for s in [0, 40, 30, 20, 10, -1]}
    
    for i in range(len(all_results[name]['preds'])):
        yp = all_results[name]['preds'][i]
        yt = all_results[name]['trues'][i]
        ys = all_results[name]['snrs'][i]
        
        unique_snrs = np.unique(ys)
        for s in unique_snrs:
            idx = (ys == s)
            f1 = f1_score(yt[idx], yp[idx], average='macro', zero_division=0)
            if s not in snr_f1s: snr_f1s[s] = []
            snr_f1s[s].append(f1)
            
    print(f"\n{name}:")
    for s in sorted(snr_f1s.keys(), reverse=True):
        if len(snr_f1s[s]) > 0:
            print(f"  SNR {s}: {np.mean(snr_f1s[s])*100:.2f}%")

print("\n" + "="*50)
print("PART 4: PER-CLASS ANALYSIS")
print("="*50)
for name in models.keys():
    if len(all_results[name]['preds']) == 0: continue
    # seed 0 per-class
    yp = all_results[name]['preds'][0]
    yt = all_results[name]['trues'][0]
    f1s = f1_score(yt, yp, average=None, zero_division=0)
    print(f"\n{name} Seed 0 Per-Class F1 (first 10): {f1s[:10]}")
