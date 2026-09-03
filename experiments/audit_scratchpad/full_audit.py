import json
import numpy as np
import os
from sklearn.metrics import f1_score

RESULTS_DIR = '/home/tsaini/Desktop/btp_code/btp-pq-ensemble/results'
MULTISEED_DIR = os.path.join(RESULTS_DIR, 'multiseed')

models = {
    'Classical Ensemble': {
        'json': os.path.join(MULTISEED_DIR, 'baseline_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'baseline_seed{}_preds.npz')
    },
    'DASNet': {
        'json': os.path.join(MULTISEED_DIR, 'dasnet_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'dasnet_seed{}_preds.npz')
    },
    'Original DualPQ-D': {
        'json': os.path.join(MULTISEED_DIR, 'dualpq_concat_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'dualpq_concat_seed{}_preds.npz')
    },
    'Frozen-DASNet DualPQ': {
        'json': os.path.join(MULTISEED_DIR, 'fixed_frozen_dualpq_seed{}.json'),
        'preds': os.path.join(MULTISEED_DIR, 'fixed_frozen_dualpq_seed{}_preds.npz')
    },
    'MGCNN-SDTransformer': {
        'json': os.path.join(RESULTS_DIR, 'mgcnn_sdtransformer_seed{}.json'),
        'preds': os.path.join(RESULTS_DIR, 'mgcnn_sdtransformer_seed{}_preds.npz')
    }
}

print("="*50)
print("PART 1: VERIFYING NUMBERS")
print("="*50)

all_results = {}
for name, paths in models.items():
    print(f"\n--- {name} ---")
    seeds_f1 = []
    preds_all = []
    trues_all = []
    snrs_all = []
    
    for i in range(5):
        p_path = paths['preds'].format(i)
        if os.path.exists(p_path):
            try:
                arr = np.load(p_path)
                yp = None
                if 'yp' in arr: yp = arr['yp']
                elif 'preds' in arr: yp = arr['preds']
                elif 'y_pred' in arr: yp = arr['y_pred']
                elif 'E_soft_vote' in arr: yp = np.argmax(arr['E_soft_vote'], axis=1) # though it might just be yp
                
                yt = None
                if 'yte' in arr: yt = arr['yte']
                elif 'labels' in arr: yt = arr['labels']
                elif 'y_true' in arr: yt = arr['y_true']
                
                ys = None
                if 'ste' in arr: ys = arr['ste']
                elif 'snrs' in arr: ys = arr['snrs']
                
                if yp is not None and yt is not None:
                    calc_f1 = f1_score(yt, yp, average='macro', zero_division=0)
                    seeds_f1.append(calc_f1)
                    preds_all.append(yp)
                    trues_all.append(yt)
                    if ys is not None:
                        snrs_all.append(ys)
                    else:
                        print(f"Warning: No SNRs found in {p_path}")
                        snrs_all.append(np.zeros_like(yt))
                else:
                    print(f"Could not find yp/yt in {arr.files}")
            except Exception as e:
                print(f"Error loading {p_path}: {e}")
    
    if len(seeds_f1) > 0:
        mean_f1 = np.mean(seeds_f1)
        std_f1 = np.std(seeds_f1, ddof=1)
        ci = 1.96 * std_f1 / np.sqrt(len(seeds_f1))
        print(f"Seeds Macro-F1: {[f'{x*100:.2f}%' for x in seeds_f1]}")
        print(f"Overall: {mean_f1*100:.2f} ± {std_f1*100:.2f}% (95% CI: ± {ci*100:.2f}%)")
    
    all_results[name] = {
        'seeds_f1': seeds_f1,
        'preds': preds_all,
        'trues': trues_all,
        'snrs': snrs_all
    }

print("\n" + "="*50)
print("PART 2: STATISTICAL SIGNIFICANCE (Paired Bootstrap for Seed 0)")
print("="*50)

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

fd_preds = all_results['Frozen-DASNet DualPQ']['preds']
if len(fd_preds) > 0:
    y_true = all_results['Frozen-DASNet DualPQ']['trues'][0]
    fd_pred = fd_preds[0]
    
    for comp in ['Classical Ensemble', 'DASNet', 'MGCNN-SDTransformer', 'Original DualPQ-D']:
        if len(all_results[comp]['preds']) > 0:
            comp_pred = all_results[comp]['preds'][0]
            md, cil, ciu, p = bootstrap_macro_f1(y_true, fd_pred, comp_pred)
            print(f"Frozen-DASNet vs {comp}:")
            print(f"  Observed Diff: {md*100:.2f}%")
            print(f"  95% CI: [{cil*100:.2f}%, {ciu*100:.2f}%]")
            print(f"  p-value: {p:.4f}")
            print(f"  Significant (p<0.05)? {'Yes' if p < 0.05 else 'No'}")

print("\n" + "="*50)
print("PART 3: PER-SNR ANALYSIS")
print("="*50)

for name in models.keys():
    if len(all_results[name]['preds']) == 0: continue
    
    snr_f1s = {999: [], 40: [], 30: [], 20: [], 10: [], 0: []}
    
    for i in range(len(all_results[name]['preds'])):
        yp = all_results[name]['preds'][i]
        yt = all_results[name]['trues'][i]
        ys = all_results[name]['snrs'][i]
        
        unique_snrs = np.unique(ys)
        for s in unique_snrs:
            idx = (ys == s)
            if np.sum(idx) == 0: continue
            f1 = f1_score(yt[idx], yp[idx], average='macro', zero_division=0)
            
            s_map = s
            if s == 999 or s == -1: s_map = 999
            if s_map in snr_f1s:
                snr_f1s[s_map].append(f1)
            else:
                if s == 0: snr_f1s[0].append(f1)
                elif s == -1: snr_f1s[0].append(f1)
                else: snr_f1s[s] = [f1]
            
    print(f"\n{name}:")
    mean_snrs = {}
    for s in [999, 40, 30, 20, 10, 0]:
        if s in snr_f1s and len(snr_f1s[s]) > 0:
            m = np.mean(snr_f1s[s])
            mean_snrs[s] = m
            name_s = "Clean" if s == 999 else f"{s}dB"
            print(f"  {name_s}: {m*100:.2f}%")
            
    if 999 in mean_snrs and 0 in mean_snrs:
        print(f"  Degradation Clean -> 0dB: {(mean_snrs[999] - mean_snrs[0])*100:.2f}%")
    if 20 in mean_snrs and 10 in mean_snrs:
        print(f"  Degradation 20 -> 10dB: {(mean_snrs[20] - mean_snrs[10])*100:.2f}%")
    if 10 in mean_snrs and 0 in mean_snrs:
        print(f"  Degradation 10 -> 0dB: {(mean_snrs[10] - mean_snrs[0])*100:.2f}%")

print("\n" + "="*50)
print("PART 4: PER-CLASS ANALYSIS (Averaged over 5 seeds)")
print("="*50)

for name in models.keys():
    if len(all_results[name]['preds']) == 0: continue
    
    class_f1s_seeds = []
    for i in range(len(all_results[name]['preds'])):
        yp = all_results[name]['preds'][i]
        yt = all_results[name]['trues'][i]
        f1s = f1_score(yt, yp, average=None, zero_division=0)
        class_f1s_seeds.append(f1s)
        
    avg_class_f1 = np.mean(class_f1s_seeds, axis=0)
    
    print(f"\n{name} Top 5 Classes: {np.argsort(avg_class_f1)[-5:][::-1]} (Scores: {[f'{x*100:.2f}%' for x in sorted(avg_class_f1)[-5:][::-1]]})")
    print(f"{name} Bottom 5 Classes: {np.argsort(avg_class_f1)[:5]} (Scores: {[f'{x*100:.2f}%' for x in sorted(avg_class_f1)[:5]]})")

print("\n" + "="*50)
print("PART 5/6: MGCNN AUDIT - checking hyperparams in config")
print("="*50)

j_path = models['MGCNN-SDTransformer']['json'].format(0)
if os.path.exists(j_path):
    with open(j_path, 'r') as f:
        data = json.load(f)
    print("MGCNN Config:")
    for k, v in data.get('config', {}).items():
        print(f"  {k}: {v}")
