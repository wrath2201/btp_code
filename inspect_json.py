import json
import os

results_dir = '/home/tsaini/Desktop/btp_code/btp-pq-ensemble/results/multiseed'
files = {
    'baseline': 'baseline_seed0.json',
    'dasnet': 'dasnet_seed0.json',
    'original': 'dualpq_concat_seed0.json',
    'frozen': 'fixed_frozen_dualpq_seed0.json',
}
for k, v in files.items():
    p = os.path.join(results_dir, v)
    if os.path.exists(p):
        with open(p, 'r') as f:
            data = json.load(f)
            print(f"{k} keys:", list(data.keys()))
            if k == 'baseline':
                if 'ensemble_test' in data:
                    print("  ensemble_test macro_f1:", data['ensemble_test']['macro_f1'])
            elif k == 'dasnet':
                print("  dasnet test_f1:", data.get('test_f1'))
            elif k == 'original':
                print("  original test_macro_f1:", data.get('test_macro_f1'))
            elif k == 'frozen':
                print("  frozen test_macro_f1:", data.get('test_macro_f1'))

import glob
mgcnn_files = glob.glob('/home/tsaini/Desktop/btp_code/btp-pq-ensemble/results/mgcnn_sdtransformer_seed0.json')
if mgcnn_files:
    with open(mgcnn_files[0], 'r') as f:
        data = json.load(f)
        print("mgcnn keys:", list(data.keys()))
        print("  mgcnn test_macro_f1:", data.get('test_macro_f1'))
