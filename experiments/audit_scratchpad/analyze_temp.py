import json
import glob
import numpy as np

files = glob.glob('results/mgcnn_sdtransformer_seed*.json')
test_f1s = []
best_epochs = []
val_f1s = []
snr_res = {'999': [], '40': [], '30': [], '20': [], '10': [], '0': []}

for f in sorted(files):
    with open(f) as fp:
        data = json.load(fp)
        test_f1s.append(data['test']['macro_f1'])
        val_f1s.append(data['val']['macro_f1'])
        best_epochs.append(data['config']['best_epoch'])
        for k in snr_res.keys():
            if k in data['test_per_snr']:
                snr_res[k].append(data['test_per_snr'][k]['macro_f1'])

print('Seed | Best Val F1 | Test Macro-F1 | Best Epoch')
for i in range(len(test_f1s)):
    print(f'{i:4d} | {val_f1s[i]:.4f}      | {test_f1s[i]:.4f}        | {best_epochs[i]:4d}')

mean_f1 = np.mean(test_f1s)
std_f1 = np.std(test_f1s)
ci = 1.96 * std_f1 / np.sqrt(len(test_f1s))

print(f'\nMean ± SD: {mean_f1*100:.2f}% ± {std_f1*100:.2f}%')
print(f'95% CI: [{ (mean_f1-ci)*100:.2f}%, { (mean_f1+ci)*100:.2f}%]')

print('\nPer-SNR Macro-F1 (Means):')
print(f"Clean: {np.mean(snr_res['999'])*100:.2f}%")
print(f"40dB : {np.mean(snr_res['40'])*100:.2f}%")
print(f"30dB : {np.mean(snr_res['30'])*100:.2f}%")
print(f"20dB : {np.mean(snr_res['20'])*100:.2f}%")
print(f"10dB : {np.mean(snr_res['10'])*100:.2f}%")
print(f" 0dB : {np.mean(snr_res['0'])*100:.2f}%")
