import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def main():
    # SNR levels mapped to their JSON keys
    # 999 is Clean in the JSON format
    snr_keys = ["999", "40", "30", "20", "10", "0"]
    snr_labels = ["Clean", "40dB", "30dB", "20dB", "10dB", "0dB"]
    
    results = {
        "dualpq": {"all": [], "snrs": {k: [] for k in snr_keys}},
        "dasnet": {"all": [], "snrs": {k: [] for k in snr_keys}},
        "baseline": {"all": [], "snrs": {k: [] for k in snr_keys}}
    }
    
    for f in sorted(glob.glob("results/multiseed/*.json")):
        with open(f) as fp:
            d = json.load(fp)
            
            model = None
            if "dualpq_concat" in f:
                model = "dualpq"
            elif "dasnet" in f:
                model = "dasnet"
            elif "baseline" in f:
                model = "baseline"
            else:
                continue
            
            # Extract scores based on model type
            if model in ["dualpq", "dasnet"]:
                if "test" in d and "macro_f1" in d["test"]:
                    results[model]["all"].append(d["test"]["macro_f1"])
                    for k in snr_keys:
                        if "test_per_snr" in d and k in d["test_per_snr"]:
                            results[model]["snrs"][k].append(d["test_per_snr"][k]["macro_f1"])
            else: # Baseline
                if "geometric_vote" in d:
                    ens = d["geometric_vote"]
                    results[model]["all"].append(ens["test"]["macro_f1"])
                    for k in snr_keys:
                        if k in ens["test_per_snr"]:
                            results[model]["snrs"][k].append(ens["test_per_snr"][k]["macro_f1"])
                            
    # Ensure output directory exists
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Print Overall Summary Table
    print("="*60)
    print("FINAL 5-SEED VALIDATION RESULTS (TEST MACRO-F1)")
    print("="*60)
    models = [("DualPQ-D (Concat)", "dualpq"), ("DASNet (Learnable DST)", "dasnet"), ("Classical Baseline (Geometric)", "baseline")]
    for name, key in models:
        scores = results[key]["all"]
        mean_val = np.mean(scores) * 100
        std_val = np.std(scores) * 100
        print(f"{name:35s}: {mean_val:.2f}% ± {std_val:.2f}%")
    print("="*60)
    
    # 2. Compute SNR Means and Stds
    snr_means = {model: [] for model in ["dualpq", "dasnet", "baseline"]}
    snr_stds = {model: [] for model in ["dualpq", "dasnet", "baseline"]}
    
    for _, key in models:
        for sk in snr_keys:
            vals = results[key]["snrs"][sk]
            if len(vals) == 0:
                vals = [0]
            snr_means[key].append(np.mean(vals) * 100)
            snr_stds[key].append(np.std(vals) * 100)
            
    # Print SNR Breakdowns
    print("\nPER-SNR BREAKDOWN (Means):")
    header = f"{'Model':35s} " + " ".join([f"{l:>8s}" for l in snr_labels])
    print(header)
    for name, key in models:
        row_str = f"{name:35s} " + " ".join([f"{v:8.2f}" for v in snr_means[key]])
        print(row_str)
        
    # 3. Plot F1 vs SNR Curve
    plt.figure(figsize=(10, 6))
    x = np.arange(len(snr_labels))
    
    colors = {"dualpq": "#2E86C1", "dasnet": "#E74C3C", "baseline": "#27AE60"}
    labels = {"dualpq": "DualPQ-Net (Proposed)", "dasnet": "DASNet (Deep Only)", "baseline": "Classical Ensemble"}
    
    for key in ["baseline", "dasnet", "dualpq"]:
        y = np.array(snr_means[key])
        err = np.array(snr_stds[key])
        plt.plot(x, y, marker='o', linewidth=2.5, markersize=8, color=colors[key], label=labels[key])
        plt.fill_between(x, y - err, y + err, color=colors[key], alpha=0.15)
        
    plt.xticks(x, snr_labels)
    plt.ylabel("Macro F1-Score (%)")
    plt.xlabel("Signal-to-Noise Ratio (SNR)")
    plt.title("Performance Under Increasing Noise (5-Seed Average)")
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/figures/fig_f1_vs_snr.png", dpi=300)
    plt.close()
    
    # 4. Plot Delta SNR (DualPQ vs Baseline)
    plt.figure(figsize=(10, 5))
    delta = np.array(snr_means["dualpq"]) - np.array(snr_means["baseline"])
    delta_err = np.sqrt(np.array(snr_stds["dualpq"])**2 + np.array(snr_stds["baseline"])**2) # Error propagation
    
    bars = plt.bar(x, delta, yerr=delta_err, capsize=5, color="#8E44AD", alpha=0.8, edgecolor="black")
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    
    # Annotate bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        v_offset = 2 if height >= 0 else -3.5
        plt.text(bar.get_x() + bar.get_width()/2., height + v_offset, f"{height:+.1f}%", 
                 ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold', fontsize=10)

    plt.xticks(x, snr_labels)
    plt.ylabel("Absolute Improvement (%)")
    plt.title("DualPQ-Net Improvement over Classical Baseline")
    plt.ylim(min(delta)-10, max(delta)+10)
    plt.tight_layout()
    plt.savefig("results/figures/fig_delta_snr.png", dpi=300)
    plt.close()
    
    print("\n[SUCCESS] Generated figures in results/figures/")

if __name__ == "__main__":
    main()
