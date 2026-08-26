#!/bin/bash
set -e

# Activate the PyTorch environment
source ../.venv-dasnet/bin/activate

echo "Starting DualPQ-Net Experiments..."

# Experiment D: Simple Concatenation
echo "Running Experiment D (Concat)..."
python3 scripts/run_dualpq.py --gate concat --out results/dualpq_concat.json

# Experiment E: SNR-Conditioned Learned Gate (The Proposed Method)
echo "Running Experiment E (SNR Learned Gate)..."
python3 scripts/run_dualpq.py --gate snr_learned --out results/dualpq_snr_learned.json

# Experiment F: Hard SNR Routing
echo "Running Experiment F (Hard SNR Gate)..."
python3 scripts/run_dualpq.py --gate snr_hard --out results/dualpq_snr_hard.json

# Experiment G: Feature-Conditioned Gate (No Explicit SNR)
echo "Running Experiment G (Feature Gate)..."
python3 scripts/run_dualpq.py --gate feature_learned --out results/dualpq_feature_learned.json

echo "All DualPQ-Net experiments completed successfully!"
