#!/bin/bash
set -e

# Activate the environment
source ../.venv-dasnet/bin/activate
mkdir -p results/multiseed

echo "========================================="
echo "STARTING FINAL RESEARCH VALIDATION"
echo "========================================="

# for i in {0..4}; do
#     echo "-----------------------------------------"
#     echo " RUNNING SEED $i"
#     echo "-----------------------------------------"
#     
#     echo "[1/3] DualPQ-D (Concat) Seed $i..."
#     python3 scripts/run_dualpq.py --gate concat --seed $i --split-seed 0 --out results/multiseed/dualpq_concat_seed${i}.json
# 
#     echo "[2/3] DASNet (Learnable DST) Seed $i..."
#     python3 scripts/run_dasnet.py --seed $i --split-seed 0 --out results/multiseed/dasnet_seed${i}.json
# done

# The baseline is run outside the torch environment or inside it if dependencies match.
# Since the classical baseline requires scikit-learn, lightgbm, etc., we can run it here
# if the environment has them (which it does, since build_dataset succeeded).
for i in {0..4}; do
    echo "-----------------------------------------"
    echo " RUNNING BASELINE SEED $i"
    echo "-----------------------------------------"
    python3 scripts/run_baseline_multiseed.py --seed $i --fast --out results/multiseed/baseline_seed${i}.json
done

echo "========================================="
echo "ALL MULTI-SEED VALIDATION RUNS COMPLETED"
echo "========================================="
