#!/bin/bash
set -e
source ../.venv-dasnet/bin/activate

echo "=========================================================="
echo "Starting Frozen-DASNet DualPQ 5-Seed Validation Protocol"
echo "=========================================================="

for i in {0..4}; do
    echo "[$(date)] Running Frozen-DASNet DualPQ Seed $i"
    python scripts/run_frozen_dualpq.py \
        --gate concat \
        --out results/multiseed/fixed_frozen_dualpq_seed${i}.json \
        --seed $i \
        --split-seed 0 \
        --epochs 40 \
        --batch 32
done

echo "[$(date)] Frozen-DASNet DualPQ 5-Seed Protocol Complete!"
