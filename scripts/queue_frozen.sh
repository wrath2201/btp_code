#!/bin/bash
set -e
source ../.venv-dasnet/bin/activate

echo "=========================================================="
echo "Starting Frozen-DASNet DualPQ 5-Seed Validation Protocol"
echo "=========================================================="

for i in {0..4}; do
    OUT_FILE="results/multiseed/frozen_dualpq_seed${i}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "Error: $OUT_FILE already exists. Remove it manually before rerunning."
        exit 1
    fi
    echo "[$(date)] Running Frozen-DASNet DualPQ Seed $i"
    python scripts/run_frozen_dualpq.py \
        --gate concat \
        --out "$OUT_FILE" \
        --seed $i \
        --split-seed 0 \
        --epochs 40 \
        --batch 32
done

echo "[$(date)] Frozen-DASNet DualPQ 5-Seed Protocol Complete!"
