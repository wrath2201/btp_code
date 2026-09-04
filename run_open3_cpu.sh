#!/bin/bash
set -e
echo "Starting OPEN-3 on CPU at $(date)"
export PYTHON_CMD="../.venv-dasnet/bin/python"

# Force PyTorch to use CPU only
export CUDA_VISIBLE_DEVICES=""

for i in 0 1 2 3 4; do
  if [ -f "results/multiseed/dualpq_classical_only_seed${i}.json" ]; then
    echo "Seed $i already completed. Skipping."
    continue
  fi
  $PYTHON_CMD scripts/run_dualpq.py --gate concat --seed $i --split-seed 0 \
    --classical-only --no-aug \
    --out results/multiseed/dualpq_classical_only_seed${i}.json
done
echo "OPEN-3 finished at $(date)"
