#!/bin/bash
set -e
echo "Starting experiments at $(date)"
export PYTHON_CMD="../.venv-dasnet/bin/python"

echo "=== OPEN-1: Rerun MGCNN-SDTransformer seeds 1-4 with --split-seed 0 ==="
for i in 1 2 3 4; do
  $PYTHON_CMD scripts/run_mgcnn_sdtransformer.py \
    --seed $i --split-seed 0 \
    --out results/mgcnn_sdtransformer_seed${i}.json \
    --save-ckpt results/multiseed/mgcnn_sdtransformer_seed${i}_best.pt
  $PYTHON_CMD scripts/reconstruct_preds.py mgcnn --seed $i --split-seed 0
done

echo "=== OPEN-3: Classical-branch-only ablation ==="
for i in 0 1 2 3 4; do
  $PYTHON_CMD scripts/run_dualpq.py --gate concat --seed $i --split-seed 0 \
    --classical-only \
    --out results/multiseed/dualpq_classical_only_seed${i}.json
done

echo "=== OPEN-4: Original DualPQ-D with --no-aug and no AMP ==="
for i in 0 1 2 3 4; do
  $PYTHON_CMD scripts/run_dualpq.py --gate concat --seed $i --split-seed 0 \
    --no-aug --no-amp \
    --out results/multiseed/dualpq_concat_noaug_seed${i}.json
done

echo "=== OPEN-6: Regenerate per_class summary.json ==="
$PYTHON_CMD scripts/eval_per_class_snr.py \
    --model "Classical Ensemble (weighted_vote)=results/preds/classical_weighted_vote_seed*_preds.npz" \
    --model "Classical Ensemble (geometric_vote)=results/preds/classical_geometric_vote_seed*_preds.npz" \
    --model "DASNet=results/preds/dasnet_seed*_preds.npz" \
    --model "MGCNN-SDTransformer=results/preds/mgcnn_seed*_preds.npz" \
    --model "Frozen-DASNet DualPQ=results/multiseed/frozen_dualpq_seed*_preds.npz"

echo "=== OPEN-7: Regenerate stats tests ==="
$PYTHON_CMD scripts/stats_tests.py

echo "Experiments finished at $(date)"
