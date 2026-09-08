#!/bin/bash

# run_subset_experiments.sh
# Runs the subset evaluation pipeline, preserving progress and using the thermal governor.

set -e

# Change to the root of the project
cd "$(dirname "$0")"

PYTHON_CMD="../.venv-dasnet/bin/python"

# Set CUDA config for reproducibility and memory
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SUBSETS=("top17" "top21" "top25")
SEED=0

for SUBSET in "${SUBSETS[@]}"; do
    echo "=========================================================="
    echo "Starting experiments for subset: ${SUBSET}"
    echo "=========================================================="
    
    OUT_DIR="results/subset_${SUBSET}"
    mkdir -p "${OUT_DIR}"
    
    # 1. DASNet
    DASNET_OUT="${OUT_DIR}/dasnet_seed${SEED}_results.json"
    if [ -f "${DASNET_OUT}" ]; then
        echo "Skipping DASNet (already complete): ${DASNET_OUT}"
    else
        echo "Running DASNet..."
        $PYTHON_CMD scripts/run_dasnet.py \
            --epochs 50 \
            --seed $SEED \
            --class-subset $SUBSET \
            --out "${DASNET_OUT}"
    fi
    
    # 2. MGCNN-SDTransformer
    MGCNN_OUT="${OUT_DIR}/mgcnn_seed${SEED}_results.json"
    if [ -f "${MGCNN_OUT}" ]; then
        echo "Skipping MGCNN (already complete): ${MGCNN_OUT}"
    else
        echo "Running MGCNN-SDTransformer..."
        $PYTHON_CMD scripts/run_mgcnn_sdtransformer.py \
            --epochs 40 \
            --seed $SEED \
            --class-subset $SUBSET \
            --out "${MGCNN_OUT}" \
            --save-ckpt "${OUT_DIR}/mgcnn_seed${SEED}_best.pt"
    fi
    
    # 3. DualPQ (End-to-End)
    DUALPQ_OUT="${OUT_DIR}/dualpq_seed${SEED}_results.json"
    if [ -f "${DUALPQ_OUT}" ]; then
        echo "Skipping DualPQ End-to-End (already complete): ${DUALPQ_OUT}"
    else
        echo "Running DualPQ (End-to-End)..."
        $PYTHON_CMD scripts/run_dualpq.py \
            --gate "snr_learned" \
            --epochs 40 \
            --seed $SEED \
            --class-subset $SUBSET \
            --out "${DUALPQ_OUT}"
    fi
    
    # 4. Frozen-DASNet DualPQ
    FROZEN_OUT="${OUT_DIR}/frozen_dualpq_seed${SEED}_results.json"
    if [ -f "${FROZEN_OUT}" ]; then
        echo "Skipping Frozen-DASNet DualPQ (already complete): ${FROZEN_OUT}"
    else
        echo "Running Frozen-DASNet DualPQ..."
        $PYTHON_CMD scripts/run_frozen_dualpq.py \
            --gate "snr_learned" \
            --epochs 40 \
            --seed $SEED \
            --class-subset $SUBSET \
            --checkpoint-dir "${OUT_DIR}" \
            --out "${FROZEN_OUT}"
    fi
    
    echo ""
    echo "Evaluation for ${SUBSET}:"
    $PYTHON_CMD scripts/eval_subsets.py --subset $SUBSET
    echo ""
done

echo "All subset experiments complete!"
