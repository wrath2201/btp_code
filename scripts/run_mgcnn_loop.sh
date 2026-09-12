#!/bin/bash
PYTHON="/home/tsaini/Desktop/btp_code/.venv-dasnet/bin/python"

for i in {1..4}; do
    echo "Running MGCNN seed $i..."
    $PYTHON scripts/run_mgcnn_sdtransformer.py \
        --seed $i \
        --split-seed 0 \
        --out results/mgcnn_sdtransformer_seed${i}.json \
        --save-ckpt results/multiseed/mgcnn_sdtransformer_seed${i}_best.pt
done
