#!/bin/sh
# Runs the two conference-critical experiments sequentially, thermally
# throttled. Log: /tmp/opencode/dasnet_queue.log
set -x
cd "$(dirname "$0")/.."
PY=../.venv-dasnet/bin/python

echo "=================================================================="
echo "RUN 1/2: main DASNet run (learnable DST)   $(date)"
echo "=================================================================="
$PY scripts/run_dasnet.py --epochs 40 --patience 10 --batch 32 --workers 3 \
    --max-temp 74 --out results/dasnet_results.json

echo "=================================================================="
echo "RUN 2/2: ablation, frozen classical 1/f window   $(date)"
echo "=================================================================="
$PY scripts/run_dasnet.py --epochs 40 --patience 10 --batch 32 --workers 3 \
    --max-temp 74 --no-learnable-dst \
    --out results/dasnet_ablation_nodst.json

echo "=================================================================="
echo "QUEUE COMPLETE   $(date)"
echo "=================================================================="
