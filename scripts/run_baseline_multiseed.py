import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import src.pipeline

# Patch the split function so that the Train/Val/Test partitions NEVER CHANGE
# regardless of what random seed is provided for the models.
original_split = src.pipeline.grouped_stratified_split

def patched_split(y, group, frac, seed):
    print(f"[Patch] Forcing grouped_stratified_split to use seed 0 (was passed {seed})")
    return original_split(y, group, frac, 0)

src.pipeline.grouped_stratified_split = patched_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-new-folds", type=int, default=None)
    a = ap.parse_args()
    
    src.pipeline.run(a.data, a.out, a.folds, a.seed, a.n_jobs, a.fast, a.max_new_folds)
