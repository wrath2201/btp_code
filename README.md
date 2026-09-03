# Frozen-DASNet DualPQ
### Decoupled Hybrid Power Quality Disturbance Classification Under Severe Noise

## 1. Project Overview
This repository contains the code and evaluation framework for classifying 29 distinct Power Quality Disturbance (PQD) classes under extreme noise conditions. The benchmark evaluates deep, classical, and hybrid architectures using a rigorous waveform-grouped protocol across five independent training seeds.

## 2. Research Question / Motivation
Can we improve the robustness of PQD classification under severe noise (e.g., 10 dB and 0 dB) by fusing deep learned representations with classical signal-processing features? Furthermore, does decoupling the optimization of the deep and classical branches prevent training instability?

## 3. Contributions
1. A project-developed PQD classification architecture using a differentiable/adaptive Stockwell-transform-based representation with SNR-conditioned deep processing.
2. A hybrid architecture combining learned deep representations with handcrafted classical signal features.
3. A two-stage frozen-representation training strategy that substantially reduces run-to-run variability relative to the original jointly trained DualPQ configuration.
4. A grouped evaluation protocol covering 29 PQD classes and six noise conditions while keeping all variants of a base waveform within one partition.
5. A five-seed evaluation with preserved raw predictions enabling independent verification of reported metrics.

## 4. Final Proposed Method
**Frozen-DASNet DualPQ** is our final proposed fusion strategy. It combines a stage-1-trained, frozen DASNet representation with a trainable physical-feature branch and classification head. The study evaluates whether decoupling the stage-1-trained deep representation from end-to-end joint optimization improves robustness and stability. Stage 1 and stage 2 use the same training partition, with validation-based model selection.

![Frozen-DASNet DualPQ Architecture](docs/figures/frozen_dualpq_architecture.png)

## 5. Model Provenance

| Model | Category | Role |
|---|---|---|
| **Classical Ensemble** | Baseline | Classical signal-processing/ML reference |
| **DASNet** | Proposed architecture | Evaluate the project-developed deep representation independently |
| **Original DualPQ-D** | Proposed variant | Original jointly trained hybrid |
| **Frozen-DASNet DualPQ** | Final proposed method | Stage-1-trained DASNet representation frozen during Stage 2 and fused with classical features |
| **MGCNN-SDTransformer** | External/reimplemented baseline | Comparison against a published architecture under our evaluation protocol |

## 6. Experimental Protocol

| Property | Value |
|---|---|
| Classes | 29 |
| Waveform length | 1280 |
| Sampling rate | 6400 Hz |
| Fundamental frequency | 50 Hz |
| Cycles | 10 |
| Evaluation conditions | Clean, 40, 30, 20, 10, 0 dB |
| Dataset size | 34,800 |
| Split | Grouped stratified 70/15/15 |
| Primary metric | Macro-F1 |
| Deep/hybrid training seeds | 5 |
| Reported variability | Mean ± sample SD (ddof=1) |

## 7. Final Results

| Model | Mean Macro-F1 | SD |
|---|---:|---:|
| Classical Ensemble | 71.52 | 0.84 |
| DASNet | 69.72 | 9.11 |
| MGCNN-SDTransformer | 66.59 | 0.98 |
| Original DualPQ-D | 61.63 | 15.58 |
| **Frozen-DASNet DualPQ** | **74.46** | **1.08** |

### Training-Strategy Comparison
*Original DualPQ-D* (61.63 ± 15.58) vs. *Frozen-DASNet DualPQ* (74.46 ± 1.08). 
This comparison evaluates the effect of the two-stage frozen representation strategy relative to the original jointly trained hybrid. The substantial reduction in run-to-run variability is consistent with improved optimization stability.

## 8. Per-SNR Results

| Condition | Classical Ensemble | DASNet | MGCNN-SDTransformer | Original DualPQ-D | **Frozen-DASNet DualPQ** |
|---|---:|---:|---:|---:|---:|
| Clean | 93.30 | 85.59 | 73.18 | 77.29 | **91.18** |
| 40 dB | 93.34 | 85.83 | 73.65 | 77.56 | **91.69** |
| 30 dB | 91.56 | 84.75 | 72.93 | 75.38 | **89.56** |
| 20 dB | 82.52 | 77.62 | 70.36 | 68.32 | **82.36** |
| 10 dB | 52.88 | 58.74 | 60.18 | 49.20 | **62.36** |
| 0 dB  | 15.55 | 25.80 | 49.26 | 22.04 | **27.00** |

## 9. Key Findings
- **Frozen-DASNet DualPQ** has the highest mean Macro-F1 among evaluated methods.
- The absolute improvement over Classical Ensemble is approximately 2.94 percentage points.
- The larger distinction is run-to-run stability: Frozen-DASNet SD = 1.08 vs Original DualPQ-D SD = 15.58.
- At 10 dB, Frozen-DASNet is approximately tied with Classical Ensemble.
- At 0 dB, performance remains poor across the board.
- Results are consistent with improved optimization stability from the two-stage frozen training strategy.

## 10. Reproducibility Quickstart

*(Note: Dataset generation and model training are computationally expensive and require a GPU).*

**1. Environment Setup:**
```bash
python -m venv .venv-dasnet
source .venv-dasnet/bin/activate
pip install -r requirements.txt
```

**2. Dataset Generation (Produces 34,800 rows):**
```bash
for k in 0 1 2 3 4; do python scripts/build_dataset.py --step $k --n-base 200; done
python scripts/build_dataset.py --merge
```

**3. DASNet Stage-1 Training:**
```bash
python scripts/run_dasnet.py --out results/multiseed/dasnet_seed0.json --seed 0
```

**4. Frozen-DASNet Stage-2 Training:**
```bash
python scripts/run_frozen_dualpq.py --out results/multiseed/frozen_dualpq_seed0.json --checkpoint-dir results/multiseed --seed 0
```

**5. Evaluation / Artifacts:**
All final evaluated artifacts and predictions are stored in `results/multiseed/`.
For a full manifest, see `results/FINAL_RESULTS.md`.

## 11. Limitations
1. The dataset is entirely synthetic.
2. Real-world electrical measurement validation is not yet performed.
3. Performance at 0 dB remains low.
4. Five seeds measure training-run variability on the same grouped split; they are not five independent dataset partitions.
5. Generalization to unseen real-world operating conditions remains future work.

## 12. Repository Structure
- `src/`: Core architecture implementations (`dasnet.py`, `dualpq.py`, `pipeline.py`).
- `scripts/`: Training and dataset building scripts.
- `experiments/`: Analytical and evaluation scripts.
- `results/`: Contains finalized evaluations (`FINAL_RESULTS.md`) and historical provenance directories.
- `docs/`: Extensive project documentation and figures.

## 13. Publication
This repository supports our submission on hybrid fusion strategies for power quality disturbance classification. Please see `PUBLICATION_AUDIT.md` for our scientific audit framework.
