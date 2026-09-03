# Frozen-DASNet DualPQ
### Decoupled Hybrid Power Quality Disturbance Classification Under Severe Noise

This repository contains the code and evaluation framework for classifying 29 distinct Power Quality (PQ) disturbances under extreme noise conditions. The benchmark evaluates deep, classical, and hybrid architectures using a rigorous waveform-grouped protocol across five independent training seeds.

## Where to Start

- For a quick plain-English overview: [START_HERE.md](docs/START_HERE.md)
- For the final scientific validation audit: [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md)
- For model provenance: **Model Provenance & Our Contributions** (below)
- For replication instructions: [REPLICATION_GUIDE.md](docs/REPLICATION_GUIDE.md)
- For historical investigation of the classical baseline: [REPORT.md](docs/REPORT.md)

## Research Question
Can we improve the robustness of Power Quality Disturbance classification under severe noise (e.g., 10 dB and 0 dB) by fusing deep learned representations with classical signal-processing features? Furthermore, does decoupling the optimization of the deep and classical branches prevent training instability?

## Final Method
Frozen-DASNet DualPQ is our proposed fusion strategy. It combines a pretrained, frozen DASNet representation with a trainable physical-feature branch and classification head. The study evaluates whether decoupling the pretrained deep representation from end-to-end joint optimization improves robustness and stability under the grouped noise-aware benchmark.

## Architecture

```text
1024-point waveform
        ↓
Pretrained DASNet
        ↓
Frozen deep representation
        ↓
                    ┌───────────────┐
191 physical       │               │
features ───────→ Trainable MLP    │
                    │               │
                    └───────┬───────┘
                            ↓
                 Feature Fusion
                            ↓
                  Classification Head
```

## Model Provenance & Our Contributions

We do not claim ownership or novelty for the underlying DASNet/DST architecture, MGCNN-SDTransformer architecture, or established classical signal-processing features. Our methodological contribution is the specific decoupled frozen-DASNet fusion strategy and its evaluation under the controlled benchmark protocol.

| Component | Provenance | Role |
|---|---|---|
| **Classical Ensemble** | Our implementation of established methods | Classical baseline |
| **DASNet** | External/existing architecture | Deep baseline / pretrained representation |
| **MGCNN-SDTransformer** | External published method (Jiang et al., 2025) | Benchmark |
| **Original DualPQ-D** | Our proposed initial architecture | Initial hybrid experiment |
| **Frozen-DASNet DualPQ** | Our final proposed method | Main contribution |

> [!WARNING]
> **IMPORTANT FOR PAPER AUTHORS:**
> DASNet, DST, and MGCNN-SDTransformer must be presented as existing methods/baselines. They are not our inventions. The original DualPQ-D and Frozen-DASNet DualPQ are our proposed research contributions. Classical physical features are established representations; their integration into our proposed fusion framework is part of our work.

## Final Results

Results are reported as the mean ± sample standard deviation (ddof=1) of the Macro-F1 score across five independent seeds.

| Model | Macro-F1 |
|---|---:|
| **Frozen-DASNet DualPQ** | **74.46% ± 1.08%** |
| Classical Ensemble | 71.52% ± 0.84% |
| DASNet | 69.72% ± 9.11% |
| MGCNN-SDTransformer | 66.59% ± 0.98% |
| Original DualPQ-D | 61.63% ± 15.58% |

## SNR Robustness

The table below details the performance of the final proposed method (Frozen-DASNet DualPQ) as noise increases. While highly robust down to 20 dB, extreme noise at 0 dB remains a fundamentally difficult problem.

| SNR | Frozen-DASNet DualPQ |
|---|---:|
| Clean | 91.18% |
| 40 dB | 91.69% |
| 30 dB | 89.56% |
| 20 dB | 82.36% |
| 10 dB | 62.36% |
| 0 dB | 27.00% |

## Research Evolution

Our methodology evolved sequentially through rigorous testing:
1. **Classical Baseline**: Found to be robust, but fundamentally limited on certain indistinguishable classes.
2. **DASNet Baseline**: Implemented an existing deep architecture, but found it severely degraded under extreme noise (10 dB, 0 dB).
3. **Original DualPQ-D**: Combined deep and classical features end-to-end, but observed severe seed-to-seed instability (±15.58%).
4. **Frozen-DASNet DualPQ**: Decoupled the optimization by freezing the deep branch, resulting in stable, superior performance. The observed seed instability in the original architecture is consistent with difficult joint optimization dynamics.

## Evaluation Protocol

Our rigorously controlled evaluation protocol includes:
- **29 Classes**: Synthetic dataset generated using the mathematical model by Igual et al.
- **Waveform Length**: 1024 points (downsampled from generated 1280 to standard dimensions).
- **Grouped Stratified Split**: Waveform-grouped train/validation/test splitting prevents cross-variant leakage between partitions. Noise variants of the same base waveform are kept in the same split.
- **Severe Noise Conditions**: Clean, 40 dB, 30 dB, 20 dB, 10 dB, and 0 dB evaluated.
- **Five Independent Seeds**: All deep/hybrid models trained across seeds 0, 1, 2, 3, and 4 to accurately measure variance.

## Limitations

- The dataset is simulated/synthetic rather than a large real-world measurement corpus.
- Performance decreases substantially at 0 dB, remaining an unsolved difficulty.
- Five seeds provide useful stability evidence but are still a relatively small sample.
- The MGCNN benchmark uses a different original experimental protocol from its published paper, so direct numerical comparison with published accuracy is not apples-to-apples.
- The final method demonstrates improved stability/performance under this benchmark, but broader generalization requires external real-world datasets.

## Reproducibility & Installation

The project uses two separate environments to avoid dependency conflicts.
For classical baselines and dataset generation:
```bash
pip install -r requirements.txt
```
For deep-learning models (DASNet, MGCNN, DualPQ):
```bash
pip install -r requirements-deep.txt
```

See [REPLICATION_GUIDE.md](docs/REPLICATION_GUIDE.md) for full commands.

## License / Citation

This repository is distributed under the GPL-3.0 License due to the underlying `pqmodel.py` generator ported from MATLAB. 

If you use or modify the dataset model, please cite:
> R. Igual, C. Medrano, F. J. Arcega, G. Mantescu, "Integral mathematical model of power quality disturbances", 18th International Conference on Harmonics and Quality of Power (ICHQP), 2018.

If you discuss the MGCNN-SDTransformer baseline, please cite the original Jiang et al. (2025) authors appropriately.
