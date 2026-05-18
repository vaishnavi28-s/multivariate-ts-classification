# Experiments

`benchmark.ipynb` documents the model selection study conducted as part
of the MSc thesis at Bertelsmann Marketing Services (2025–2026).

## What this notebook contains

8 time-series classification models were benchmarked on **14,073 labelled
industrial events** using **3-fold stratified cross-validation**. The study
was designed to answer one question: does static metadata (machine ID,
line speed, paper grade, supplier) improve classification accuracy when
combined with the time-series signal from camera CV scores?

## Results

| Model | AUC | Notes |
|---|---|---|
| TST (Time Series Transformer) | 0.6742 | |
| FCN | 0.6752 | |
| ConvTimeNet | 0.6968 ± 0.0131 | |
| LSTM-FCN | 0.7013 | |
| InceptionTime | 0.7270 | |
| TapNet (TS only) | 0.7820 ± 0.0129 | |
| XGBoost (TS only) | 0.7895 ± 0.0026 | Ablation — no metadata |
| TapNet + metadata (early fusion) | 0.8100 ± 0.0072 | +0.028 from metadata |
| **XGBoost + metadata (full)** | **0.8595 ± 0.0009** | **Best — +0.070 from metadata** |

**Metadata contribution across both model families: consistent +0.03 to +0.07 AUC.**

The production pipeline in `src/` uses XGBoost + metadata — the best-performing
model, and the only one deployed as a production microservice.

## How to read this notebook

The notebook is structured as:
- **Cells 1–3**: data loading and CV fold preparation
- **Cells 4–12**: one cell per model (FCN, InceptionTime, LSTM-FCN, TapNet, ConvTimeNet, TST)
- **Cell 13**: XGBoost full model (TS + metadata)
- **Cell 14**: XGBoost feature importance analysis
- **Cell 15**: XGBoost ablation — TS only, no metadata
- **Cell 16**: TapNet + metadata early fusion

## Data note

Cell outputs are from runs on proprietary industrial sensor data
(Bertelsmann Marketing Services, Germany). All event IDs, file paths,
and internal machine identifiers have been removed from outputs.
Results are reported as aggregate metrics only.

The production pipeline is fully data-agnostic — see `data/README.md`
for the expected input schema.
