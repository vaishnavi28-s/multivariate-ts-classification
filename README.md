# multivariate-ts-classification

[![CI](https://github.com/VaishnaviS28/multivariate-ts-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/VaishnaviS28/multivariate-ts-classification/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end multivariate time-series fault classification pipeline with static metadata fusion — built and deployed as a production microservice at **Bertelsmann Marketing Services**.

---

## The problem

Industrial inspection systems generate two types of signals per event:

- **Time-series** — frame-by-frame classification scores from a vision system (Azure Computer Vision) across N camera frames per event
- **Static metadata** — machine ID, line speed, paper grade, supplier, detector position

Most time-series classifiers treat these separately or discard metadata entirely. This pipeline addresses the case where **fault classification requires signals from both domains simultaneously** — a pattern common in industrial deployments but underrepresented in published benchmarks.

**Quantified impact: metadata fusion adds +0.070 AUC to XGBoost and +0.028 AUC to TapNet — consistent across both model families and all 3 folds.**

---

## Results

Benchmarked on **14,073 labelled industrial events** using **3-fold stratified cross-validation**.

| Model | AUC | Notes |
|---|---|---|
| TST (Time Series Transformer) | 0.6742 | |
| FCN | 0.6752 | |
| ConvTimeNet | 0.6968 ± 0.0131 | |
| LSTM-FCN | 0.7013 | |
| InceptionTime | 0.7270 | |
| TapNet (TS only) | 0.7820 ± 0.0129 | Deep learning baseline |
| XGBoost (TS only) | 0.7895 ± 0.0026 | Ablation — metadata removed |
| TapNet + metadata fusion | 0.8100 ± 0.0072 | +0.028 from metadata |
| **XGBoost + metadata fusion** | **0.8595 ± 0.0009** | **Best — +0.070 from metadata** |

Full benchmarking experiment with all fold results, confusion matrices, and ROC curves: [`experiments/benchmark.ipynb`](experiments/benchmark.ipynb)

---

## Architecture

```
raw event JSONs (camera CV scores + metadata)
           │
           ▼
┌──────────────────────────────────────────┐
│              preprocessing.py            │
│  • validate frames                       │
│  • align 300-frame window to tear peak   │
│  • extract 68 numeric features           │
│  • parse metadata (unit stripping)       │
└────────────┬─────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
time-series        static metadata
features           (printer, grade,
(per-camera        speed, grammage,
 stats, slopes,    supplier, detector)
 entropy, drift)
    └────────┬────────┘
             ▼
┌──────────────────────────────────────────┐
│                 train.py                 │
│  • StandardScaler (train-only)           │
│  • OneHotEncoder  (train-only)           │
│  • XGBoost + scale_pos_weight            │
│  • threshold tuning on val set           │
│  • artefact saving (model+scaler+OHE+thr)│
└────────────┬─────────────────────────────┘
             ▼
┌──────────────────────────────────────────┐
│              inference.py                │
│  • load artefact                         │
│  • predict: green / yellow / red zones   │
│  • colour-coded Excel output             │
└────────────┬─────────────────────────────┘
             ▼
┌──────────────────────────────────────────┐
│                 api.py                   │
│  FastAPI  POST /predict                  │
│           POST /predict/batch            │
│           GET  /health                   │
└──────────────────────────────────────────┘
```

---

## Key design decisions

**Train-only normalisation** — scaler and OHE are fit exclusively on training data and applied to val/test. Prevents the data leakage that commonly inflates reported accuracy in industrial ML benchmarks.

**Decision boundary threshold tuning** — thresholds are tuned per fold on the validation set to maximise F1 on the minority fault class (~8.5% of events). Default 0.5 would significantly underperform on this imbalanced dataset.

**Metadata as a first-class input** — static metadata is encoded separately (OHE for categoricals, scaled for numerics) and fused at the model level alongside time-series features.

**Zone-based output** — predictions are reported in three confidence zones derived from the data distribution of 14,073 labelled events:

| Zone | Probability | Meaning |
|---|---|---|
| 🟢 Keine Reklamation | < 0.30 | 95.8% truly machine fault — reliable |
| 🟡 Unsicher | 0.30 – 0.69 | Genuinely ambiguous — review needed |
| 🔴 Reklamation | ≥ 0.70 | 62.7%+ truly paper fault |

---

## Data

This repository ships no data. The pipeline is data-agnostic — bring your own event JSONs.

See [`data/README.md`](data/README.md) for the full schema specification.

Originally developed on proprietary industrial sensor data at Bertelsmann Marketing Services, Germany.

---

## Quickstart

```bash
git clone https://github.com/VaishnaviS28/multivariate-ts-classification
cd multivariate-ts-classification
pip install -r requirements.txt
```

**Train:**
```bash
python -m src.cli train --data_dir /path/to/labelled/event/zips
```

**Score new events:**
```bash
python -m src.cli score
```

**Single event prediction:**
```bash
python -m src.cli predict --event_json /path/to/event.json
```

**Run as microservice:**
```bash
docker compose up
# → http://localhost:8000/predict
# → http://localhost:8000/docs
```

---

## Project structure

```
multivariate-ts-classification/
├── src/
│   ├── preprocessing.py   # frame validation, sequence building, feature extraction
│   ├── train.py           # CV pipeline, XGBoost training, artefact saving
│   ├── threshold.py       # decision boundary tuning on validation set
│   ├── inference.py       # artefact loading, batch scoring, single-event predict
│   ├── excel_export.py    # colour-coded Excel output (green/yellow/red zones)
│   ├── api.py             # FastAPI microservice
│   └── cli.py             # CLI: train / score / predict
├── experiments/
│   ├── benchmark.ipynb    # 8 models × 3 folds × 14,073 events — full results
│   └── README.md
├── tests/
│   ├── test_preprocessing.py
│   └── test_threshold.py
├── configs/
│   └── config.yaml
├── data/
│   └── README.md          # data schema + how to bring your own data
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Citation

```bibtex
@misc{sreekumar2026mvts,
  author = {Sreekumar, Vaishnavi},
  title  = {multivariate-ts-classification: End-to-end time-series fault
            classification with metadata fusion for industrial inspection},
  year   = {2026},
  url    = {https://github.com/VaishnaviS28/multivariate-ts-classification}
}
```

---

## License

MIT
