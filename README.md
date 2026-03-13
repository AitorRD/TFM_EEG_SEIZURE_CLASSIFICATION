# EEG Seizure Classification

![CI](https://github.com/aitor/TFM_EEG_SEIZURE_CLASSIFICATION/actions/workflows/ci.yml/badge.svg)

End-to-end framework for epileptic seizure detection from scalp EEG signals. Built as a Master's Thesis (TFM) project, it covers the full pipeline from raw EDF recordings to trained ML/DL models with explainability (XAI).

---

## Dataset

**Siena Scalp EEG Database** — 14 patients, 1–5 EDF recordings each, sampled at 100 Hz across 19 channels. All dates are de-identified.

- 📥 Download: [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- Place the raw EDF files under `data/raw/siena-scalp-eeg-database-1.0.0/`

---

## Project Structure

```
.
├── main.py                        # CLI entry point
├── config.yaml                    # Experiment configuration
├── data.py                        # Data pipeline (EDF → CSV → windows)
├── requirements.txt
├── .github/workflows/ci.yml       # CI: lint, syntax, imports, config validation
│
├── experimentation/               # Python package
│   ├── __init__.py
│   ├── utils.py                   # Config loading, helpers
│   ├── models.py                  # ML/DL model factory (LR, RF, SVC, KNN, XGB, CNN, LSTM, GRU)
│   ├── tuning.py                  # Optuna hyperparameter optimisation
│   ├── graphs.py                  # Result plots + raw EEG visualisation
│   ├── xai.py                     # SHAP, LIME, brain topomap
│   ├── experimentation.py         # Experiment orchestrator
│   └── dl_models.py               # PyTorch model definitions
│
└── data/
    ├── raw/                       # Original EDF/CSV files
    └── processed/                 # Windowed datasets, features, saved models
```

---

## Pipeline

### 1 — Data processing (`data.py`)

```
EDF files  →  CSV per session  →  train/val/test split  →  10-second windows
```

| Step | Script | Output |
|------|--------|--------|
| EDF → CSV | `data/processed/convertion-edf-csv.py` | `data/raw/csv-data/` |
| Concatenation & split (70/15/15) | `data/processed/concat.py` | `dataset_clipped/{train,val,test}.csv` |
| Windowing (10 s, 25% overlap) | `data/processed/window.py` | `data/processed/windowed/` |

EEG channels extracted: `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`

### 2 — Experimentation (`main.py`)

All models and hyperparameters are configured in `config.yaml`.

**ML models:** Logistic Regression, Random Forest, SVC, KNN, XGBoost  
**DL models:** CNN-1D, LSTM, GRU, EEG Transformer (d_model=64, 4 heads, 2 layers)

Feature extraction via **tsfresh**, selection via **SelectKBest (k=50)**, optimisation via **Optuna**.

### 3 — Evaluation & XAI

- Metrics: Accuracy, Precision, Recall, F1-Score (val + test)
- Plots: ROC curves, confusion matrices, metrics table
- Explainability: SHAP summary, LIME explanations, brain topomap

---

## Quickstart

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Full pipeline (train → eval → plots → xai)
python main.py

# Use a custom config
python main.py --config config.yaml

# Run a single phase
python main.py --only train
python main.py --only eval
python main.py --only plots
python main.py --only xai
```

### GPU selection

```bash
CUDA_VISIBLE_DEVICES=0 python main.py   # RTX 4000 Ada
CUDA_VISIBLE_DEVICES=1 python main.py   # RTX 3090
```

---

## Configuration

All experiment settings live in `config.yaml`:

| Section | Key fields |
|---------|-----------|
| `experiment` | `type` (ml/dl), `device`, `random_state`, `n_jobs` |
| `paths` | Data, features, labels, models, results directories |
| `feature_extraction` | tsfresh parameters |
| `feature_selection` | method, `k` |
| `cross_validation` | folds, scoring |
| `optuna` | `n_trials`, pruner, timeout |
| `models` | Per-model `enabled`, default params, Optuna search space |

---

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR to `main`:

| Job | What it checks |
|-----|---------------|
| **Lint** | flake8 on `main.py` + `experimentation/` |
| **Syntax** | `py_compile` on all `.py` files |
| **Imports** | All public package symbols importable |
| **Config** | `config.yaml` contains required top-level keys |

---

## Requirements

Python 3.12+. Main dependencies:

```
numpy, pandas, scikit-learn, xgboost
torch, skorch
mne, tsfresh
optuna
shap, lime
matplotlib, seaborn
pyyaml, joblib
```

Full list in `requirements.txt`.