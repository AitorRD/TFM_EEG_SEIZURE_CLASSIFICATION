"""
LOSO (Leave-One-Subject-Out) with tsfresh features + TabPFN / TabICL.

Pipeline per fold:
  1. Extract tsfresh features per patient (cached to disk after first run).
  2. Combine remaining patients → train split.
  3. Fit SelectKBest(f_classif, k) on train only, transform train and test.
  4. Train TabPFN / TabICL on train features.
  5. Evaluate on the left-out patient (test).

Output CSV uses the same format as loso_foundation.py so plot_loso_results.py
works with the combined results of both scripts.

Usage:
    python loso_tabular.py                       # TabPFN + TabICL (default)
    python loso_tabular.py --models tabpfn       # TabPFN only
    python loso_tabular.py --k 30                # SelectKBest with k=30
    python loso_tabular.py --no-cache            # Force feature re-extraction
    python loso_tabular.py --append-csv images/results/loso/loso_fold_metrics.csv
"""

from __future__ import annotations

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mirrors config.yaml feature_extraction.custom_fc_parameters
TSFRESH_FC_PARAMS = {
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_peaks": [{"n": 3}, {"n": 5}],
    "root_mean_square": None,
    "autocorrelation": [{"lag": 1}],
}

MODEL_DISPLAY_NAMES = {
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
}

# Metrics reported — same keys as loso_foundation.py
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1 Score", "F1 Macro", "F1 Micro", "ROC AUC"]

EEG_CHANNELS = [
    "EEG Fp1", "EEG Fp2", "EEG F7", "EEG F3", "EEG Fz", "EEG F4", "EEG F8",
    "EEG T3", "EEG C3", "EEG Cz", "EEG C4", "EEG T4", "EEG T5", "EEG P3",
    "EEG Pz", "EEG P4", "EEG T6", "EEG O1", "EEG O2",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOSO tsfresh + TabPFN/TabICL seizure classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+",
        choices=["tabpfn", "tabicl"], default=["tabpfn", "tabicl"],
    )
    p.add_argument(
        "--raw-dir", type=str, default="data/raw/csv-data",
        help="Root directory with per-patient clipped CSVs (PN_XX/*_clipped.csv)",
    )
    p.add_argument("--window-size", type=int, default=1000,
                   help="Samples per window (default 1000 = 10 s at 100 Hz)")
    p.add_argument("--window-overlap", type=float, default=0.25,
                   help="Window overlap fraction (default 0.25)")
    p.add_argument(
        "--features-cache-dir", type=str, default="data/processed/loso_features",
        help="Directory for per-patient tsfresh feature cache",
    )
    p.add_argument(
        "--output-dir", type=str, default="images/results/loso",
    )
    p.add_argument(
        "--k", type=int, default=50,
        help="SelectKBest k (features kept per fold)",
    )
    p.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel jobs for tsfresh extraction",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Force re-extraction even if cache exists",
    )
    p.add_argument(
        "--random-state", type=int, default=42,
    )
    p.add_argument(
        "--append-csv", type=str, default=None,
        help="Path to existing loso_fold_metrics.csv to append results to",
    )
    p.add_argument(
        "--tabpfn-device", type=str, default="auto",
    )
    p.add_argument(
        "--tabicl-device", type=str, default="cuda:0",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading + windowing from raw/csv-data
# ---------------------------------------------------------------------------

_RENAME_COLS = {"EEG CZ": "EEG Cz", "EEG FP2": "EEG Fp2"}


def _window_df(df: pd.DataFrame, window_size: int, overlap: float) -> pd.DataFrame:
    """Sliding window per session."""
    step = int(window_size * (1 - overlap))
    windows = []
    for session_id in df["idSession"].unique():
        s_df = df[df["idSession"] == session_id].reset_index(drop=True)
        for start in range(0, len(s_df) - window_size + 1, step):
            w = s_df.iloc[start:start + window_size].copy()
            w["window_id"] = f"{session_id}_{start}"
            windows.append(w)
    return pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()


def load_and_window_patient(
    patient_id: str,
    raw_dir: str,
    window_size: int,
    overlap: float,
) -> pd.DataFrame:
    """Read clipped CSVs for one patient, apply sliding window."""
    patient_dir = Path(raw_dir) / patient_id
    dfs = []
    for csv_file in sorted(patient_dir.glob("*_clipped.csv")):
        try:
            d = pd.read_csv(csv_file)
            if len(d) > 0:
                dfs.append(d)
        except Exception as e:
            print(f"  [WARN] {csv_file.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    rename = {k: v for k, v in _RENAME_COLS.items() if k in df.columns}
    if rename:
        df.rename(columns=rename, inplace=True)
    df.fillna(0, inplace=True)
    return _window_df(df, window_size, overlap)


def list_patients(raw_dir: str) -> List[str]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    return sorted(d.name for d in raw_path.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# tsfresh feature extraction (with per-patient disk cache)
# ---------------------------------------------------------------------------

def _extract_for_df(df_patient: pd.DataFrame, n_jobs: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract tsfresh features from a patient's windowed DataFrame."""
    import multiprocessing
    if n_jobs <= 0:
        n_jobs = multiprocessing.cpu_count()

    # Use relative time within each window (avoids absolute timestamp drift)
    df_feat = df_patient.copy()
    df_feat["_t"] = df_feat.groupby("window_id").cumcount()
    df_feat = df_feat.rename(columns={"window_id": "_id"})

    signal_cols = [c for c in df_feat.columns if c in EEG_CHANNELS]
    if not signal_cols:
        raise ValueError("No EEG channel columns found in windowed data.")

    features = extract_features(
        df_feat[["_id", "_t"] + signal_cols],
        column_id="_id",
        column_sort="_t",
        default_fc_parameters=TSFRESH_FC_PARAMS,
        disable_progressbar=True,
        n_jobs=n_jobs,
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    tsfresh_impute(features)  # in-place, handles all-NaN columns correctly

    labels = df_patient.groupby("window_id")["Seizure"].max().astype(int)
    labels.index = labels.index.astype(str)
    features.index = features.index.astype(str)
    labels = labels.reindex(features.index)

    return features, labels


def load_or_extract_patient_features(
    patient_id: str,
    df_patient: pd.DataFrame,
    cache_dir: Path,
    n_jobs: int,
    no_cache: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    feat_path = cache_dir / f"features_{patient_id}.csv"
    lab_path  = cache_dir / f"labels_{patient_id}.csv"

    if not no_cache and feat_path.exists() and lab_path.exists():
        features = pd.read_csv(feat_path, index_col=0)
        labels   = pd.read_csv(lab_path,  index_col=0).squeeze()
        n_sz = int(labels.sum())
        print(
            f"  [{patient_id}] Cache loaded — {len(features)} windows "
            f"(seizure={n_sz}, non-seizure={len(features) - n_sz})"
        )
        return features, labels

    print(f"  [{patient_id}] Extracting tsfresh features ({len(df_patient):,} rows)...")
    features, labels = _extract_for_df(df_patient, n_jobs)

    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(feat_path)
    labels.to_csv(lab_path, header=True)

    n_sz = int(labels.sum())
    print(
        f"  [{patient_id}] Done — {len(features)} windows "
        f"(seizure={n_sz}, non-seizure={len(features) - n_sz})  cached → {feat_path.name}"
    )
    return features, labels


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(model_key: str, args: argparse.Namespace):
    if model_key == "tabpfn":
        if not TABPFN_AVAILABLE:
            raise ImportError("tabpfn not installed. Run: pip install tabpfn")
        return TabPFNClassifier(device=args.tabpfn_device, ignore_pretraining_limits=True)

    if model_key == "tabicl":
        if not TABICL_AVAILABLE:
            raise ImportError("tabicl not installed. Run: pip install tabicl")
        return TabICLClassifier(device=args.tabicl_device, random_state=args.random_state)

    raise ValueError(f"Unsupported model: {model_key}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    metrics = {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "F1 Macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 Micro":  float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    if y_score is not None and np.unique(y_true).size > 1:
        try:
            metrics["ROC AUC"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["ROC AUC"] = None
    else:
        metrics["ROC AUC"] = None
    return metrics


# ---------------------------------------------------------------------------
# Per-fold execution
# ---------------------------------------------------------------------------

def run_fold(
    patient_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_key: str,
    args: argparse.Namespace,
) -> Optional[Dict]:
    """Train and evaluate one LOSO fold for one tabular model."""

    # Align features across train/test (tsfresh may produce different cols per patient)
    common_cols = X_train.columns.intersection(X_test.columns)
    if len(common_cols) == 0:
        print(f"    [SKIP] {patient_id} / {model_key}: no common features")
        return None
    X_tr = X_train[common_cols].values.astype(np.float32)
    X_te = X_test[common_cols].values.astype(np.float32)
    y_tr = y_train.values.astype(np.int64)
    y_te = y_test.values.astype(np.int64)

    # Feature selection fitted on train only
    k = min(args.k, X_tr.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_tr = selector.fit_transform(X_tr, y_tr)
    X_te = selector.transform(X_te)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if len(np.unique(y_tr)) < 2:
        print(f"    [SKIP] {patient_id} / {model_key}: train has only one class")
        return None

    model = create_model(model_key, args)

    print(f"    Fitting {MODEL_DISPLAY_NAMES[model_key]} on {len(X_tr)} train windows...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_te)[:, 1]
        except Exception:
            pass

    metrics = compute_metrics(y_te, y_pred, y_score)

    return {
        "patient_id": patient_id,
        "metrics": metrics,
        "y_true": y_te,
        "y_pred": y_pred,
        "y_score": y_score,
    }


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------

def aggregate_metrics(fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for res in fold_results:
        for metric, value in res["metrics"].items():
            if value is not None:
                buckets.setdefault(metric, []).append(value)
    return {
        m: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for m, vals in buckets.items()
    }


def print_loso_summary(
    model_name: str,
    fold_results: List[Dict],
    aggregated: Dict[str, Dict[str, float]],
) -> None:
    print(f"\n{model_name} — LOSO Summary ({len(fold_results)} folds completed)")
    print("-" * 52)
    for metric, stats in aggregated.items():
        print(f"  {metric:<14} {stats['mean']:.4f} ± {stats['std']:.4f}")


def save_fold_metrics_csv(
    all_model_results: Dict[str, List[Dict]],
    output_dir: Path,
    append_path: Optional[str],
) -> Path:
    rows = []
    for model_name, fold_results in all_model_results.items():
        for res in fold_results:
            row = {"model": model_name, "patient": res["patient_id"]}
            row.update({k: v for k, v in res["metrics"].items() if v is not None})
            rows.append(row)

    new_df = pd.DataFrame(rows)

    if append_path and os.path.exists(append_path):
        existing = pd.read_csv(append_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        out_path = Path(append_path)
        combined.to_csv(out_path, index=False)
        print(f"  Results appended to: {out_path}")
    else:
        out_path = output_dir / "loso_fold_metrics.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(out_path, index=False)
        print(f"  Per-fold metrics saved: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loso_confusion_matrix(
    fold_results: List[Dict],
    model_name: str,
    save_path: Path,
) -> None:
    """Summed confusion matrix across all LOSO folds for one model."""
    total_cm = np.zeros((2, 2), dtype=np.int64)
    for res in fold_results:
        total_cm += confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])

    tn, fp, fn, tp = total_cm.ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(total_cm, cmap="Blues")
    ax.set_title(
        f"{model_name} — LOSO ({len(fold_results)} folds)\n"
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        fontsize=12,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Seizure", "Seizure"])
    ax.set_yticklabels(["No Seizure", "Seizure"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(total_cm[r, c]), ha="center", va="center", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir  = Path(args.features_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in args.models if m in MODEL_DISPLAY_NAMES]
    if not models:
        raise ValueError("No valid models specified.")

    print("=" * 60)
    print("  LOSO — TSFRESH + TABULAR MODELS — EEG SEIZURE CLASSIFICATION")
    print("=" * 60)
    print(f"  Models         : {', '.join(MODEL_DISPLAY_NAMES[m] for m in models)}")
    print(f"  SelectKBest k  : {args.k}")
    print(f"  Features cache : {cache_dir}")
    print(f"  Raw data dir   : {args.raw_dir}")
    print(f"  Window: {args.window_size} samples ({args.window_size // 100}s), overlap {args.window_overlap:.0%}")

    # ----------------------------------------------------------------
    # Step 1: discover patients and extract (or load) tsfresh features
    # ----------------------------------------------------------------
    patients = list_patients(args.raw_dir)
    print(f"\nPatients ({len(patients)}): {patients}")

    print("\nExtracting tsfresh features per patient (windowing on-the-fly from raw)...")
    patient_features: Dict[str, pd.DataFrame] = {}
    patient_labels:   Dict[str, pd.Series]    = {}

    for patient_id in patients:
        windowed = load_and_window_patient(
            patient_id, args.raw_dir, args.window_size, args.window_overlap,
        )
        if len(windowed) == 0:
            print(f"  [{patient_id}] No data — skipping")
            continue
        feats, labs = load_or_extract_patient_features(
            patient_id, windowed, cache_dir, args.n_jobs, args.no_cache,
        )
        patient_features[patient_id] = feats
        patient_labels[patient_id]   = labs

    patients = [p for p in patients if p in patient_features]  # keep only valid ones

    # ----------------------------------------------------------------
    # Step 3: LOSO loop
    # ----------------------------------------------------------------
    all_model_results: Dict[str, List[Dict]] = {}

    for model_key in models:
        model_name = MODEL_DISPLAY_NAMES[model_key]
        print(f"\n{'=' * 60}")
        print(f"  {model_name.upper()} — LOSO  ({len(patients)} folds)")
        print("=" * 60)

        fold_results: List[Dict] = []

        for fold_idx, patient_id in enumerate(patients):
            print(f"\n  Fold {fold_idx + 1}/{len(patients)}  test={patient_id}")

            # Test: left-out patient
            X_test = patient_features[patient_id]
            y_test = patient_labels[patient_id]

            # Train: all remaining patients combined
            train_feat_parts = [
                patient_features[p] for p in patients if p != patient_id
            ]
            train_lab_parts = [
                patient_labels[p] for p in patients if p != patient_id
            ]

            X_train = pd.concat(train_feat_parts, axis=0)
            y_train = pd.concat(train_lab_parts,  axis=0)

            # Drop windows where label is NaN (patients with no seizure info)
            valid_mask = y_train.notna()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            valid_test_mask = y_test.notna()
            X_test_clean = X_test[valid_test_mask]
            y_test_clean = y_test[valid_test_mask]

            if len(X_test_clean) == 0:
                print(f"    [SKIP] {patient_id}: no valid test samples")
                continue

            result = run_fold(
                patient_id=patient_id,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test_clean,
                y_test=y_test_clean,
                model_key=model_key,
                args=args,
            )

            if result is not None:
                fold_results.append(result)
                roc = result["metrics"].get("ROC AUC")
                roc_str = f"{roc:.4f}" if roc is not None else "N/A"
                print(
                    f"    F1={result['metrics'].get('F1 Score', 0):.4f}  "
                    f"F1-macro={result['metrics'].get('F1 Macro', 0):.4f}  "
                    f"ROC-AUC={roc_str}"
                )

        if not fold_results:
            print(f"  [WARN] No valid folds for {model_name}")
            continue

        aggregated = aggregate_metrics(fold_results)
        all_model_results[model_name] = fold_results
        print_loso_summary(model_name, fold_results, aggregated)

        cm_path = output_dir / "graphs" / f"loso_confusion_{model_key}.png"
        plot_loso_confusion_matrix(fold_results, model_name, cm_path)

        # Save concatenated predictions so plot_loso_results.py can draw ROC curves
        preds_dir = output_dir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        valid = [r for r in fold_results if r["y_score"] is not None]
        if valid:
            pd.DataFrame({
                "y_true":  np.concatenate([r["y_true"]  for r in valid]),
                "y_score": np.concatenate([r["y_score"] for r in valid]),
            }).to_csv(preds_dir / f"{model_key}_loso_predictions.csv", index=False)

    # ----------------------------------------------------------------
    # Step 4: save results
    # ----------------------------------------------------------------
    if not all_model_results:
        print("\n[ERROR] No results to save.")
        return

    csv_path = save_fold_metrics_csv(all_model_results, output_dir, args.append_csv)
    print(f"\nRun 'python plot_loso_results.py --csv {csv_path}' to generate plots.")

    print("\n" + "=" * 60)
    print("  LOSO TABULAR COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
