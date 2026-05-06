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
    python loso_tabular.py --xai-enabled --xai-methods shap lime
    python loso_tabular.py --append-csv images/results/loso/loso_fold_metrics.csv
"""

from __future__ import annotations

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    p.add_argument(
        "--xai-enabled",
        action="store_true",
        help="Enable XAI per LOSO fold patient plus global averages",
    )
    p.add_argument(
        "--xai-methods",
        nargs="+",
        default=["shap", "lime"],
        choices=["shap", "lime"],
        help="XAI methods to run",
    )
    p.add_argument(
        "--xai-dir", type=str, default=None,
        help="Output directory for XAI (default: <output-dir>/xai/tabular)",
    )
    p.add_argument(
        "--xai-max-samples", type=int, default=256,
        help="Max test windows per patient used for XAI",
    )
    p.add_argument(
        "--xai-top-features", type=int, default=20,
        help="Top features shown in XAI plots",
    )
    p.add_argument(
        "--xai-patient-instances", type=int, default=24,
        help="Number of test windows explained per patient",
    )
    p.add_argument(
        "--xai-lime-perturbations", type=int, default=1000,
        help="Synthetic samples used by each LIME explanation",
    )
    p.add_argument(
        "--xai-shap-background", type=int, default=64,
        help="Background train windows used by SHAP KernelExplainer",
    )
    p.add_argument(
        "--xai-shap-nsamples", type=int, default=128,
        help="Kernel SHAP evaluation budget per explained window",
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(0)
    non_numeric_cols = [c for c in df.columns if c not in set(numeric_cols)]
    if non_numeric_cols:
        df.loc[:, non_numeric_cols] = df.loc[:, non_numeric_cols].fillna("")
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
# XAI helpers
# ---------------------------------------------------------------------------

def _xai_sample_indices(total: int, max_samples: int, seed: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False))


def _xai_choose_indices(scores: np.ndarray, n_instances: int, seed: int) -> np.ndarray:
    total = len(scores)
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if n_instances <= 0 or n_instances >= total:
        return np.arange(total, dtype=np.int64)

    selected: List[int] = []
    sorted_idx = np.argsort(scores)
    anchors = [int(sorted_idx[0]), int(sorted_idx[-1]), int(sorted_idx[len(sorted_idx) // 2])]
    for idx in anchors:
        if idx not in selected:
            selected.append(idx)
    if len(selected) >= n_instances:
        return np.asarray(sorted(selected[:n_instances]), dtype=np.int64)

    rng = np.random.default_rng(seed + 97)
    remaining = [i for i in range(total) if i not in selected]
    needed = max(0, n_instances - len(selected))
    if needed > 0 and remaining:
        sampled = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
        selected.extend(int(i) for i in sampled)
    return np.asarray(sorted(selected), dtype=np.int64)


def _xai_plot_importance_bars(
    feature_names: List[str],
    values: np.ndarray,
    title: str,
    xlabel: str,
    save_path: Path,
    top_k: int,
    signed: bool = False,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    if len(feature_names) == 0 or values.size == 0:
        return

    order = np.argsort(np.abs(values) if signed else values)[::-1]
    order = order[:min(top_k, len(order))]
    selected_features = [feature_names[i] for i in order]
    selected_values = values[order]

    if signed:
        colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in selected_values]
    else:
        colors = ["#34495e"] * len(selected_values)

    y_pos = np.arange(len(selected_features))
    plt.figure(figsize=(9, 6))
    plt.barh(y_pos, selected_values, color=colors)
    plt.yticks(y_pos, selected_features)
    plt.gca().invert_yaxis()
    if signed:
        plt.axvline(0.0, color="black", linewidth=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    XAI plot saved: {save_path}")


def _xai_save_importance_csv(
    importance: pd.Series,
    save_path: Path,
    include_signed: Optional[pd.Series] = None,
) -> None:
    if include_signed is None:
        df = importance.rename("importance").reset_index()
    else:
        signed = include_signed.reindex(importance.index)
        df = pd.DataFrame(
            {
                "feature": importance.index,
                "importance": importance.values,
                "signed_importance": signed.values,
            }
        )
    df.columns = ["feature"] + list(df.columns[1:])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"    XAI CSV saved: {save_path}")


def _xai_predict_proba_binary(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(np.asarray(X, dtype=np.float32))
    proba = np.asarray(raw, dtype=np.float32)

    if proba.ndim == 1:
        pos = np.clip(proba, 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    if proba.ndim == 2 and proba.shape[1] == 1:
        pos = np.clip(proba[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    if proba.ndim == 2 and proba.shape[1] >= 2:
        if hasattr(model, "classes_"):
            classes = list(getattr(model, "classes_"))
            pos_idx = classes.index(1) if 1 in classes else min(1, len(classes) - 1)
        else:
            pos_idx = 1
        pos = np.clip(proba[:, pos_idx], 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    raise RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}")


def _xai_parse_shap_binary(shap_values: Any, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[-1], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    arr = np.asarray(shap_values, dtype=np.float32)
    if arr.ndim == 3:
        class_idx = 1 if arr.shape[2] > 1 else 0
        arr = arr[:, :, class_idx]
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        raise RuntimeError(f"Unsupported SHAP array shape: {arr.shape}")

    if arr.shape[1] != n_features:
        raise RuntimeError(f"SHAP feature mismatch: {arr.shape[1]} vs expected {n_features}")
    return arr


def _xai_run_shap_for_patient(
    model_key: str,
    model_name: str,
    patient_id: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    score_signal: np.ndarray,
    args: argparse.Namespace,
    patient_dir: Path,
) -> Optional[pd.Series]:
    try:
        import shap
    except ImportError:
        print("    [INFO] SHAP not installed, skipping SHAP.")
        return None

    if len(X_test) < 3:
        print(f"    [INFO] {patient_id}: not enough windows for SHAP.")
        return None
    if not hasattr(model, "predict_proba"):
        print(f"    [INFO] {patient_id}: {model_name} has no predict_proba, SHAP skipped.")
        return None

    bg_idx = _xai_sample_indices(
        total=len(X_train),
        max_samples=max(2, args.xai_shap_background),
        seed=args.random_state + 11,
    )
    background = X_train[bg_idx]

    explain_idx = _xai_choose_indices(
        score_signal, args.xai_patient_instances, seed=args.random_state + 17
    )
    X_explain = X_test[explain_idx]

    def proba_fn(data: np.ndarray) -> np.ndarray:
        return _xai_predict_proba_binary(model, data)

    try:
        explainer = shap.KernelExplainer(proba_fn, background)
        shap_values = explainer.shap_values(
            X_explain,
            nsamples=max(16, args.xai_shap_nsamples),
        )
        shap_matrix = _xai_parse_shap_binary(shap_values, n_features=X_test.shape[1])
    except Exception as exc:
        print(f"    [INFO] SHAP failed for patient={patient_id}: {exc}")
        return None

    mean_abs = np.mean(np.abs(shap_matrix), axis=0).astype(np.float32)
    top_k = min(args.xai_top_features, len(feature_names), shap_matrix.shape[1])
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_matrix[:, top_idx],
        features=X_explain[:, top_idx],
        feature_names=top_names,
        plot_type="dot",
        max_display=top_k,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Beeswarm - {model_name} - {patient_id}")
    plt.tight_layout()
    beeswarm_path = patient_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    XAI plot saved: {beeswarm_path}")

    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    _xai_plot_importance_bars(
        feature_names=feature_names,
        values=mean_abs,
        title=f"SHAP Importance - {model_name} - {patient_id}",
        xlabel="Mean |SHAP value|",
        save_path=patient_dir / "shap_importance.png",
        top_k=args.xai_top_features,
        signed=False,
    )
    _xai_save_importance_csv(importance, patient_dir / "shap_importance.csv")
    return importance


def _xai_run_lime_for_patient(
    model_key: str,
    model_name: str,
    patient_id: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    score_signal: np.ndarray,
    args: argparse.Namespace,
    patient_dir: Path,
) -> Optional[pd.Series]:
    try:
        import lime.lime_tabular
    except ImportError:
        print("    [INFO] LIME not installed, skipping LIME.")
        return None

    if len(X_test) < 2:
        print(f"    [INFO] {patient_id}: not enough windows for LIME.")
        return None
    if not hasattr(model, "predict_proba"):
        print(f"    [INFO] {patient_id}: {model_name} has no predict_proba, LIME skipped.")
        return None

    def proba_fn(data: np.ndarray) -> np.ndarray:
        return _xai_predict_proba_binary(model, data)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["No Seizure", "Seizure"],
        mode="classification",
        discretize_continuous=True,
        random_state=args.random_state,
    )

    explain_idx = _xai_choose_indices(
        score_signal, args.xai_patient_instances, seed=args.random_state + 23
    )
    collected: List[pd.Series] = []

    for idx in explain_idx:
        exp = explainer.explain_instance(
            X_test[idx],
            proba_fn,
            num_features=min(args.xai_top_features, len(feature_names)),
            num_samples=max(200, args.xai_lime_perturbations),
            top_labels=2,
        )
        try:
            lime_pairs = exp.as_list(label=1)
        except Exception:
            lime_pairs = exp.as_list()

        local_weights: Dict[str, float] = {}
        for feature_desc, weight in lime_pairs:
            mapped = None
            for fname in feature_names:
                if fname in feature_desc:
                    mapped = fname
                    break
            if mapped is None:
                mapped = feature_desc
            local_weights[mapped] = float(weight)

        row = pd.Series(local_weights).reindex(feature_names, fill_value=0.0)
        collected.append(row)

    if not collected:
        print(f"    [INFO] {patient_id}: LIME produced no explanations.")
        return None

    avg_signed = pd.concat(collected, axis=1).mean(axis=1)
    avg_abs = avg_signed.abs().sort_values(ascending=False)

    _xai_plot_importance_bars(
        feature_names=feature_names,
        values=avg_signed.reindex(feature_names).to_numpy(dtype=np.float32),
        title=f"LIME Importance - {model_name} - {patient_id}",
        xlabel="Mean local contribution (negative to positive)",
        save_path=patient_dir / "lime_importance.png",
        top_k=args.xai_top_features,
        signed=True,
    )
    _xai_save_importance_csv(
        avg_abs,
        patient_dir / "lime_importance.csv",
        include_signed=avg_signed,
    )
    return avg_abs


def run_fold_xai_tabular(
    patient_id: str,
    model_key: str,
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_score: Optional[np.ndarray],
    feature_names: List[str],
    args: argparse.Namespace,
) -> Dict[str, pd.Series]:
    if not args.xai_enabled:
        return {}
    if len(feature_names) == 0:
        return {}

    xai_root = Path(args.xai_dir) if args.xai_dir else Path(args.output_dir) / "xai" / "tabular"
    patient_dir = xai_root / model_key / "patients" / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    test_idx = _xai_sample_indices(
        total=len(X_test),
        max_samples=args.xai_max_samples,
        seed=args.random_state + len(patient_id) * 29,
    )
    train_idx = _xai_sample_indices(
        total=len(X_train),
        max_samples=max(args.xai_max_samples, args.xai_shap_background),
        seed=args.random_state + len(patient_id) * 31,
    )

    X_te = X_test[test_idx]
    X_tr = X_train[train_idx]
    if y_score is not None:
        score_signal = np.asarray(y_score, dtype=np.float32)[test_idx]
    elif hasattr(model, "predict_proba"):
        score_signal = _xai_predict_proba_binary(model, X_te)[:, 1]
    else:
        score_signal = np.asarray(model.predict(X_te), dtype=np.float32)

    print(f"    Running XAI for patient {patient_id}...")
    out: Dict[str, pd.Series] = {}
    method_set = set(args.xai_methods)

    if "shap" in method_set:
        shap_importance = _xai_run_shap_for_patient(
            model_key=model_key,
            model_name=model_name,
            patient_id=patient_id,
            model=model,
            X_train=X_tr,
            X_test=X_te,
            feature_names=feature_names,
            score_signal=score_signal,
            args=args,
            patient_dir=patient_dir,
        )
        if shap_importance is not None:
            out["shap"] = shap_importance

    if "lime" in method_set:
        lime_importance = _xai_run_lime_for_patient(
            model_key=model_key,
            model_name=model_name,
            patient_id=patient_id,
            model=model,
            X_train=X_tr,
            X_test=X_te,
            feature_names=feature_names,
            score_signal=score_signal,
            args=args,
            patient_dir=patient_dir,
        )
        if lime_importance is not None:
            out["lime"] = lime_importance

    meta_path = patient_dir / "meta.csv"
    pd.DataFrame(
        {
            "patient": [patient_id],
            "model": [model_name],
            "num_test_windows": [len(X_test)],
            "num_xai_windows": [len(X_te)],
            "xai_methods": [",".join(sorted(out.keys())) if out else ""],
        }
    ).to_csv(meta_path, index=False)

    return out


def save_global_xai_averages_tabular(
    model_key: str,
    model_name: str,
    fold_results: List[Dict],
    args: argparse.Namespace,
) -> None:
    if not args.xai_enabled:
        return

    xai_root = Path(args.xai_dir) if args.xai_dir else Path(args.output_dir) / "xai" / "tabular"
    global_dir = xai_root / model_key / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    for method in ["shap", "lime"]:
        series_list: List[pd.Series] = []
        for res in fold_results:
            xai_obj = res.get("xai")
            if isinstance(xai_obj, dict) and method in xai_obj:
                series = xai_obj[method]
                if isinstance(series, pd.Series) and not series.empty:
                    series_list.append(series.rename(res["patient_id"]))

        if not series_list:
            continue

        matrix = pd.concat(series_list, axis=1).fillna(0.0)
        mean_importance = matrix.mean(axis=1).sort_values(ascending=False)
        std_importance = matrix.std(axis=1).reindex(mean_importance.index)

        csv_path = global_dir / f"{method}_importance_mean.csv"
        pd.DataFrame(
            {
                "feature": mean_importance.index,
                "mean_importance": mean_importance.values,
                "std_importance": std_importance.values,
            }
        ).to_csv(csv_path, index=False)
        print(f"  XAI global CSV saved: {csv_path}")

        _xai_plot_importance_bars(
            feature_names=list(mean_importance.index),
            values=mean_importance.values.astype(np.float32),
            title=f"{method.upper()} Mean Importance - {model_name} (all patients)",
            xlabel="Mean importance across patients",
            save_path=global_dir / f"{method}_importance_mean.png",
            top_k=args.xai_top_features,
            signed=False,
        )


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
    selected_mask = selector.get_support()
    selected_feature_names = common_cols[selected_mask].tolist()

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
            y_score = _xai_predict_proba_binary(model, X_te)[:, 1]
        except Exception:
            pass

    metrics = compute_metrics(y_te, y_pred, y_score)
    xai_outputs = run_fold_xai_tabular(
        patient_id=patient_id,
        model_key=model_key,
        model_name=MODEL_DISPLAY_NAMES[model_key],
        model=model,
        X_train=X_tr,
        X_test=X_te,
        y_score=y_score,
        feature_names=selected_feature_names,
        args=args,
    )

    return {
        "patient_id": patient_id,
        "metrics": metrics,
        "y_true": y_te,
        "y_pred": y_pred,
        "y_score": y_score,
        "xai": xai_outputs,
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
    if args.xai_dir is None:
        args.xai_dir = str(output_dir / "xai" / "tabular")

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
    print(f"  XAI enabled    : {args.xai_enabled}")
    if args.xai_enabled:
        print(f"  XAI methods    : {', '.join(args.xai_methods)}")
        print(f"  XAI output dir : {args.xai_dir}")

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

        save_global_xai_averages_tabular(
            model_key=model_key,
            model_name=model_name,
            fold_results=fold_results,
            args=args,
        )

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
