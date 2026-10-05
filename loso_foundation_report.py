"""
LOSO with tabular foundation models (TabICL, TabPFN v2/v3) — .py counterpart
of loso_foundation.ipynb, parameterized by --dataset (siena | chbmit) and
extended to write a Markdown/txt report meant as raw input for a later XAI
study (per-fold metrics, aggregated metrics, confusion totals, and the
tsfresh features most consistently kept by SelectKBest across folds).

Target: predict whether there will be a seizure in the *next* window,
defined as >= seizure-threshold fraction of that window's samples having
Seizure = 1.

Usage:
    python loso_foundation_report.py --dataset siena
    python loso_foundation_report.py --dataset chbmit --models tabicl tabpfnv2
    python loso_foundation_report.py --dataset chbmit --no-cache --top-features 40
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import pathlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute

# Timestamp + force-flush every print() in this module. This run has died
# silently mid-LOSO more than once with an empty log despite `python -u` —
# piping through an external `awk`/`tee` chain added buffering stages outside
# our control. Timestamping here removes the need for that awk stage entirely
# (plain `python script.py 2>&1 | tee log.txt` is enough) and guarantees each
# line is on disk the instant it's printed, regardless of the shell pipeline.
_print = print


def print(*args, **kwargs):  # noqa: A001 - intentional shadow, see comment above
    kwargs.setdefault("flush", True)
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    _print(ts, *args, **kwargs)

try:
    from tabicl import TabICLClassifier
    HAS_TABICL = True
except ImportError:
    HAS_TABICL = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

try:
    from tabfm import TabFMClassifier
    from tabfm import tabfm_v1_0_0_pytorch as _tabfm_backend
    HAS_TABFM = True
except ImportError:
    HAS_TABFM = False

try:
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
    from autogluon.tabular.models.mitra._internal.config.enums import Task as _MitraTask
    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import (
        DatasetFinetune as _MitraDatasetFinetune,
    )
    HAS_MITRA = True
except ImportError:
    HAS_MITRA = False


def _mitra_getitem_keep_minority(self, idx):
    """Replacement for Mitra's DatasetFinetune.__getitem__.

    Mitra is given the full train set (no cap from us), but when that OOMs it
    halves `max_samples_support` until it fits (~4-5k rows on a 24GB card with
    200 features) and then draws the support rows uniformly at random for each
    query batch, keeping only ~1% of CHB-MIT's seizure windows. This keeps
    every minority-class row and samples only the majority class; everything
    else matches the original method.
    """
    n, size = self.n_samples_support, self.support_size
    if size < n and self.cfg.task == _MitraTask.CLASSIFICATION:
        y = np.asarray(self.y_support)
        values, counts = np.unique(y, return_counts=True)
        majority = values[counts.argmax()]
        minority_idx = np.flatnonzero(y != majority)
        majority_idx = np.flatnonzero(y == majority)
        if len(minority_idx) >= size:
            support_indices = self.rng.choice(minority_idx, size=size, replace=False)
        else:
            support_indices = self.rng.permutation(np.concatenate([
                minority_idx,
                self.rng.choice(majority_idx, size=size - len(minority_idx), replace=False),
            ]))
    else:
        support_indices = self.rng.choice(n, size=size, replace=False)

    return {
        "x_support": torch.as_tensor(self.x_support[support_indices]),
        "y_support": torch.as_tensor(self.y_support[support_indices]),
        "x_query": torch.as_tensor(self.x_queries[idx]),
        "y_query": torch.as_tensor(self.y_queries[idx]),
    }


if HAS_MITRA:
    import torch
    _MitraDatasetFinetune.__getitem__ = _mitra_getitem_keep_minority

# TabFM's pretrained weights are a separate load step from the classifier
# constructor (unlike TabPFN/TabICL, which auto-download lazily inside their
# own constructor); load once and reuse across every LOSO fold.
_TABFM_MODEL = None


def _get_tabfm_backend_model():
    """Returns the cached TabFM weights, moved to the GPU for this fit.

    load() defaults to CPU (unlike TabPFN/TabICL, which pick CUDA themselves),
    and on CPU a ~55k-row CHB-MIT fold takes hours. The weights are parked back
    on CPU by _release_gpu() after each fit: left on the GPU they took ~3GB that
    TabPFNv2 needed on the next fold (OOM from fold 5 onwards)."""
    global _TABFM_MODEL
    import torch
    if _TABFM_MODEL is None:
        _TABFM_MODEL = _tabfm_backend.load()
    if torch.cuda.is_available():
        _TABFM_MODEL.to("cuda")
    return _TABFM_MODEL


def _release_gpu() -> None:
    """Frees GPU memory held after a model's fit/predict so the next model
    starts from an empty card."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if _TABFM_MODEL is not None:
        _TABFM_MODEL.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_tabpfn_token() -> None:
    """TabPFN's hosted inference needs TABPFN_TOKEN; pick it up from ~/.bashrc
    if it's not already in the environment (mirrors loso_foundation.ipynb)."""
    if "TABPFN_TOKEN" in os.environ:
        return
    try:
        bashrc = (pathlib.Path.home() / ".bashrc").read_text()
        m = re.search(r'export TABPFN_TOKEN="([^"]+)"', bashrc)
        if m:
            os.environ["TABPFN_TOKEN"] = m.group(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset configuration (mirrors the DATASET switch in loso_foundation.ipynb)
# ---------------------------------------------------------------------------

# CHB-MIT has ~5x more sessions than Siena and, even after capping interictal
# files to 30 min (see convertion-edf-csv-v2.py), produces far more 30s
# windows. EfficientFCParameters (~782 stats/channel, ~18k features for 23
# channels) took ~12h for a single patient there. This is the same reduced
# set loso_tabular.py already uses (mirrors config.yaml's
# feature_extraction.custom_fc_parameters) — ~8 stats/channel instead of 782.
LIGHT_FC_PARAMETERS = {
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_peaks": [{"n": 3}, {"n": 5}],
    "root_mean_square": None,
    "autocorrelation": [{"lag": 1}],
}

DATASET_CONFIGS = {
    "siena": {
        "data_dir": "data/processed/dataset_clipped_30",
        "features_dir": "data/processed/tsfresh_features",
        "output_dir": "images/results/loso_tabular_tpfn_icl",
        "eeg_channels": [
            "EEG Fp1", "EEG Fp2", "EEG F7",  "EEG F3",  "EEG Fz",  "EEG F4", "EEG F8",
            "EEG T3",  "EEG C3",  "EEG Cz",  "EEG C4",  "EEG T4",  "EEG T5", "EEG P3",
            "EEG Pz",  "EEG P4",  "EEG T6",  "EEG O1",  "EEG O2",
        ],
        "rename_cols": {"EEG CZ": "EEG Cz", "EEG FP2": "EEG Fp2"},
        "fc_parameters": None,  # None = EfficientFCParameters(); already proven feasible here
    },
    "chbmit": {
        "data_dir": "data/raw/csv-data-v2",
        "features_dir": "data/processed/tsfresh_features_chbmit",
        "output_dir": "images/results/loso_tabular_tpfn_icl_chbmit",
        # Bipolar montage, not perfectly fixed across patients (e.g. chb12
        # switches to a CS2-referential montage midway through its session).
        # MNE also renames the duplicated "T8-P8" channel found in several
        # files to "T8-P8-0"/"T8-P8-1".
        "eeg_channels": [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ",
            "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
            "T8-P8-0", "T8-P8-1",
        ],
        "rename_cols": {},
        "fc_parameters": LIGHT_FC_PARAMETERS,
    },
}

MODEL_DISPLAY_NAMES = {
    "tabicl": "TabICL", "tabpfnv2": "TabPFNv2", "tabpfnv3": "TabPFNv3", "tabfm": "TabFM",
    "mitra": "Mitra",
}

# TabICL and TabPFN (with its own memory-saving mode) handle CHB-MIT's full
# LOSO train sets (~55k rows). TabFM keeps a per-cell (row x feature)
# embedding of the whole table on the GPU with no offloading and OOMs even on
# a 24GB card at that size, so only it gets subsampled before fit(). Mitra
# gets the full train set too and sizes its context itself (see _build_model
# and _mitra_getitem_keep_minority).
IN_CONTEXT_ROW_LIMITED = {"tabfm"}


def _context_row_cap(model_key: str, max_incontext_rows: int) -> int:
    """Row cap for this model's in-context train set; 0 = full train set."""
    if model_key in IN_CONTEXT_ROW_LIMITED:
        return max_incontext_rows
    return 0


def subsample_train(
    X_train: np.ndarray, y_train: np.ndarray, max_rows: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_train) <= max_rows:
        return X_train, y_train
    # Seizure windows are <1% of CHB-MIT train rows; a uniform sample would
    # throw most of them away. Keep every positive and cut only negatives
    # (falls back to sampling positives if they alone exceed the cap).
    rng = np.random.RandomState(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train != 1)
    if len(pos_idx) >= max_rows:
        idx = rng.choice(pos_idx, size=max_rows, replace=False)
    else:
        neg_keep = rng.choice(neg_idx, size=max_rows - len(pos_idx), replace=False)
        idx = rng.permutation(np.concatenate([pos_idx, neg_keep]))
    return X_train[idx], y_train[idx]


def _model_available(model_key: str) -> bool:
    if model_key == "tabicl":
        return HAS_TABICL
    if model_key == "tabfm":
        return HAS_TABFM
    if model_key == "mitra":
        return HAS_MITRA
    return HAS_TABPFN


def _build_model(model_key: str, seed: int, n_context_rows: int):
    if model_key == "tabicl":
        return TabICLClassifier(random_state=seed)
    if model_key == "tabpfnv2":
        return TabPFNClassifier.create_default_for_version(
            "v2", random_state=seed, ignore_pretraining_limits=True,
        )
    if model_key == "tabpfnv3":
        return TabPFNClassifier.create_default_for_version(
            "v3", random_state=seed, ignore_pretraining_limits=True,
        )
    if model_key == "tabfm":
        return TabFMClassifier(model=_get_tabfm_backend_model(), random_state=seed)
    if model_key == "mitra":
        # Mitra fine-tunes its weights on the train set by default (50
        # gradient steps); disabled so it is pure in-context learning like
        # the other models.
        model = MitraClassifier(fine_tune=False, seed=seed, verbose=False)
        # Mitra hard-codes max_samples_support=8192 in its config; start it
        # at the full train set instead. On OOM Mitra itself halves this and
        # retries (printing "Reducing max_samples_support ..."), so it ends
        # up with the largest context that fits; the patched sampler keeps
        # every seizure window in it.
        create_config = model._create_config

        def _create_config(task, dim_output, time_limit=None):
            cfg, model_cls = create_config(task, dim_output, time_limit)
            cfg.hyperparams["max_samples_support"] = n_context_rows
            return cfg, model_cls

        model._create_config = _create_config
        return model
    raise ValueError(f"Unknown model: {model_key}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOSO TabICL/TabPFN seizure classification with a Markdown/txt "
                    "report for downstream XAI work",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="siena")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Overrides the --dataset default clipped-CSV directory")
    p.add_argument("--features-dir", type=str, default=None,
                   help="Overrides the --dataset default tsfresh feature cache directory")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Overrides the --dataset default results directory")
    p.add_argument("--models", nargs="+", choices=list(MODEL_DISPLAY_NAMES),
                   default=list(MODEL_DISPLAY_NAMES))
    p.add_argument("--window-sec", type=int, default=30,
                   help="Window length in seconds; label = seizure in the *next* window")
    p.add_argument(
        "--window-overlap", type=float, default=0.25,
        help="Overlap fraction between consecutive current-windows (0 = old non-overlapping "
             "behavior). Matches loso_tabular.py's default; more (overlapping) training "
             "windows without needing more raw data. Encoded into the feature cache filename "
             "so different values don't silently reuse a stale cache.",
    )
    p.add_argument("--fs", type=int, default=100, help="Sampling rate of the clipped CSVs (Hz)")
    p.add_argument("--seizure-threshold", type=float, default=0.5)
    p.add_argument("--max-features", type=int, default=200)
    p.add_argument(
        "--max-incontext-rows", type=int, default=0,
        help="Row cap applied only to TabFM before fit() — it holds the full training "
             "set as GPU transformer context and OOMs on large LOSO folds. Keeps every "
             "positive and subsamples negatives. 0 (default) = no cap, use the full train "
             "set. TabICL/TabPFN/Mitra always get the full train set (Mitra shrinks its "
             "own context on OOM, keeping every positive).",
    )
    p.add_argument(
        "--min-channel-coverage", type=float, default=0.5,
        help="Minimum fraction of the dataset's EEG channels a session must have; "
             "sessions below this are skipped instead of zero-filled",
    )
    p.add_argument("--n-jobs", type=int, default=4, help="tsfresh parallel jobs")
    p.add_argument(
        "--fc-parameters", choices=["auto", "efficient", "light"], default="auto",
        help="tsfresh feature set. auto = the dataset default (siena: efficient, chbmit: light). "
             "Overriding it writes to separate cache/output dirs (suffix _efficient/_light) "
             "unless --features-dir/--output-dir are given.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true", help="Force tsfresh re-extraction")
    p.add_argument(
        "--no-resume-loso", action="store_true",
        help="Ignore any (model, patient) results already checkpointed under "
             "<output-dir>/fold_results/ and recompute every fold from scratch",
    )
    p.add_argument(
        "--top-features", type=int, default=25,
        help="How many most-frequently-selected features to list in the report",
    )
    p.add_argument("--report-format", choices=["md", "txt"], default="md")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading + windowing
# ---------------------------------------------------------------------------

def create_windows_for_patient(
    patient_id: str,
    data_dir: Path,
    eeg_channels: List[str],
    rename_cols: Dict[str, str],
    window_size: int,
    seizure_threshold: float,
    min_channel_coverage: float,
    window_overlap: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[dict]]]:
    patient_dir = data_dir / patient_id
    all_windows, all_labels, all_meta = [], [], []

    for csv_file in sorted(patient_dir.glob("*_clipped.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  [WARN] {csv_file.name}: {e}")
            continue
        if df.empty:
            continue

        rename = {k: v for k, v in rename_cols.items() if k in df.columns}
        if rename:
            df.rename(columns=rename, inplace=True)

        present = sum(1 for ch in eeg_channels if ch in df.columns)
        if present / len(eeg_channels) < min_channel_coverage:
            print(
                f"  [WARN] {csv_file.name}: only {present}/{len(eeg_channels)} "
                f"expected EEG channels present (montage mismatch?) — skipping"
            )
            continue

        for ch in eeg_channels:
            if ch not in df.columns:
                df[ch] = 0.0
        df.fillna(0, inplace=True)

        eeg     = df[eeg_channels].values.astype(np.float32)
        seizure = df["Seizure"].values
        T       = len(eeg)

        # Current-window start slides by `step` (< window_size when overlap > 0),
        # generating more (overlapping) training windows. The *label* window
        # immediately following each one stays a full, non-overlapping
        # window_size chunk — only the input window overlaps.
        step    = max(1, int(window_size * (1 - window_overlap)))
        session = csv_file.stem

        i = 0
        t_cur = 0
        while t_cur + 2 * window_size <= T:
            t_next = t_cur + window_size

            next_seizure = seizure[t_next: t_next + window_size]
            label        = int(next_seizure.mean() >= seizure_threshold)

            all_windows.append(eeg[t_cur:t_next])
            all_labels.append(label)
            all_meta.append({"patient_id": patient_id, "session": session, "window_idx": i})

            i += 1
            t_cur += step

    if not all_windows:
        return None, None, None

    return np.stack(all_windows), np.array(all_labels, np.int64), all_meta


def list_patients(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# tsfresh feature extraction (per-patient disk cache)
# ---------------------------------------------------------------------------

def windows_to_wide_df(windows: np.ndarray, window_ids: List[str], eeg_channels: List[str]) -> pd.DataFrame:
    N, T, C = windows.shape
    df_dict = {
        "id":   np.repeat(window_ids, T),
        "time": np.tile(np.arange(T), N),
    }
    for c_idx, ch in enumerate(eeg_channels):
        df_dict[ch] = windows[:, :, c_idx].ravel()
    return pd.DataFrame(df_dict)


def _align_cached_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Caches written before the reindex fix in extract_and_cache store
    tsfresh's rows sorted by window_id *as strings* (p_0, p_1, p_10, p_100, ...)
    but the label column in window order (0, 1, 2, ...), so each row carried
    another window's label. Those caches are detected by their row order and
    the labels re-aligned; caches written after the fix are in window order
    and returned unchanged."""
    idx = features["window_id"].str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    if (np.diff(idx) > 0).all():
        return features
    features = features.copy()
    features["label"] = features["label"].to_numpy()[idx]
    return features


def extract_and_cache(
    patient_id: str,
    data_dir: Path,
    features_dir: Path,
    eeg_channels: List[str],
    rename_cols: Dict[str, str],
    window_size: int,
    seizure_threshold: float,
    min_channel_coverage: float,
    n_jobs: int,
    no_cache: bool,
    fc_parameters: Optional[dict],
    window_overlap: float = 0.0,
) -> Optional[pd.DataFrame]:
    # Overlap changes how many windows/rows a patient produces, so it must be
    # part of the cache key — otherwise a rerun with a different overlap would
    # silently load stale features computed under the old windowing.
    overlap_suffix = "" if window_overlap == 0 else f"_ov{window_overlap:g}"
    feat_path = features_dir / f"{patient_id}{overlap_suffix}.parquet"
    if not no_cache and feat_path.exists():
        print(f"  {patient_id}: loading from cache...")
        return _align_cached_labels(pd.read_parquet(feat_path))

    windows, labels, meta = create_windows_for_patient(
        patient_id, data_dir, eeg_channels, rename_cols,
        window_size, seizure_threshold, min_channel_coverage, window_overlap,
    )
    if windows is None:
        print(f"  {patient_id}: no data, skipping")
        return None

    N    = len(windows)
    wids = [f"{patient_id}_{i}" for i in range(N)]

    print(f"  {patient_id}: extracting features ({N} windows)...", flush=True)
    wide_df = windows_to_wide_df(windows, wids, eeg_channels)

    features = extract_features(
        wide_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_parameters if fc_parameters is not None else EfficientFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True,
        impute_function=tsfresh_impute,
    )

    # tsfresh returns rows sorted by id as strings (p_0, p_1, p_10, ...);
    # back to window order so the positional labels below line up.
    features = features.reindex(wids)
    features.index.name = "window_id"
    features = features.reset_index()
    features["patient_id"] = patient_id
    features["label"]      = labels

    features_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feat_path, index=False)
    print(f"  {patient_id}: {features.shape[1] - 3} features saved")
    return features


# ---------------------------------------------------------------------------
# LOSO
# ---------------------------------------------------------------------------

META_COLS = {"window_id", "patient_id", "label"}


PREPROCESS_BLOCK_COLS = 1000


def _impute_and_scale(
    X_train: np.ndarray, X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute, drop constant columns and standardize. Returns the kept
    column mask (relative to the input columns) alongside both splits."""
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_test  = imp.transform(X_test)
    keep = ~np.isnan(imp.statistics_)  # all-NaN columns are dropped by the imputer

    var = VarianceThreshold(threshold=0.0)
    X_train = var.fit_transform(X_train)
    X_test  = var.transform(X_test)
    keep[keep] = var.get_support()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, keep


def preprocess_fold(
    train_parts: List[np.ndarray], y_train: np.ndarray, X_test: np.ndarray,
    feature_names: List[str], max_features: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fit the preprocessing pipeline on train, transform both splits, and
    track which feature names survive each step (needed for the report).

    `train_parts` are the per-patient train arrays (infs already NaN). The full
    ~55k x 18.6k train matrix is never built: imputer, variance filter, scaler
    and f_classif are all per-column, so pass 1 scores the features in column
    blocks and pass 2 re-fits the pipeline on just the selected columns, giving
    the same result as running it on the full matrix. Building it (plus the
    sklearn copies) spiked memory pressure enough for systemd-oomd to kill the
    run at the start of a fold.
    """
    names = np.array(feature_names)
    n_cols = len(names)

    kept_cols: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for start in range(0, n_cols, PREPROCESS_BLOCK_COLS):
        cols = np.arange(start, min(start + PREPROCESS_BLOCK_COLS, n_cols))
        Xb = np.concatenate([part[:, cols] for part in train_parts])
        Xb, _, keep = _impute_and_scale(Xb, X_test[:, cols])
        kept_cols.append(cols[keep])
        if Xb.shape[1]:
            scores.append(f_classif(Xb, y_train)[0])
    kept = np.concatenate(kept_cols)

    if len(kept) > max_features:
        # Same selection rule as SelectKBest._get_support_mask.
        all_scores = np.concatenate(scores).astype(np.float64)
        all_scores[np.isnan(all_scores)] = np.finfo(np.float64).min
        top = np.zeros(len(kept), dtype=bool)
        top[np.argsort(all_scores, kind="mergesort")[-max_features:]] = True
        kept = kept[top]

    X_train = np.concatenate([part[:, kept] for part in train_parts])
    X_train, X_test, keep = _impute_and_scale(X_train, X_test[:, kept])
    assert keep.all()
    return X_train.astype(np.float32), X_test.astype(np.float32), names[kept].tolist()


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    metrics = {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "F1 Macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if np.unique(y_true).size > 1:
        try:
            metrics["ROC AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["ROC AUC"] = None
    else:
        metrics["ROC AUC"] = None
    return metrics


def aggregate(fold_results: List[Dict]) -> Dict:
    buckets: Dict[str, List[float]] = {}
    for res in fold_results:
        for k, v in res["metrics"].items():
            if v is not None:
                buckets.setdefault(k, []).append(v)
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in buckets.items()}


# ---------------------------------------------------------------------------
# Per-fold checkpointing (crash-safe: a LOSO run over 20+ patients x several
# foundation models can take hours, and has died silently mid-run more than
# once. Every (model, patient) result is written to disk as soon as it's
# computed — both the metrics (appended to the CSV immediately, not only at
# the very end) and the raw predictions (needed to rebuild the confusion
# matrices / ROC curves / report without re-running anything). A relaunch
# skips whatever's already on disk instead of redoing it.
# ---------------------------------------------------------------------------

def _fold_result_path(output_dir: Path, model_key: str, patient_id: str) -> Path:
    return output_dir / "fold_results" / f"{model_key}__{patient_id}.parquet"


def save_fold_result(
    output_dir: Path, model_key: str, model_name: str, patient_id: str,
    metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
    selected_features: Optional[List[str]] = None,
) -> None:
    path = _fold_result_path(output_dir, model_key, patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
    }).to_parquet(path, index=False)

    csv_path = output_dir / "loso_fold_metrics.csv"
    row = {"model": model_name, "patient": patient_id}
    row.update({k: v for k, v in metrics.items() if v is not None})
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def load_fold_result(output_dir: Path, model_key: str, patient_id: str) -> Optional[Dict]:
    path = _fold_result_path(output_dir, model_key, patient_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    y_true, y_pred, y_proba = df["y_true"].values, df["y_pred"].values, df["y_proba"].values
    return {
        "patient": patient_id,
        "metrics": evaluate_fold(y_true, y_pred, y_proba),
        "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion(fold_results: List[Dict], model_name: str, save_path: Path) -> np.ndarray:
    total_cm = np.zeros((2, 2), dtype=np.int64)
    for res in fold_results:
        total_cm += confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])
    tn, fp, fn, tp = total_cm.ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(total_cm, cmap="Blues")
    ax.set_title(f"{model_name} — LOSO\nTN={tn}  FP={fp}  FN={fn}  TP={tp}", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Seizure", "Seizure"])
    ax.set_yticklabels(["No Seizure", "Seizure"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(total_cm[r, c]), ha="center", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")
    return total_cm


def plot_roc(all_results: Dict[str, List[Dict]], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    for model_name, fold_results in all_results.items():
        y_true  = np.concatenate([r["y_true"]  for r in fold_results])
        y_proba = np.concatenate([r["y_proba"] for r in fold_results])
        if np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc(fpr, tpr):.3f})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — Next-window seizure prediction (LOSO)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


# ---------------------------------------------------------------------------
# Report (Markdown/txt — input for a later XAI study)
# ---------------------------------------------------------------------------

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report(
    args: argparse.Namespace,
    patients: List[str],
    patient_window_stats: List[Tuple[str, int, int]],
    all_results: Dict[str, List[Dict]],
    confusion_totals: Dict[str, np.ndarray],
    feature_selection_counts: Counter,
    n_folds_run: int,
    output_dir: Path,
) -> str:
    lines = []
    lines.append(f"# LOSO Foundation-Model Report — {args.dataset}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    cfg_rows = [
        ["Dataset", args.dataset],
        ["Data dir", str(args.data_dir)],
        ["Features cache", str(args.features_dir)],
        ["Models requested", ", ".join(MODEL_DISPLAY_NAMES[m] for m in args.models)],
        ["Window", f"{args.window_sec}s ({args.window_sec * args.fs} samples @ {args.fs}Hz), overlap {args.window_overlap:.0%}"],
        ["tsfresh feature set", args.tsfresh_params_desc],
        ["Seizure threshold", args.seizure_threshold],
        ["Max features (SelectKBest k)", args.max_features],
        ["Max in-context rows (TabFM only, all positives kept)",
         "no cap (full train set)" if args.max_incontext_rows <= 0 else args.max_incontext_rows],
        ["Mitra context", "full train set; halved by Mitra on OOM until it fits, "
                          "all positives kept (see log for the size used per fold)"],
        ["Mitra fine-tuning", "disabled (pure in-context learning)"],
        ["Min channel coverage", args.min_channel_coverage],
        ["Seed", args.seed],
    ]
    lines.append(_md_table(["Parameter", "Value"], cfg_rows))
    lines.append("")

    lines.append("## Per-patient window / seizure counts")
    lines.append("")
    rows = [
        [pid, n, n1, f"{100 * n1 / n:.1f}%" if n else "n/a"]
        for pid, n, n1 in patient_window_stats
    ]
    lines.append(_md_table(["Patient", "Windows", "Seizure windows (t+1)", "Seizure rate"], rows))
    lines.append("")

    lines.append("## Per-fold metrics")
    lines.append("")
    fold_rows = []
    for model_name, fold_results in all_results.items():
        for res in fold_results:
            m = res["metrics"]
            fold_rows.append([
                model_name, res["patient"],
                f"{m['Accuracy']:.4f}", f"{m['Precision']:.4f}", f"{m['Recall']:.4f}",
                f"{m['F1']:.4f}", f"{m['F1 Macro']:.4f}",
                "n/a" if m["ROC AUC"] is None else f"{m['ROC AUC']:.4f}",
            ])
    lines.append(_md_table(
        ["Model", "Patient", "Accuracy", "Precision", "Recall", "F1", "F1 Macro", "ROC AUC"],
        fold_rows,
    ))
    lines.append("")

    lines.append("## Aggregated metrics (mean ± std across folds)")
    lines.append("")
    agg_rows = []
    for model_name, fold_results in all_results.items():
        agg = aggregate(fold_results)
        agg_rows.append([
            model_name, len(fold_results),
            *(f"{agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}" if k in agg else "n/a"
              for k in ["Accuracy", "Precision", "Recall", "F1", "F1 Macro", "ROC AUC"]),
        ])
    lines.append(_md_table(
        ["Model", "Folds", "Accuracy", "Precision", "Recall", "F1", "F1 Macro", "ROC AUC"],
        agg_rows,
    ))
    lines.append("")

    lines.append("## Confusion totals (summed across folds)")
    lines.append("")
    cm_rows = []
    for model_name, cm in confusion_totals.items():
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append([model_name, tn, fp, fn, tp])
    lines.append(_md_table(["Model", "TN", "FP", "FN", "TP"], cm_rows))
    lines.append("")

    lines.append(f"## Most frequently selected features (top {args.top_features})")
    lines.append("")
    lines.append(
        f"Counted across {n_folds_run} LOSO folds — how many folds' `SelectKBest` "
        "kept each tsfresh feature. Meant as a starting point for feature-importance "
        "/ XAI analysis, not a substitute for it."
    )
    lines.append("")
    top = feature_selection_counts.most_common(args.top_features)
    feat_rows = [[i + 1, name, f"{count}/{n_folds_run}"] for i, (name, count) in enumerate(top)]
    lines.append(_md_table(["Rank", "Feature", "Selected in"], feat_rows))
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Per-fold metrics CSV: `{output_dir / 'loso_fold_metrics.csv'}`")
    for model_name in all_results:
        lines.append(
            f"- {model_name} confusion matrix: "
            f"`{output_dir / 'graphs' / f'confusion_{model_name.lower()}.png'}`"
        )
    lines.append(f"- ROC curves: `{output_dir / 'graphs' / 'roc_curves.png'}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _load_tabpfn_token()

    cfg = DATASET_CONFIGS[args.dataset]
    args.data_dir     = Path(args.data_dir or cfg["data_dir"])
    default_mode = "efficient" if cfg["fc_parameters"] is None else "light"
    fc_mode = default_mode if args.fc_parameters == "auto" else args.fc_parameters
    fc_suffix = "" if fc_mode == default_mode else f"_{fc_mode}"
    # window_overlap changes what each fold's train/test features look like,
    # so — like the tsfresh cache filename — it must be part of the results
    # directory too, or resuming/reading loso_fold_metrics.csv could silently
    # mix runs computed under different windowing.
    overlap_suffix = "" if args.window_overlap == 0 else f"_ov{args.window_overlap:g}"
    suffix = fc_suffix + overlap_suffix
    args.features_dir = Path(args.features_dir or (cfg["features_dir"] + fc_suffix))
    output_dir         = Path(args.output_dir or (cfg["output_dir"] + suffix))
    eeg_channels        = cfg["eeg_channels"]
    rename_cols          = cfg["rename_cols"]
    fc_parameters        = None if fc_mode == "efficient" else LIGHT_FC_PARAMETERS
    window_size = args.window_sec * args.fs

    models = [m for m in args.models if _model_available(m)]
    missing = [m for m in args.models if m not in models]
    if missing:
        print(f"[WARN] not installed, skipping: {', '.join(MODEL_DISPLAY_NAMES[m] for m in missing)}")
    if not models:
        raise RuntimeError("No requested models are installed (tabicl / tabpfn).")

    print("=" * 60)
    print("  LOSO — TABULAR FOUNDATION MODELS — EEG SEIZURE CLASSIFICATION")
    print("=" * 60)
    print(f"  Dataset        : {args.dataset}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Models         : {', '.join(MODEL_DISPLAY_NAMES[m] for m in models)}")
    print(f"  Window         : {args.window_sec}s ({window_size} samples @ {args.fs}Hz), overlap {args.window_overlap:.0%}")
    print(f"  Channels       : {len(eeg_channels)}")
    args.tsfresh_params_desc = (
        "EfficientFCParameters (~782 stats/channel)" if fc_parameters is None
        else f"light custom set ({len(fc_parameters)} families/channel)"
    )
    print(f"  tsfresh params : {args.tsfresh_params_desc}")

    output_dir.mkdir(parents=True, exist_ok=True)
    args.features_dir.mkdir(parents=True, exist_ok=True)

    patients = list_patients(args.data_dir)
    print(f"\nPatients ({len(patients)}): {patients}")

    print("\nExtracting / loading tsfresh features...")
    patient_features: Dict[str, pd.DataFrame] = {}
    patient_window_stats: List[Tuple[str, int, int]] = []
    for pid in patients:
        df_feat = extract_and_cache(
            pid, args.data_dir, args.features_dir, eeg_channels, rename_cols,
            window_size, args.seizure_threshold, args.min_channel_coverage,
            args.n_jobs, args.no_cache, fc_parameters, args.window_overlap,
        )
        if df_feat is not None:
            patient_features[pid] = df_feat
            n = len(df_feat)
            n1 = int(df_feat["label"].sum())
            patient_window_stats.append((pid, n, n1))
    print(f"\nFeatures ready for {len(patient_features)} patients")

    valid_patients = [p for p in patients if p in patient_features]
    all_results: Dict[str, List[Dict]] = {}
    confusion_totals: Dict[str, np.ndarray] = {}
    feature_selection_counts: Counter = Counter()
    n_folds_run = 0

    # Convert each patient's features to float32 arrays once, instead of
    # pd.concat-ing ~55k x 18.6k float64 DataFrames every fold: that peaked
    # at ~50GB RAM and got the run killed by systemd-oomd (efficient
    # fc-parameters on CHB-MIT). Column order matches what the per-fold
    # concat produced for fold 1 (union in order of appearance, first
    # patient last since it was the fold-1 test patient).
    feature_cols: List[str] = []
    seen = set()
    for p in valid_patients[1:] + valid_patients[:1]:
        for c in patient_features[p].columns:
            if c not in META_COLS and c not in seen:
                seen.add(c)
                feature_cols.append(c)
    patient_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for p in valid_patients:
        df = patient_features.pop(p)
        X = df.reindex(columns=feature_cols).to_numpy(dtype=np.float32)
        X[np.isinf(X)] = np.nan
        patient_arrays[p] = (X, df["label"].to_numpy(dtype=np.int64))
        del df
    gc.collect()

    print(f"\nLOSO: {len(valid_patients)} patients\n")

    for fold_idx, test_pid in enumerate(valid_patients):
        train_pids = [p for p in valid_patients if p != test_pid]
        train_parts = [patient_arrays[p][0] for p in train_pids]
        y_train = np.concatenate([patient_arrays[p][1] for p in train_pids])
        X_test, y_test = patient_arrays[test_pid]

        print(f"Fold {fold_idx + 1}/{len(valid_patients)}  test={test_pid}")
        print(f"  train={(len(y_train), X_test.shape[1])}  pos={y_train.sum()}/{len(y_train)}")
        print(f"  test ={X_test.shape}   pos={y_test.sum()}/{len(y_test)}")

        if y_train.sum() == 0 or y_test.sum() == 0:
            print("  [SKIP] positive class absent in train or test\n")
            continue

        X_tr, X_te, selected_names = preprocess_fold(
            train_parts, y_train, X_test, feature_cols, args.max_features,
        )
        feature_selection_counts.update(selected_names)
        n_folds_run += 1
        print(f"  features after preprocessing: {X_tr.shape[1]}")

        for model_key in models:
            model_name = MODEL_DISPLAY_NAMES[model_key]

            if not args.no_resume_loso:
                cached = load_fold_result(output_dir, model_key, test_pid)
                if cached is not None:
                    all_results.setdefault(model_name, []).append(cached)
                    roc = cached["metrics"]["ROC AUC"]
                    print(
                        f"  → {model_name}: resumed from disk  "
                        f"F1={cached['metrics']['F1']:.4f}  "
                        f"ROC={'N/A' if roc is None else f'{roc:.4f}'}"
                    )
                    continue

            row_cap = _context_row_cap(model_key, args.max_incontext_rows)
            if 0 < row_cap < len(X_tr):
                X_fit, y_fit = subsample_train(X_tr, y_train, row_cap, args.seed)
                print(
                    f"  → {model_name}: subsampling train {len(X_tr)} -> {len(X_fit)} rows "
                    f"(GPU context limit)  pos={int((y_fit == 1).sum())}/{len(y_fit)}"
                )
            else:
                X_fit, y_fit = X_tr, y_train

            # Two complete, independently-timestamped lines instead of a
            # print(..., end=" ") continuation — lets the log show exactly
            # how long each model's fit()/predict() took, and plays nicely
            # with the timestamped `print` wrapper above (no mid-line stamps).
            print(f"  → {model_name}: fitting...")
            try:
                model = _build_model(model_key, args.seed, len(X_fit))
                model.fit(X_fit, y_fit)
                if model_key == "mitra":
                    n_ctx = min(model.trainers[0].cfg.hyperparams["max_samples_support"], len(X_fit))
                    print(
                        f"  → {model_name}: context {n_ctx}/{len(X_fit)} rows  "
                        f"pos={min(int((y_fit == 1).sum()), n_ctx)}/{n_ctx}"
                    )
                y_pred  = model.predict(X_te)
                y_proba = model.predict_proba(X_te)[:, 1]
                metrics = evaluate_fold(y_test, y_pred, y_proba)

                # Persisted immediately — a crash on the *next* fold/model no
                # longer loses this one (this run has died mid-LOSO 3 times).
                save_fold_result(
                    output_dir, model_key, model_name, test_pid,
                    metrics, y_test, y_pred, y_proba,
                )

                all_results.setdefault(model_name, []).append({
                    "patient": test_pid, "metrics": metrics,
                    "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,
                })
                roc = metrics["ROC AUC"]
                print(
                    f"  → {model_name}: F1={metrics['F1']:.4f}  "
                    f"F1-macro={metrics['F1 Macro']:.4f}  "
                    f"ROC={'N/A' if roc is None else f'{roc:.4f}'}"
                )
            except Exception as e:
                print(f"  → {model_name}: [ERROR] {e}")
            finally:
                model = None
                _release_gpu()
        print()
        # Release this fold's arrays before the next fold's concatenate, so
        # two folds' train sets are never alive at once.
        train_parts = X_tr = X_te = X_fit = model = None
        gc.collect()

    print("=== LOSO SUMMARY ===\n")
    for model_name, fold_results in all_results.items():
        agg = aggregate(fold_results)
        print(f"{model_name}  ({len(fold_results)} folds)")
        for metric, stats in agg.items():
            print(f"  {metric:<12}  {stats['mean']:.4f} ± {stats['std']:.4f}")
        print()

    rows = []
    for model_name, fold_results in all_results.items():
        for res in fold_results:
            row = {"model": model_name, "patient": res["patient"]}
            row.update({k: v for k, v in res["metrics"].items() if v is not None})
            rows.append(row)
    csv_path = output_dir / "loso_fold_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Per-fold metrics: {csv_path}\n")

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    for model_name, fold_results in all_results.items():
        confusion_totals[model_name] = plot_confusion(
            fold_results, model_name, graphs_dir / f"confusion_{model_name.lower()}.png",
        )
    plot_roc(all_results, graphs_dir / "roc_curves.png")

    report = build_report(
        args, patients, patient_window_stats, all_results,
        confusion_totals, feature_selection_counts, n_folds_run, output_dir,
    )
    report_path = output_dir / f"loso_report.{args.report_format}"
    report_path.write_text(report)
    print(f"\nReport written to: {report_path}")
    print("LOSO complete.")


if __name__ == "__main__":
    main()
