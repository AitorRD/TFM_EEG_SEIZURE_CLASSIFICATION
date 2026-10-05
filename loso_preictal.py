"""
Experiment 1 — LOSO seizure *prediction* ("time to attack") with the tabular
foundation models (TabICL, TabPFNv3, TabFM, Mitra; TabPFNv2 optional).

loso_foundation_report.py labels a window positive when the *next* window is
>= 50% seizure, which mixes prediction with early detection (most positives
are windows right at/inside the seizure). Here the question is strictly
prospective:

    Given the 30 s EEG window ending at t, does a seizure *start* in
    (t + SPH, t + SPH + horizon]?           default SPH = 0, horizon = 60 s

Window classes (per window, using the real recording time in "Time (s)"):
  - preictal  (1): onset within (SPH, SPH + horizon] after the window end
  - interictal(0): next onset > interictal-gap (default 5 min) away, or none
  - excluded     : window overlaps a seizure (ictal), starts < postictal-sec
                   after a seizure end, spans a clipping discontinuity, or
                   falls in the ambiguous zone between horizon and
                   interictal-gap. Excluded test windows are still predicted
                   (except ictal/postictal/gap ones) so the time-to-onset
                   curve covers the whole run-up to each seizure.

Features: the tsfresh cache written by loso_foundation_report.py is reused
as-is (same 30 s / 25%-overlap windows), only relabeled — no re-extraction.
To get more than ~2-3 preictal windows per seizure, extra preictal windows
are extracted at a denser step (--aug-step-sec) and added to the *training*
patients only; the test patient keeps the original uniform grid.

Protocol per LOSO fold (test patient P):
  - V = --n-val-patients other patients (seeded), T = the rest.
  - Preprocessing (impute / variance / scale / SelectKBest) fit on T only.
  - In-context set: T subsampled to --max-context-rows (all positives kept),
    the same rows for every model.
  - Decision threshold = the one maximising F1 on V; then applied to P.
  - Metrics on P: ROC-AUC, PR-AUC, sensitivity/specificity/F1 at that
    threshold, seizure-level sensitivity, mean warning time and false-alarm
    windows per hour.

Usage:
    python loso_preictal.py
    python loso_preictal.py --models tabpfnv3 mitra --horizon-sec 60 --sph-sec 0
    python loso_preictal.py --aug-step-sec 0          # no preictal augmentation
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute

import loso_foundation_report as L
from loso_foundation_report import print  # noqa: A004 - timestamped, flushed print

DEFAULT_MODELS = ["tabicl", "tabpfnv3", "tabfm", "mitra"]


# ---------------------------------------------------------------------------
# Mitra context sampling
# ---------------------------------------------------------------------------

def _mitra_getitem_stratified(self, idx):
    """Replacement for Mitra's DatasetFinetune.__getitem__ (overrides the
    keep-every-minority-row patch from loso_foundation_report.py).

    Here the context handed to every model is already rebalanced (all
    positives kept, negatives subsampled), so with the dense preictal windows
    positives can outnumber the ~3.5k support rows Mitra fits on the GPU.
    Keeping all of them would give Mitra an almost all-positive context;
    sampling stratified keeps the same class ratio the other models see.
    """
    n, size = self.n_samples_support, self.support_size
    if size < n and self.cfg.task == L._MitraTask.CLASSIFICATION:
        y = np.asarray(self.y_support)
        values, counts = np.unique(y, return_counts=True)
        minority = values[counts.argmin()]
        pos_idx = np.flatnonzero(y == minority)
        neg_idx = np.flatnonzero(y != minority)
        n_pos = min(len(pos_idx), max(1, int(round(size * len(pos_idx) / n))))
        support_indices = self.rng.permutation(np.concatenate([
            self.rng.choice(pos_idx, size=n_pos, replace=False),
            self.rng.choice(neg_idx, size=size - n_pos, replace=False),
        ]))
    else:
        support_indices = self.rng.choice(n, size=size, replace=False)

    return {
        "x_support": L.torch.as_tensor(self.x_support[support_indices]),
        "y_support": L.torch.as_tensor(self.y_support[support_indices]),
        "x_query": L.torch.as_tensor(self.x_queries[idx]),
        "y_query": L.torch.as_tensor(self.y_queries[idx]),
    }


if L.HAS_MITRA:
    L._MitraDatasetFinetune.__getitem__ = _mitra_getitem_stratified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser(description: str = "LOSO seizure prediction (preictal vs interictal) "
                                    "with tabular foundation models") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=list(L.DATASET_CONFIGS), default="chbmit")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--features-dir", type=str, default=None,
                   help="tsfresh cache written by loso_foundation_report.py (reused, only relabeled)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--fc-parameters", choices=["auto", "efficient", "light"], default="efficient",
                   help="Must match the cache being reused (efficient = the TabFM/Mitra/TabPFN runs)")
    p.add_argument("--window-sec", type=int, default=30)
    p.add_argument("--window-overlap", type=float, default=0.25,
                   help="Overlap of the cached windows (selects the *_ov<x>.parquet cache)")
    p.add_argument("--fs", type=int, default=100)
    p.add_argument("--seizure-threshold", type=float, default=0.5,
                   help="Only used to check the cache rows line up with the replayed windows")
    p.add_argument("--min-channel-coverage", type=float, default=0.5)

    g = p.add_argument_group("prediction target")
    g.add_argument("--horizon-sec", type=float, default=60.0,
                   help="Positive if a seizure starts within this many seconds after SPH")
    g.add_argument("--sph-sec", type=float, default=0.0,
                   help="Seizure prediction horizon: gap between window end and the horizon")
    g.add_argument("--interictal-gap-sec", type=float, default=300.0,
                   help="Negative only if the next onset is further than this; windows between "
                        "SPH+horizon and this are excluded as ambiguous")
    g.add_argument("--postictal-sec", type=float, default=300.0,
                   help="Exclude windows starting less than this after a seizure ends")
    g.add_argument("--aug-step-sec", type=float, default=7.5,
                   help="Step of the extra preictal windows added to the training patients "
                        "(0 = no augmentation)")

    g = p.add_argument_group("models / protocol")
    g.add_argument("--models", nargs="+", choices=list(L.MODEL_DISPLAY_NAMES), default=DEFAULT_MODELS)
    g.add_argument("--max-features", type=int, default=200)
    g.add_argument("--max-context-rows", type=int, default=10000,
                   help="Rows in the in-context set (all positives kept, negatives subsampled); "
                        "the same set for every model. 0 = whole training split")
    g.add_argument("--n-val-patients", type=int, default=3,
                   help="Training patients held out per fold to pick the decision threshold")
    g.add_argument("--tta-max-sec", type=float, default=1800.0,
                   help="x-range of the time-to-onset plot")
    g.add_argument("--tta-bin-sec", type=float, default=30.0)

    p.add_argument("--n-jobs", type=int, default=8, help="tsfresh jobs (augmentation only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true",
                   help="Rebuild window metadata and augmentation features")
    p.add_argument("--no-resume-loso", action="store_true")
    return p


def resolve_paths(args: argparse.Namespace, out_prefix: str = "loso_preictal") -> None:
    cfg = L.DATASET_CONFIGS[args.dataset]
    default_mode = "efficient" if cfg["fc_parameters"] is None else "light"
    args.fc_mode = default_mode if args.fc_parameters == "auto" else args.fc_parameters
    fc_suffix = "" if args.fc_mode == default_mode else f"_{args.fc_mode}"
    args.data_dir = Path(args.data_dir or cfg["data_dir"])
    args.features_dir = Path(args.features_dir or (cfg["features_dir"] + fc_suffix))
    args.fc_dict = None if args.fc_mode == "efficient" else L.LIGHT_FC_PARAMETERS
    args.eeg_channels = cfg["eeg_channels"]
    args.rename_cols = cfg["rename_cols"]
    args.window_size = args.window_sec * args.fs
    args.step_sec = max(1, int(args.window_size * (1 - args.window_overlap))) / args.fs
    args.ov_suffix = "" if args.window_overlap == 0 else f"_ov{args.window_overlap:g}"
    target = f"h{args.horizon_sec:g}_sph{args.sph_sec:g}"
    args.output_dir = Path(args.output_dir or
                           f"images/results/{out_prefix}_{args.dataset}_{args.fc_mode}_{target}")


# ---------------------------------------------------------------------------
# Window metadata (replays loso_foundation_report.py's windowing)
# ---------------------------------------------------------------------------

def _read_session(csv_file: Path, args, full: bool = False) -> Optional[pd.DataFrame]:
    """Same skip rules as L.create_windows_for_patient, so the replayed window
    index i lines up with the cached window_id f"{pid}_{i}"."""
    try:
        header = pd.read_csv(csv_file, nrows=0).rename(columns=args.rename_cols).columns
        df = pd.read_csv(csv_file, engine="pyarrow",
                         usecols=None if full else ["Time (s)", "Seizure"])
    except Exception as e:
        print(f"  [WARN] {csv_file.name}: {e}")
        return None
    if df.empty:
        return None
    present = sum(1 for ch in args.eeg_channels if ch in header)
    if present / len(args.eeg_channels) < args.min_channel_coverage:
        return None
    if full:
        df = df.rename(columns=args.rename_cols)
        for ch in args.eeg_channels:
            if ch not in df.columns:
                df[ch] = 0.0
        df = df.fillna(0)
    return df


def window_properties(time: np.ndarray, seizure: np.ndarray, starts: np.ndarray,
                      window_size: int, fs: int) -> Dict[str, np.ndarray]:
    """Per-window timing relative to the session's seizures. Times come from
    "Time (s)", which keeps the original EDF time across the gaps left by
    clipping, so time-to-onset stays correct even across a discontinuity."""
    seizure = seizure.astype(bool)
    ends = starts + window_size
    t_start, t_end = time[starts], time[ends - 1]
    cs = np.concatenate([[0], np.cumsum(seizure, dtype=np.int64)])
    edges = np.diff(seizure.astype(np.int8), prepend=0, append=0)
    onsets = time[np.flatnonzero(edges == 1)]
    offsets = time[np.flatnonzero(edges == -1) - 1]

    k = np.searchsorted(onsets, t_end, side="right")
    tto = np.full(len(starts), np.inf)
    has_next = k < len(onsets)
    tto[has_next] = onsets[k[has_next]] - t_end[has_next]

    j = np.searchsorted(offsets, t_start, side="left") - 1
    tsince = np.full(len(starts), np.inf)
    has_prev = j >= 0
    tsince[has_prev] = t_start[has_prev] - offsets[j[has_prev]]

    return {
        "t_start": t_start, "t_end": t_end,
        "gap": (t_end - t_start) > (window_size - 0.5) / fs,
        "ictal": (cs[ends] - cs[starts]) > 0,
        "tto": tto, "tsince": tsince,
    }


def assign_labels(meta: pd.DataFrame, args) -> np.ndarray:
    """-2 = unusable (ictal / postictal / gap), -1 = ambiguous, 0 = interictal, 1 = preictal."""
    valid = ~meta["ictal"].values & ~meta["gap"].values & (meta["tsince"].values > args.postictal_sec)
    tto = meta["tto"].values
    y = np.full(len(meta), -2, dtype=np.int8)
    y[valid] = -1
    y[valid & (tto > args.sph_sec) & (tto <= args.sph_sec + args.horizon_sec)] = 1
    y[valid & (tto > max(args.interictal_gap_sec, args.sph_sec + args.horizon_sec))] = 0
    return y


def load_window_meta(pid: str, args) -> pd.DataFrame:
    path = args.features_dir / "preictal_meta" / f"{pid}{args.ov_suffix}.parquet"
    if not args.no_cache and path.exists():
        return pd.read_parquet(path)

    W = args.window_size
    step = max(1, int(W * (1 - args.window_overlap)))
    parts, i = [], 0
    for csv_file in sorted((args.data_dir / pid).glob("*_clipped.csv")):
        df = _read_session(csv_file, args)
        if df is None:
            continue
        time = df["Time (s)"].to_numpy(np.float64)
        seizure = df["Seizure"].fillna(0).to_numpy(np.int64)
        starts = np.arange(0, len(time) - 2 * W + 1, step)
        if len(starts) == 0:
            continue
        props = window_properties(time, seizure, starts, W, args.fs)
        cs = np.concatenate([[0], np.cumsum(seizure)])
        old_label = ((cs[starts + 2 * W] - cs[starts + W]) / W >= args.seizure_threshold).astype(np.int64)
        parts.append(pd.DataFrame({
            "window_id": [f"{pid}_{k}" for k in range(i, i + len(starts))],
            "session": csv_file.stem, "start_row": starts, "old_label": old_label, **props,
        }))
        i += len(starts)

    meta = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(path, index=False)
    return meta


# ---------------------------------------------------------------------------
# Dense preictal windows for the training patients
# ---------------------------------------------------------------------------

def load_or_extract_aug(pid: str, args, feature_cols: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if args.aug_step_sec <= 0:
        return None
    name = (f"{pid}_w{args.window_sec}_h{args.horizon_sec:g}_sph{args.sph_sec:g}"
            f"_pi{args.postictal_sec:g}_s{args.aug_step_sec:g}.parquet")
    path = args.features_dir / "preictal_aug" / name

    if args.no_cache or not path.exists():
        W = args.window_size
        step = max(1, int(round(args.aug_step_sec * args.fs)))
        windows, ttos = [], []
        for csv_file in sorted((args.data_dir / pid).glob("*_clipped.csv")):
            light = _read_session(csv_file, args)
            if light is None or not light["Seizure"].any():
                continue
            df = _read_session(csv_file, args, full=True)
            time = df["Time (s)"].to_numpy(np.float64)
            seizure = df["Seizure"].to_numpy(np.int64)
            starts = np.arange(0, len(time) - W + 1, step)
            meta = pd.DataFrame(window_properties(time, seizure, starts, W, args.fs))
            keep = assign_labels(meta, args) == 1
            if not keep.any():
                continue
            eeg = df[args.eeg_channels].to_numpy(np.float32)
            windows.extend(eeg[s:s + W] for s in starts[keep])
            ttos.extend(meta["tto"].values[keep])

        if windows:
            print(f"  {pid}: extracting {len(windows)} dense preictal windows...")
            wids = [f"{pid}_aug{k}" for k in range(len(windows))]
            feats = extract_features(
                L.windows_to_wide_df(np.stack(windows), wids, args.eeg_channels),
                column_id="id", column_sort="time",
                default_fc_parameters=args.fc_dict if args.fc_dict is not None else EfficientFCParameters(),
                n_jobs=args.n_jobs, disable_progressbar=True, impute_function=tsfresh_impute,
            )
            feats.index.name = "window_id"
            feats = feats.reset_index()
            feats["tto"] = np.asarray(ttos)
        else:
            feats = pd.DataFrame({"window_id": pd.Series(dtype=str), "tto": pd.Series(dtype=float)})
        path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path, index=False)

    feats = pd.read_parquet(path)
    if feats.empty:
        return None
    X = feats.reindex(columns=feature_cols).to_numpy(dtype=np.float32)
    X[np.isinf(X)] = np.nan
    return X, np.ones(len(X), dtype=np.int64)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class PatientData:
    X: np.ndarray            # usable windows only (label != -2)
    y: np.ndarray            # -1 ambiguous, 0 interictal, 1 preictal
    tto: np.ndarray          # seconds from window end to next onset (inf = none)
    event: np.ndarray        # "<session>@<onset>" for windows with an upcoming onset, else ""
    window_id: np.ndarray
    aug: Optional[Tuple[np.ndarray, np.ndarray]] = None


def load_dataset(args) -> Tuple[List[str], List[str], Dict[str, PatientData], List[Tuple[str, int, int, int]]]:
    patients = L.list_patients(args.data_dir)
    print(f"\nPatients ({len(patients)}): {patients}")
    print("\nLoading tsfresh features (cache from loso_foundation_report.py)...")
    feats: Dict[str, pd.DataFrame] = {}
    for pid in patients:
        df = L.extract_and_cache(
            pid, args.data_dir, args.features_dir, args.eeg_channels, args.rename_cols,
            args.window_size, args.seizure_threshold, args.min_channel_coverage,
            args.n_jobs, False, args.fc_dict, args.window_overlap,
        )
        if df is not None:
            feats[pid] = df
    patients = [p for p in patients if p in feats]

    # Same column union / order as loso_foundation_report.main.
    feature_cols: List[str] = []
    seen = set()
    for p in patients[1:] + patients[:1]:
        for c in feats[p].columns:
            if c not in L.META_COLS and c not in seen:
                seen.add(c)
                feature_cols.append(c)

    print("\nRelabeling windows (preictal / interictal) and loading augmentation...")
    data: Dict[str, PatientData] = {}
    stats = []
    for pid in patients:
        df = feats.pop(pid)
        meta = load_window_meta(pid, args).set_index("window_id")
        if len(meta) != len(df):
            raise RuntimeError(
                f"{pid}: {len(df)} cached windows but {len(meta)} replayed — the cache was built "
                f"with different windowing/data; rerun with matching --window-sec/--window-overlap"
            )
        meta = meta.loc[df["window_id"]]
        if not np.array_equal(meta["old_label"].values, df["label"].values):
            raise RuntimeError(f"{pid}: replayed windows don't match the cached labels")

        y = assign_labels(meta, args)
        usable = y != -2
        X = df.loc[usable].reindex(columns=feature_cols).to_numpy(dtype=np.float32)
        X[np.isinf(X)] = np.nan
        m = meta[usable]
        onset = (m["t_end"] + m["tto"]).values
        event = np.where(np.isfinite(onset),
                         m["session"].values + "@" + np.round(onset, 1).astype(str), "")
        data[pid] = PatientData(X=X, y=y[usable], tto=m["tto"].values, event=event,
                                window_id=m.index.values,
                                aug=load_or_extract_aug(pid, args, feature_cols))
        n_aug = 0 if data[pid].aug is None else len(data[pid].aug[1])
        stats.append((pid, int((y == 1).sum()), int((y == 0).sum()), n_aug))
        print(f"  {pid}: preictal={stats[-1][1]}  interictal={stats[-1][2]}  "
              f"ambiguous={int((y == -1).sum())}  unusable={int((~usable).sum())}  aug={n_aug}")
        del df
    gc.collect()
    return patients, feature_cols, data, stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.nanargmax(f1)])


def evaluate(y: np.ndarray, p: np.ndarray, thr: float, tto: np.ndarray, event: np.ndarray,
             args) -> Dict[str, Optional[float]]:
    lab = y >= 0
    yl, pl = y[lab], p[lab]
    pred = (pl >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yl, pred, labels=[0, 1]).ravel()
    two = np.unique(yl).size > 1
    m: Dict[str, Optional[float]] = {
        "Accuracy": float(accuracy_score(yl, pred)),
        "Precision": float(precision_score(yl, pred, zero_division=0)),
        "Recall": float(recall_score(yl, pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else None,
        "F1": float(f1_score(yl, pred, zero_division=0)),
        "F1 Macro": float(f1_score(yl, pred, average="macro", zero_division=0)),
        "ROC AUC": float(roc_auc_score(yl, pl)) if two else None,
        "PR AUC": float(average_precision_score(yl, pl)) if two else None,
        "Threshold": float(thr),
    }

    # Seizure-level: a seizure counts as predicted if any of its preictal
    # windows raises an alarm; warning time = earliest alarm before onset
    # within the interictal gap (alarms 1-5 min early are useful warnings too).
    alarm = p >= thr
    detected, warnings = [], []
    for ev in np.unique(event[y == 1]):
        in_ev = event == ev
        detected.append(bool(alarm[in_ev & (y == 1)].any()))
        early = in_ev & alarm & (tto > args.sph_sec) & (tto <= max(args.interictal_gap_sec,
                                                                   args.sph_sec + args.horizon_sec))
        if detected[-1] and early.any():
            warnings.append(float(tto[early].max()))
    m["Seizure Sensitivity"] = float(np.mean(detected)) if detected else None
    m["Warning Time (s)"] = float(np.mean(warnings)) if warnings else None
    inter_hours = (y == 0).sum() * args.step_sec / 3600
    m["FA/h"] = float((alarm & (y == 0)).sum() / inter_hours) if inter_hours else None
    m["TP"], m["FP"], m["FN"], m["TN"] = int(tp), int(fp), int(fn), int(tn)
    return m


SUMMARY_METRICS = ["ROC AUC", "PR AUC", "Recall", "Specificity", "Precision", "F1", "F1 Macro",
                   "Seizure Sensitivity", "Warning Time (s)", "FA/h"]


def aggregate(fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in SUMMARY_METRICS:
        vals = [r["metrics"][k] for r in fold_results if r["metrics"].get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


# ---------------------------------------------------------------------------
# Per-fold checkpointing
# ---------------------------------------------------------------------------

def _fold_path(output_dir: Path, key: str, pid: str) -> Path:
    return output_dir / "fold_results" / f"{key}__{pid}.parquet"


def save_fold(output_dir: Path, key: str, name: str, pid: str, d: PatientData,
              proba: np.ndarray, thr: float, metrics: Dict) -> None:
    path = _fold_path(output_dir, key, pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"window_id": d.window_id, "y": d.y, "tto": d.tto, "event": d.event,
                  "y_proba": proba, "threshold": thr}).to_parquet(path, index=False)
    csv_path = output_dir / "loso_fold_metrics.csv"
    row = {"model": name, "patient": pid, **{k: v for k, v in metrics.items() if v is not None}}
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def load_fold(output_dir: Path, key: str, pid: str, args) -> Optional[Dict]:
    path = _fold_path(output_dir, key, pid)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    y, p, tto, ev = df["y"].values, df["y_proba"].values, df["tto"].values, df["event"].values
    thr = float(df["threshold"].iloc[0])
    return {"patient": pid, "y": y, "proba": p, "tto": tto, "threshold": thr,
            "metrics": evaluate(y, p, thr, tto, ev, args)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion(fold_results: List[Dict], name: str, path: Path) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for r in fold_results:
        lab = r["y"] >= 0
        cm += confusion_matrix(r["y"][lab], (r["proba"][lab] >= r["threshold"]).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} — LOSO\nTN={tn}  FP={fp}  FN={fn}  TP={tp}", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Interictal", "Preictal"]); ax.set_yticklabels(["Interictal", "Preictal"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {path}")
    return cm


def plot_roc_pr(all_results: Dict[str, List[Dict]], path: Path, title: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    a1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
    prevalence = None
    for name, folds in all_results.items():
        y = np.concatenate([r["y"] for r in folds]); p = np.concatenate([r["proba"] for r in folds])
        lab = y >= 0
        y, p = y[lab], p[lab]
        if np.unique(y).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(y, p):.3f})")
        prevalence = y.mean()
    if prevalence is not None:
        a2.axhline(prevalence, color="k", ls="--", lw=1.2, label=f"Random ({prevalence:.3f})")
    a1.set_xlabel("FPR"); a1.set_ylabel("TPR"); a1.set_title(f"ROC — {title}")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title(f"Precision-Recall — {title}")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {path}")


def plot_time_to_onset(all_results: Dict[str, List[Dict]], path: Path, args) -> None:
    """Alarm rate and mean predicted probability vs. time left to the seizure
    onset, pooled over all test folds — the "time to attack" view. Dotted
    lines: the same quantities on windows with no upcoming seizure."""
    edges = np.arange(0, args.tta_max_sec + args.tta_bin_sec, args.tta_bin_sec)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, folds in all_results.items():
        tto = np.concatenate([r["tto"] for r in folds])
        p = np.concatenate([r["proba"] for r in folds])
        alarm = np.concatenate([r["proba"] >= r["threshold"] for r in folds])
        b = np.digitize(tto, edges) - 1
        inside = np.isfinite(tto) & (b >= 0) & (b < len(centers))
        rate = [alarm[inside & (b == i)].mean() if (inside & (b == i)).any() else np.nan
                for i in range(len(centers))]
        mean_p = [p[inside & (b == i)].mean() if (inside & (b == i)).any() else np.nan
                  for i in range(len(centers))]
        line, = a1.plot(-centers, rate, marker="o", ms=3, lw=1.5, label=name)
        a2.plot(-centers, mean_p, marker="o", ms=3, lw=1.5, color=line.get_color(), label=name)
        none = ~np.isfinite(tto)
        if none.any():
            a1.axhline(alarm[none].mean(), color=line.get_color(), ls=":", lw=1)
            a2.axhline(p[none].mean(), color=line.get_color(), ls=":", lw=1)
    for a in (a1, a2):
        a.axvspan(-(args.sph_sec + args.horizon_sec), -args.sph_sec, color="tab:red", alpha=0.12,
                  label="prediction horizon")
        a.axvline(-args.interictal_gap_sec, color="grey", ls="--", lw=1)
        a.grid(alpha=0.3)
    a1.set_ylabel("Alarm rate (p ≥ fold threshold)")
    a2.set_ylabel("Mean predicted probability")
    a2.set_xlabel("Time to seizure onset (s)  —  window end relative to onset")
    a1.set_title("Time to attack — LOSO test windows (dotted: no upcoming seizure)")
    a1.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(stats: Dict, k: str) -> str:
    return f"{stats[k]['mean']:.4f} ± {stats[k]['std']:.4f}" if k in stats else "n/a"


def build_report(args, title: str, stats_rows, all_results, cms, extra_config, extra_sections) -> str:
    lines = [f"# {title} — {args.dataset}", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
    cfg_rows = [
        ["Target", f"seizure onset within ({args.sph_sec:g}, {args.sph_sec + args.horizon_sec:g}] s "
                   f"after the end of a {args.window_sec}s window"],
        ["Interictal", f"next onset > {args.interictal_gap_sec:g}s away (or none); windows in between excluded"],
        ["Postictal exclusion", f"{args.postictal_sec:g}s after seizure end"],
        ["Windows", f"{args.window_sec}s, overlap {args.window_overlap:.0%} (step {args.step_sec:g}s), "
                    f"cache {args.features_dir}"],
        ["Train augmentation", "off" if args.aug_step_sec <= 0 else
         f"dense preictal windows every {args.aug_step_sec:g}s (training patients only)"],
        ["tsfresh feature set", args.fc_mode],
        ["SelectKBest k", args.max_features],
        ["In-context rows", "whole training split" if args.max_context_rows <= 0 else
         f"{args.max_context_rows} (all positives kept)"],
        ["Threshold", f"max F1 on {args.n_val_patients} held-out training patients per fold"],
        ["Seed", args.seed],
        *extra_config,
    ]
    lines += ["## Configuration", "", L._md_table(["Parameter", "Value"], cfg_rows), ""]
    lines += ["## Per-patient windows", "",
              L._md_table(["Patient", "Preictal", "Interictal", "Aug. preictal (train only)"],
                          [list(r) for r in stats_rows]), ""]

    rows = []
    for name, folds in all_results.items():
        agg = aggregate(folds)
        rows.append([name, len(folds), *(_fmt(agg, k) for k in SUMMARY_METRICS)])
    lines += ["## Aggregated metrics (mean ± std across folds)", "",
              L._md_table(["Model", "Folds", *SUMMARY_METRICS], rows), ""]

    lines += ["## Per-fold metrics", ""]
    fold_rows = []
    for name, folds in all_results.items():
        for r in folds:
            m = r["metrics"]
            fold_rows.append([name, r["patient"], *(
                "n/a" if m.get(k) is None else f"{m[k]:.4f}" for k in SUMMARY_METRICS + ["Threshold"])])
    lines += [L._md_table(["Model", "Patient", *SUMMARY_METRICS, "Threshold"], fold_rows), ""]

    lines += ["## Confusion totals (per-fold thresholds, summed)", "",
              L._md_table(["Model", "TN", "FP", "FN", "TP"],
                          [[n, *cm.ravel()] for n, cm in cms.items()]), ""]
    lines += extra_sections
    lines += ["## Notes", "",
              "- CHB-MIT seizure recordings are clipped to ±30 min around each seizure, so the "
              "usual >= 4 h interictal separation is impossible; 'interictal' here means more than "
              f"{args.interictal_gap_sec:g}s before the next onset or a seizure-free recording.",
              "- FA/h counts alarm *windows* on interictal data per hour of interictal recording "
              "(no refractory period).",
              "- `graphs/time_to_onset.png` shows how the alarm rate / probability evolves as the "
              "onset approaches.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LOSO runner (shared with loso_preictal_finetune.py)
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str                                  # checkpoint / file key
    name: str                                 # display name
    build: Callable[[int], Any]               # n_context_rows -> unfitted estimator
    uses_val: bool = False                    # fit(X, y, X_val=..., y_val=...) for early stopping


def foundation_specs(models: List[str], seed: int) -> List[ModelSpec]:
    return [ModelSpec(k, L.MODEL_DISPLAY_NAMES[k], partial(L._build_model, k, seed)) for k in models]


def run_loso(args, specs: List[ModelSpec], title: str,
             extra_config: Optional[List[List[str]]] = None,
             extra_sections_fn: Optional[Callable[[Dict[str, List[Dict]]], List[str]]] = None) -> Dict[str, List[Dict]]:
    L._load_tabpfn_token()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  {title.upper()}")
    print("=" * 60)
    print(f"  Dataset   : {args.dataset}  ({args.data_dir})")
    print(f"  Models    : {', '.join(s.name for s in specs)}")
    print(f"  Target    : onset within ({args.sph_sec:g}, {args.sph_sec + args.horizon_sec:g}] s "
          f"after a {args.window_sec}s window")
    print(f"  Output    : {args.output_dir}")

    patients, feature_cols, data, stats_rows = load_dataset(args)
    all_results: Dict[str, List[Dict]] = {}

    print(f"\nLOSO: {len(patients)} patients\n")
    for fold_idx, test_pid in enumerate(patients):
        test = data[test_pid]
        print(f"Fold {fold_idx + 1}/{len(patients)}  test={test_pid}  "
              f"preictal={int((test.y == 1).sum())}  interictal={int((test.y == 0).sum())}")
        if not (test.y == 1).any() or not (test.y == 0).any():
            print("  [SKIP] test patient lacks preictal or interictal windows\n")
            continue

        todo = []
        for s in specs:
            cached = None if args.no_resume_loso else load_fold(args.output_dir, s.key, test_pid, args)
            if cached is not None:
                all_results.setdefault(s.name, []).append(cached)
                print(f"  → {s.name}: resumed from disk  ROC={cached['metrics']['ROC AUC']}  "
                      f"PR={cached['metrics']['PR AUC']}")
            else:
                todo.append(s)
        if not todo:
            print()
            continue

        others = [p for p in patients if p != test_pid]
        rng = np.random.RandomState(args.seed + fold_idx)
        cand = [p for p in others if (data[p].y == 1).any()]
        val_pids = sorted(str(p) for p in
                          rng.choice(cand, size=min(args.n_val_patients, len(cand) - 1), replace=False))
        train_pids = [p for p in others if p not in val_pids]

        train_parts, y_parts = [], []
        for p in train_pids:
            lab = data[p].y >= 0
            train_parts.append(data[p].X[lab]); y_parts.append(data[p].y[lab].astype(np.int64))
            if data[p].aug is not None:
                train_parts.append(data[p].aug[0]); y_parts.append(data[p].aug[1])
        y_train = np.concatenate(y_parts)
        val_X = np.concatenate([data[p].X[data[p].y >= 0] for p in val_pids])
        val_y = np.concatenate([data[p].y[data[p].y >= 0] for p in val_pids]).astype(np.int64)

        X_tr, X_eval, _ = L.preprocess_fold(
            train_parts, y_train, np.concatenate([val_X, test.X]), feature_cols, args.max_features,
        )
        X_va, X_te = X_eval[:len(val_X)], X_eval[len(val_X):]
        train_parts = val_X = X_eval = None
        if 0 < args.max_context_rows < len(X_tr):
            X_fit, y_fit = L.subsample_train(X_tr, y_train, args.max_context_rows, args.seed)
        else:
            X_fit, y_fit = X_tr, y_train
        print(f"  val={val_pids}  train={len(y_train)} rows (pos={int(y_train.sum())})  "
              f"context={len(y_fit)} (pos={int(y_fit.sum())})  features={X_fit.shape[1]}")

        for s in todo:
            print(f"  → {s.name}: fitting...")
            model = None
            try:
                model = s.build(len(X_fit))
                if s.uses_val:
                    model.fit(X_fit, y_fit, X_val=X_va, y_val=val_y)
                else:
                    model.fit(X_fit, y_fit)
                if hasattr(model, "trainers") and model.trainers:
                    n_ctx = min(model.trainers[0].cfg.hyperparams["max_samples_support"], len(X_fit))
                    print(f"  → {s.name}: Mitra support {n_ctx}/{len(X_fit)} rows (stratified)")
                p_va = model.predict_proba(X_va)[:, 1]
                p_te = model.predict_proba(X_te)[:, 1]
                thr = best_f1_threshold(val_y, p_va)
                metrics = evaluate(test.y, p_te, thr, test.tto, test.event, args)
                save_fold(args.output_dir, s.key, s.name, test_pid, test, p_te, thr, metrics)
                all_results.setdefault(s.name, []).append({
                    "patient": test_pid, "y": test.y, "proba": p_te, "tto": test.tto,
                    "threshold": thr, "metrics": metrics,
                })
                print(f"  → {s.name}: ROC={metrics['ROC AUC']:.4f}  PR={metrics['PR AUC']:.4f}  "
                      f"thr={thr:.3f}  sens={metrics['Recall']:.3f}  spec={metrics['Specificity']:.3f}  "
                      f"seizures={metrics['Seizure Sensitivity']}  FA/h={metrics['FA/h']:.2f}")
            except Exception as e:
                print(f"  → {s.name}: [ERROR] {type(e).__name__}: {e}")
            finally:
                model = None
                L._release_gpu()
        print()
        X_tr = X_fit = X_va = X_te = None
        gc.collect()

    if not all_results:
        print("[ERROR] no results")
        return all_results

    print("=== LOSO SUMMARY ===\n")
    for name, folds in all_results.items():
        print(f"{name}  ({len(folds)} folds)")
        for k, st in aggregate(folds).items():
            print(f"  {k:<20} {st['mean']:.4f} ± {st['std']:.4f}")
        print()

    # Rewrite the CSV from the in-memory results (the per-fold appends may
    # contain duplicates from interrupted runs).
    rows = [{"model": n, "patient": r["patient"], **{k: v for k, v in r["metrics"].items() if v is not None}}
            for n, folds in all_results.items() for r in folds]
    pd.DataFrame(rows).to_csv(args.output_dir / "loso_fold_metrics.csv", index=False)

    graphs = args.output_dir / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    cms = {n: plot_confusion(f, n, graphs / f"confusion_{n.lower().replace(' ', '_')}.png")
           for n, f in all_results.items()}
    plot_roc_pr(all_results, graphs / "roc_pr_curves.png",
                f"seizure within {args.horizon_sec:g}s (SPH {args.sph_sec:g}s)")
    plot_time_to_onset(all_results, graphs / "time_to_onset.png", args)

    report = build_report(args, title, stats_rows, all_results, cms, extra_config or [],
                          extra_sections_fn(all_results) if extra_sections_fn else [])
    (args.output_dir / "loso_report.md").write_text(report)
    print(f"\nReport written to: {args.output_dir / 'loso_report.md'}")
    print("LOSO complete.")
    return all_results


def main() -> None:
    args = build_parser().parse_args()
    resolve_paths(args)
    models = [m for m in args.models if L._model_available(m)]
    missing = [m for m in args.models if m not in models]
    if missing:
        print(f"[WARN] not installed, skipping: {', '.join(missing)}")
    run_loso(args, foundation_specs(models, args.seed), "LOSO seizure prediction — tabular foundation models")


if __name__ == "__main__":
    main()
