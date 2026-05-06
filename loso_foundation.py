"""
LOSO (Leave-One-Subject-Out) with foundation models for EEG seizure classification.

For each fold one complete patient is left out as the test set; all remaining
patients are used for threshold tuning (and TSMixer training).

  - Zero-shot models (Chronos2, Moirai2): loaded ONCE, reused across every fold.
  - Trainable model (TSMixer): re-created and re-fitted from scratch each fold.

Outputs (under --output-dir, default images/results/loso/):
  loso_fold_metrics.csv          — per-fold, per-model metrics
  loso_metrics_table.png         — aggregated metrics table (mean ± std)
  graphs/loso_confusion_<model>.png  — summed confusion matrix across folds
  graphs/loso_roc_curves.png     — ROC curves (concatenated predictions)

Usage:
    python loso_foundation.py
    python loso_foundation.py --model chronos2 --channel "EEG F3"
    python loso_foundation.py --models chronos2 moirai2 --max-windows 128
    python loso_foundation.py --model tsmixer --tsmixer-epochs 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from foundation_models import (
    MODEL_DISPLAY_NAMES,
    SplitData,
    TSMixerForecaster,
    build_scores,
    create_forecaster,
    evaluate_classification,
    resolve_models,
    select_threshold,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOSO foundation-model seizure classification over EEG windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", choices=["chronos2", "moirai2", "tsmixer"], default=None)
    p.add_argument(
        "--models", nargs="+",
        choices=["chronos2", "moirai2", "tsmixer"], default=None,
    )

    p.add_argument("--channel", type=str, default="EEG F3")
    p.add_argument("--context-length", type=int, default=800)
    p.add_argument("--prediction-length", type=int, default=200)
    p.add_argument(
        "--max-windows", type=int, default=0,
        help="Max val windows per fold used for threshold tuning (0 = all)",
    )
    p.add_argument("--window-order", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--raw-dir", type=str, default="data/raw/csv-data",
        help="Root directory with per-patient clipped CSVs (PN_XX/*_clipped.csv)",
    )
    p.add_argument("--window-size", type=int, default=1000,
                   help="Samples per window (default 1000 = 10 s at 100 Hz)")
    p.add_argument("--window-overlap", type=float, default=0.25,
                   help="Window overlap fraction (default 0.25)")
    p.add_argument(
        "--output-dir", type=str, default="images/results/loso",
        help="Root output directory for LOSO results",
    )

    # Chronos2 / Moirai2
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chronos-model-id", type=str, default="amazon/chronos-2")
    p.add_argument("--moirai2-model-id", type=str, default="Salesforce/moirai-2.0-R-small")
    p.add_argument("--batch-size", type=int, default=32)

    # TSMixer
    p.add_argument("--tsmixer-epochs", type=int, default=20)
    p.add_argument("--tsmixer-learning-rate", type=float, default=1e-3)
    p.add_argument("--tsmixer-dropout", type=float, default=0.1)
    p.add_argument("--tsmixer-hidden-dim", type=int, default=128)
    p.add_argument("--tsmixer-blocks", type=int, default=3)
    p.add_argument("--tsmixer-val-ratio", type=float, default=0.1)
    p.add_argument("--tsmixer-device", type=str, default="cpu", choices=["cpu", "gpu"])

    # XAI
    p.add_argument(
        "--xai-enabled",
        action="store_true",
        help="Enable LOSO explainability for each tested patient and global averages",
    )
    p.add_argument(
        "--xai-methods",
        nargs="+",
        default=["shap", "lime", "embedded"],
        choices=["shap", "lime", "embedded"],
        help="XAI methods to run per fold",
    )
    p.add_argument(
        "--xai-dir",
        type=str,
        default=None,
        help="Output directory for XAI (default: <output-dir>/xai)",
    )
    p.add_argument(
        "--xai-segments",
        type=int,
        default=20,
        help="Number of context segments used by SHAP/LIME perturbations",
    )
    p.add_argument(
        "--xai-max-samples",
        type=int,
        default=256,
        help="Max test windows per patient used for XAI",
    )
    p.add_argument(
        "--xai-top-features",
        type=int,
        default=20,
        help="Top features shown in SHAP/LIME plots",
    )
    p.add_argument(
        "--xai-patient-instances",
        type=int,
        default=24,
        help="Number of test windows explained per patient",
    )
    p.add_argument(
        "--xai-lime-perturbations",
        type=int,
        default=1000,
        help="Synthetic samples used by each LIME local explanation",
    )
    p.add_argument(
        "--xai-shap-background",
        type=int,
        default=24,
        help="Background windows used by SHAP KernelExplainer",
    )
    p.add_argument(
        "--xai-shap-nsamples",
        type=int,
        default=128,
        help="Kernel SHAP evaluation budget per explained window",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_RENAME_COLS = {"EEG CZ": "EEG Cz", "EEG FP2": "EEG Fp2"}


def _window_df(df: pd.DataFrame, window_size: int, overlap: float) -> pd.DataFrame:
    """Apply sliding window per session to a patient DataFrame."""
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
    """Read all clipped CSVs for one patient and apply sliding window."""
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


def load_and_window_all_patients(
    raw_dir: str,
    window_size: int,
    overlap: float,
) -> pd.DataFrame:
    """Load and window every patient from raw/csv-data."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    patient_ids = sorted(d.name for d in raw_path.iterdir() if d.is_dir())
    if not patient_ids:
        raise FileNotFoundError(f"No patient subdirectories found in {raw_dir}")

    parts = []
    for pid in patient_ids:
        windowed = load_and_window_patient(pid, raw_dir, window_size, overlap)
        if len(windowed) > 0:
            parts.append(windowed)
            print(f"  [{pid}] {len(windowed):,} windows")

    if not parts:
        raise RuntimeError("No windowed data produced from any patient.")

    df = pd.concat(parts, ignore_index=True)
    print(f"  Total: {len(df):,} rows — {df['idPatient'].nunique()} patients")
    return df


def df_to_split_data(
    split_name: str,
    df: pd.DataFrame,
    channel: str,
    context_length: int,
    prediction_length: int,
    max_windows: int,
    window_order: str,
    seed: int,
) -> Optional[SplitData]:
    """Build a SplitData from a (pre-filtered) DataFrame. Returns None if empty."""
    missing = {"window_id", "Seizure", channel}.difference(df.columns)
    if missing:
        raise ValueError(f"[{split_name}] Missing columns: {sorted(missing)}")

    grouped = df.groupby("window_id", sort=False)
    window_ids = list(grouped.groups.keys())

    if window_order == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(window_ids)

    if max_windows > 0:
        window_ids = window_ids[:max_windows]

    min_len = context_length + prediction_length
    series: Dict[str, np.ndarray] = {}
    labels: Dict[str, int] = {}

    for wid in window_ids:
        g = grouped.get_group(wid)
        values = g[channel].to_numpy(dtype=np.float32)
        if len(values) < min_len:
            continue
        series[str(wid)] = values[:min_len]
        labels[str(wid)] = int(g["Seizure"].max())

    final_ids = [wid for wid in window_ids if wid in series]
    if not final_ids:
        return None

    n_sz = sum(1 for wid in final_ids if labels[wid] == 1)
    print(
        f"    [{split_name.upper()}] {len(final_ids)} windows "
        f"(seizure={n_sz}, non-seizure={len(final_ids) - n_sz})"
    )
    return SplitData(
        split=split_name, window_ids=final_ids, series=series, labels=labels
    )


# ---------------------------------------------------------------------------
# XAI helpers (direct model explainability)
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


def _xai_build_segment_features(
    split_data: SplitData,
    window_ids: List[str],
    context_length: int,
    n_segments: int,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    n_segments = max(1, min(n_segments, context_length))
    segment_edges = np.linspace(0, context_length, n_segments + 1, dtype=np.int64)
    feature_names = [f"s{idx + 1:02d}_mean" for idx in range(n_segments)]

    X = np.zeros((len(window_ids), n_segments), dtype=np.float32)
    for row_idx, window_id in enumerate(window_ids):
        context = np.asarray(split_data.series[window_id][:context_length], dtype=np.float32)
        for seg_idx in range(n_segments):
            start = int(segment_edges[seg_idx])
            end = int(segment_edges[seg_idx + 1])
            if end <= start:
                end = min(context_length, start + 1)
            segment = context[start:end] if end > start else context[start:start + 1]
            if segment.size == 0:
                segment = context
            X[row_idx, seg_idx] = float(np.mean(segment))
    return X, feature_names, segment_edges


def _xai_extract_future_targets(
    split_data: SplitData,
    window_ids: List[str],
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    futures = np.zeros((len(window_ids), prediction_length), dtype=np.float32)
    for row_idx, window_id in enumerate(window_ids):
        values = np.asarray(split_data.series[window_id], dtype=np.float32)
        target = values[context_length:context_length + prediction_length]
        if len(target) != prediction_length:
            raise RuntimeError(
                f"Window {window_id}: expected {prediction_length} future samples, got {len(target)}."
            )
        futures[row_idx] = target
    return futures


def _xai_segment_means_to_context(
    segment_means: np.ndarray,
    segment_edges: np.ndarray,
    context_length: int,
) -> np.ndarray:
    context = np.zeros(context_length, dtype=np.float32)
    n_segments = min(len(segment_means), len(segment_edges) - 1)
    for seg_idx in range(n_segments):
        start = int(segment_edges[seg_idx])
        end = int(segment_edges[seg_idx + 1])
        if end <= start:
            end = min(context_length, start + 1)
        context[start:end] = float(segment_means[seg_idx])
    return context


def _xai_predict_scores_with_foundation_model(
    X_segment: np.ndarray,
    future_target: np.ndarray,
    forecaster: Any,
    segment_edges: np.ndarray,
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    X_arr = np.asarray(X_segment, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    n_rows = X_arr.shape[0]
    baseline = float(np.mean(np.abs(future_target - np.mean(future_target))))
    scores = np.full(n_rows, baseline, dtype=np.float32)

    series_dict: Dict[str, np.ndarray] = {}
    for row_idx in range(n_rows):
        context = _xai_segment_means_to_context(
            X_arr[row_idx], segment_edges=segment_edges, context_length=context_length
        )
        series_dict[str(row_idx)] = np.concatenate(
            [context, future_target.astype(np.float32)], axis=0
        ).astype(np.float32)

    preds_by_id = forecaster.forecast(series_dict)
    for row_idx in range(n_rows):
        pred = preds_by_id.get(str(row_idx))
        if pred is None:
            continue
        y_pred = np.asarray(pred, dtype=np.float32)[:prediction_length]
        if len(y_pred) != prediction_length:
            continue
        scores[row_idx] = float(np.mean(np.abs(future_target - y_pred)))

    return scores


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

    colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in selected_values] if signed else ["#34495e"] * len(selected_values)

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
    feature_names: List[str],
    values: np.ndarray,
    save_path: Path,
    value_col: str = "importance",
) -> pd.DataFrame:
    df = pd.DataFrame({"feature": feature_names, value_col: np.asarray(values, dtype=np.float32)})
    df = df.sort_values(by=value_col, ascending=False).reset_index(drop=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"    XAI CSV saved: {save_path}")
    return df


def _xai_run_shap_for_patient(
    model_key: str,
    model_name: str,
    patient_id: str,
    forecaster: Any,
    X: np.ndarray,
    y_targets: np.ndarray,
    scores: np.ndarray,
    feature_names: List[str],
    segment_edges: np.ndarray,
    args: argparse.Namespace,
    patient_dir: Path,
) -> Optional[np.ndarray]:
    try:
        import shap
    except ImportError:
        print("    [INFO] SHAP not installed, skipping SHAP.")
        return None

    if len(X) < 3:
        print(f"    [INFO] {patient_id}: not enough samples for SHAP.")
        return None

    explain_idx = _xai_choose_indices(scores, args.xai_patient_instances, seed=args.seed + 7)
    if len(explain_idx) == 0:
        print(f"    [INFO] {patient_id}: no windows selected for SHAP.")
        return None

    collected_rows: List[np.ndarray] = []
    explained_features: List[np.ndarray] = []

    for idx in explain_idx:
        pool = np.array([i for i in range(len(X)) if i != idx], dtype=np.int64)
        if len(pool) == 0:
            pool = np.array([idx], dtype=np.int64)

        bg_local_idx = _xai_sample_indices(
            total=len(pool),
            max_samples=max(2, args.xai_shap_background),
            seed=args.seed + int(idx) * 17,
        )
        background = X[pool[bg_local_idx]]
        future_target = y_targets[idx]

        def score_fn(data: np.ndarray) -> np.ndarray:
            return _xai_predict_scores_with_foundation_model(
                X_segment=data,
                future_target=future_target,
                forecaster=forecaster,
                segment_edges=segment_edges,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
            )

        try:
            explainer = shap.KernelExplainer(score_fn, background)
            shap_values = explainer.shap_values(
                X[idx:idx + 1],
                nsamples=max(16, args.xai_shap_nsamples),
            )
            row_values = np.asarray(shap_values, dtype=np.float32).reshape(-1)
            if row_values.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Unexpected SHAP row shape {row_values.shape}, expected {X.shape[1]}."
                )
            collected_rows.append(row_values)
            explained_features.append(X[idx])
        except Exception as exc:
            print(f"    [INFO] SHAP failed for patient={patient_id}, idx={idx}: {exc}")

    if not collected_rows:
        print(f"    [INFO] {patient_id}: SHAP produced no usable explanations.")
        return None

    shap_matrix = np.vstack(collected_rows)
    X_explained = np.vstack(explained_features)
    mean_abs = np.mean(np.abs(shap_matrix), axis=0).astype(np.float32)

    top_k = min(args.xai_top_features, len(feature_names), shap_matrix.shape[1])
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_matrix[:, top_idx],
        features=X_explained[:, top_idx],
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

    _xai_plot_importance_bars(
        feature_names=feature_names,
        values=mean_abs,
        title=f"SHAP Importance - {model_name} - {patient_id}",
        xlabel="Mean |SHAP value|",
        save_path=patient_dir / "shap_importance.png",
        top_k=args.xai_top_features,
        signed=False,
    )
    _xai_save_importance_csv(
        feature_names=feature_names,
        values=mean_abs,
        save_path=patient_dir / "shap_importance.csv",
        value_col="importance",
    )
    return mean_abs


def _xai_run_lime_for_patient(
    model_key: str,
    model_name: str,
    patient_id: str,
    forecaster: Any,
    X: np.ndarray,
    y_targets: np.ndarray,
    scores: np.ndarray,
    feature_names: List[str],
    segment_edges: np.ndarray,
    args: argparse.Namespace,
    patient_dir: Path,
) -> Optional[np.ndarray]:
    try:
        import lime.lime_tabular
    except ImportError:
        print("    [INFO] LIME not installed, skipping LIME.")
        return None

    if len(X) < 2:
        print(f"    [INFO] {patient_id}: not enough samples for LIME.")
        return None

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        training_labels=scores,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=args.seed,
    )

    explain_idx = _xai_choose_indices(scores, args.xai_patient_instances, seed=args.seed + 13)
    collected: List[pd.Series] = []

    for idx in explain_idx:
        future_target = y_targets[idx]

        def score_fn(data: np.ndarray) -> np.ndarray:
            return _xai_predict_scores_with_foundation_model(
                X_segment=data,
                future_target=future_target,
                forecaster=forecaster,
                segment_edges=segment_edges,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
            )

        exp = explainer.explain_instance(
            X[idx],
            score_fn,
            num_features=min(args.xai_top_features, len(feature_names)),
            num_samples=max(200, args.xai_lime_perturbations),
        )

        local_weights: Dict[str, float] = {}
        for feature_desc, weight in exp.as_list():
            mapped_name = None
            for fname in feature_names:
                if fname in feature_desc:
                    mapped_name = fname
                    break
            if mapped_name is None:
                mapped_name = feature_desc
            local_weights[mapped_name] = float(weight)

        row = pd.Series(local_weights).reindex(feature_names, fill_value=0.0)
        collected.append(row)

    if not collected:
        print(f"    [INFO] {patient_id}: LIME produced no explanations.")
        return None

    avg_signed = pd.concat(collected, axis=1).mean(axis=1).to_numpy(dtype=np.float32)
    avg_abs = np.abs(avg_signed).astype(np.float32)

    _xai_plot_importance_bars(
        feature_names=feature_names,
        values=avg_signed,
        title=f"LIME Importance - {model_name} - {patient_id}",
        xlabel="Mean local contribution (negative to positive)",
        save_path=patient_dir / "lime_importance.png",
        top_k=args.xai_top_features,
        signed=True,
    )

    df_lime = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": avg_abs,
            "signed_importance": avg_signed,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)
    lime_csv = patient_dir / "lime_importance.csv"
    df_lime.to_csv(lime_csv, index=False)
    print(f"    XAI CSV saved: {lime_csv}")
    return avg_abs


def _xai_run_embedded_tsmixer_for_patient(
    model_name: str,
    patient_id: str,
    forecaster: TSMixerForecaster,
    split_data: SplitData,
    window_ids: List[str],
    args: argparse.Namespace,
    patient_dir: Path,
) -> Optional[np.ndarray]:
    subset_series = {wid: split_data.series[wid] for wid in window_ids}
    saliency, _ = forecaster.explain_saliency(subset_series, max_samples=args.xai_max_samples)
    saliency = saliency[:args.context_length].astype(np.float32)

    x_axis = np.arange(1, len(saliency) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, saliency, color="#c0392b", linewidth=1.7)
    plt.fill_between(x_axis, saliency, alpha=0.25, color="#e74c3c")
    plt.title(f"Embedded Saliency - {model_name} - {patient_id}")
    plt.xlabel("Context timestep")
    plt.ylabel("Mean absolute gradient")
    plt.tight_layout()
    saliency_plot = patient_dir / "embedded_saliency.png"
    plt.savefig(saliency_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    XAI plot saved: {saliency_plot}")

    saliency_csv = patient_dir / "embedded_saliency.csv"
    pd.DataFrame(
        {"timestep": np.arange(1, len(saliency) + 1), "importance": saliency}
    ).to_csv(saliency_csv, index=False)
    print(f"    XAI CSV saved: {saliency_csv}")
    return saliency


def run_fold_xai(
    patient_id: str,
    model_key: str,
    model_name: str,
    forecaster: Any,
    test_data: SplitData,
    test_ids: List[str],
    test_scores: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    if not args.xai_enabled:
        return {}

    xai_root = Path(args.xai_dir) if args.xai_dir else Path(args.output_dir) / "xai"
    patient_dir = xai_root / model_key / "patients" / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    X_full, feature_names, segment_edges = _xai_build_segment_features(
        split_data=test_data,
        window_ids=test_ids,
        context_length=args.context_length,
        n_segments=args.xai_segments,
    )
    y_targets_full = _xai_extract_future_targets(
        split_data=test_data,
        window_ids=test_ids,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    scores_full = np.asarray(test_scores, dtype=np.float32)

    selected_idx = _xai_sample_indices(
        total=len(test_ids),
        max_samples=args.xai_max_samples,
        seed=args.seed + len(patient_id) * 31,
    )
    X = X_full[selected_idx]
    y_targets = y_targets_full[selected_idx]
    scores = scores_full[selected_idx]

    if len(scores) < 5:
        print(f"    [INFO] {patient_id}: not enough sampled windows for XAI.")
        return {}

    print(f"    Running XAI for patient {patient_id}...")
    method_set = set(args.xai_methods)
    out: Dict[str, np.ndarray] = {}

    if "shap" in method_set:
        shap_imp = _xai_run_shap_for_patient(
            model_key=model_key,
            model_name=model_name,
            patient_id=patient_id,
            forecaster=forecaster,
            X=X,
            y_targets=y_targets,
            scores=scores,
            feature_names=feature_names,
            segment_edges=segment_edges,
            args=args,
            patient_dir=patient_dir,
        )
        if shap_imp is not None:
            out["shap"] = shap_imp

    if "lime" in method_set:
        lime_imp = _xai_run_lime_for_patient(
            model_key=model_key,
            model_name=model_name,
            patient_id=patient_id,
            forecaster=forecaster,
            X=X,
            y_targets=y_targets,
            scores=scores,
            feature_names=feature_names,
            segment_edges=segment_edges,
            args=args,
            patient_dir=patient_dir,
        )
        if lime_imp is not None:
            out["lime"] = lime_imp

    if (
        "embedded" in method_set
        and model_key == "tsmixer"
        and isinstance(forecaster, TSMixerForecaster)
    ):
        saliency = _xai_run_embedded_tsmixer_for_patient(
            model_name=model_name,
            patient_id=patient_id,
            forecaster=forecaster,
            split_data=test_data,
            window_ids=test_ids,
            args=args,
            patient_dir=patient_dir,
        )
        if saliency is not None:
            out["embedded"] = saliency
    elif "embedded" in method_set:
        print(f"    [INFO] {patient_id}: embedded explainability is only available for TSMixer.")

    meta_path = patient_dir / "meta.csv"
    pd.DataFrame(
        {
            "patient": [patient_id],
            "model": [model_name],
            "num_test_windows": [len(test_ids)],
            "num_xai_windows": [len(selected_idx)],
            "xai_methods": [",".join(sorted(out.keys())) if out else ""],
        }
    ).to_csv(meta_path, index=False)
    return out


def save_global_xai_averages(
    model_key: str,
    model_name: str,
    fold_results: List[Dict],
    args: argparse.Namespace,
) -> None:
    if not args.xai_enabled:
        return

    xai_root = Path(args.xai_dir) if args.xai_dir else Path(args.output_dir) / "xai"
    global_dir = xai_root / model_key / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [f"s{idx + 1:02d}_mean" for idx in range(max(1, min(args.xai_segments, args.context_length)))]

    for method in ["shap", "lime"]:
        per_patient = [
            r["xai"][method]
            for r in fold_results
            if isinstance(r.get("xai"), dict) and method in r["xai"]
        ]
        if not per_patient:
            continue

        matrix = np.vstack(per_patient).astype(np.float32)
        mean_importance = matrix.mean(axis=0)
        std_importance = matrix.std(axis=0)

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_importance": mean_importance,
                "std_importance": std_importance,
            }
        ).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)
        csv_path = global_dir / f"{method}_importance_mean.csv"
        df.to_csv(csv_path, index=False)
        print(f"  XAI global CSV saved: {csv_path}")

        _xai_plot_importance_bars(
            feature_names=feature_names,
            values=mean_importance,
            title=f"{method.upper()} Mean Importance - {model_name} (all patients)",
            xlabel="Mean importance across patients",
            save_path=global_dir / f"{method}_importance_mean.png",
            top_k=args.xai_top_features,
            signed=False,
        )

    per_patient_embedded = [
        r["xai"]["embedded"]
        for r in fold_results
        if isinstance(r.get("xai"), dict) and "embedded" in r["xai"]
    ]
    if per_patient_embedded:
        matrix = np.vstack(per_patient_embedded).astype(np.float32)
        mean_saliency = matrix.mean(axis=0)
        std_saliency = matrix.std(axis=0)
        timesteps = np.arange(1, len(mean_saliency) + 1)

        csv_path = global_dir / "embedded_saliency_mean.csv"
        pd.DataFrame(
            {
                "timestep": timesteps,
                "mean_importance": mean_saliency,
                "std_importance": std_saliency,
            }
        ).to_csv(csv_path, index=False)
        print(f"  XAI global CSV saved: {csv_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, mean_saliency, color="#8e44ad", linewidth=1.8)
        plt.fill_between(
            timesteps,
            np.maximum(0.0, mean_saliency - std_saliency),
            mean_saliency + std_saliency,
            alpha=0.2,
            color="#9b59b6",
        )
        plt.title(f"Embedded Saliency Mean - {model_name} (all patients)")
        plt.xlabel("Context timestep")
        plt.ylabel("Mean absolute gradient")
        plt.tight_layout()
        plot_path = global_dir / "embedded_saliency_mean.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  XAI global plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Per-fold execution
# ---------------------------------------------------------------------------

def run_fold(
    patient_id: str,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_key: str,
    base_forecaster,
    args: argparse.Namespace,
) -> Optional[Dict]:
    """
    Run one LOSO fold for one model.

    Returns a dict with keys: patient_id, metrics, y_true, y_pred, y_score,
    window_ids, threshold. Returns None if the fold must be skipped.
    """
    # Test split: entire left-out patient, no window limit
    test_data = df_to_split_data(
        "test", test_df, args.channel,
        args.context_length, args.prediction_length,
        max_windows=0, window_order="first", seed=args.seed,
    )
    # Val split: all other patients, limited by --max-windows for efficiency
    val_data = df_to_split_data(
        "val", val_df, args.channel,
        args.context_length, args.prediction_length,
        max_windows=args.max_windows,
        window_order=args.window_order,
        seed=args.seed,
    )

    if test_data is None:
        print(f"    [SKIP] {patient_id}: no valid test windows")
        return None
    if val_data is None:
        print(f"    [SKIP] {patient_id}: no valid val windows for threshold tuning")
        return None

    # TSMixer must be re-created and re-fitted for each fold
    if model_key == "tsmixer":
        forecaster = TSMixerForecaster(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            batch_size=args.batch_size,
            epochs=args.tsmixer_epochs,
            learning_rate=args.tsmixer_learning_rate,
            dropout=args.tsmixer_dropout,
            hidden_dim=args.tsmixer_hidden_dim,
            num_blocks=args.tsmixer_blocks,
            val_ratio=args.tsmixer_val_ratio,
            seed=args.seed,
            device=args.tsmixer_device,
        )
        train_data = df_to_split_data(
            "train", val_df, args.channel,
            args.context_length, args.prediction_length,
            max_windows=args.max_windows,
            window_order=args.window_order,
            seed=args.seed - 1,
        )
        if train_data is None:
            print(f"    [SKIP] {patient_id}: no valid train windows for TSMixer")
            return None
        print(f"    Fitting TSMixer on {len(train_data.window_ids)} windows...")
        forecaster.fit(train_data.series)
    else:
        forecaster = base_forecaster  # zero-shot: reuse the already-loaded model

    # Threshold tuning on val (remaining patients)
    val_preds = forecaster.forecast(val_data.series)
    val_ids, y_val, val_scores = build_scores(
        val_data, val_preds, args.context_length, args.prediction_length,
    )
    threshold = select_threshold(val_scores, y_val)

    # Evaluation on test (left-out patient)
    test_preds = forecaster.forecast(test_data.series)
    test_ids, y_test, test_scores = build_scores(
        test_data, test_preds, args.context_length, args.prediction_length,
    )

    if len(test_ids) == 0:
        print(f"    [SKIP] {patient_id}: no aligned test predictions")
        return None

    metrics, y_pred = evaluate_classification(y_test, test_scores, threshold)
    xai_outputs = run_fold_xai(
        patient_id=patient_id,
        model_key=model_key,
        model_name=MODEL_DISPLAY_NAMES[model_key],
        forecaster=forecaster,
        test_data=test_data,
        test_ids=test_ids,
        test_scores=test_scores,
        args=args,
    )

    return {
        "patient_id": patient_id,
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score": test_scores,
        "window_ids": test_ids,
        "threshold": threshold,
        "xai": xai_outputs,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(
    fold_results: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std for each metric across folds (skips None values)."""
    buckets: Dict[str, List[float]] = {}
    for res in fold_results:
        for metric, value in res["metrics"].items():
            if value is not None:
                buckets.setdefault(metric, []).append(value)
    return {
        m: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for m, vals in buckets.items()
    }


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


def plot_loso_roc_curves(
    all_model_results: Dict[str, List[Dict]],
    save_path: Path,
) -> None:
    """ROC curves built from concatenated predictions across all folds per model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1.5)

    plotted = 0
    for model_name, fold_results in all_model_results.items():
        y_true_all = np.concatenate([r["y_true"] for r in fold_results])
        y_score_all = np.concatenate([r["y_score"] for r in fold_results])

        if np.unique(y_true_all).size < 2:
            print(f"  [INFO] {model_name}: ROC skipped (single class in concatenated labels)")
            continue

        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{model_name} (AUC={roc_auc_val:.3f})")
        plotted += 1

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Foundation Models (LOSO)")
    ax.grid(True, alpha=0.3)
    if plotted > 0:
        ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved: {save_path}")


def plot_loso_metrics_table(
    all_aggregated: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Path,
) -> None:
    """Metrics table showing 'mean ± std' per model × metric."""
    rows = {}
    all_metrics: List[str] = []
    for model_name, agg in all_aggregated.items():
        for m in agg:
            if m not in all_metrics:
                all_metrics.append(m)

    for model_name, agg in all_aggregated.items():
        rows[model_name] = {
            m: f"{agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}" if m in agg else "N/A"
            for m in all_metrics
        }

    df = pd.DataFrame(rows).T  # models as rows, metrics as columns
    df = df[all_metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(all_metrics) * 1.6), len(rows) * 0.7 + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title("Foundation Models — LOSO (mean ± std)", fontsize=12, pad=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics table saved: {save_path}")


def save_fold_metrics_csv(
    all_model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Save per-fold per-model metrics to a single CSV."""
    rows = []
    for model_name, fold_results in all_model_results.items():
        for res in fold_results:
            row = {"model": model_name, "patient": res["patient_id"]}
            row.update({k: v for k, v in res["metrics"].items() if v is not None})
            rows.append(row)

    path = output_dir / "loso_fold_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Per-fold metrics saved: {path}")


def print_loso_summary(
    model_name: str,
    fold_results: List[Dict],
    aggregated: Dict[str, Dict[str, float]],
) -> None:
    print(f"\n{model_name} — LOSO Summary ({len(fold_results)} folds completed)")
    print("-" * 52)
    for metric, stats in aggregated.items():
        print(f"  {metric:<14} {stats['mean']:.4f} ± {stats['std']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    models = resolve_models(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.xai_dir is None:
        args.xai_dir = str(output_dir / "xai")

    print("=" * 60)
    print("  LOSO — FOUNDATION MODELS — EEG SEIZURE CLASSIFICATION")
    print("=" * 60)
    print(f"  Models         : {', '.join(MODEL_DISPLAY_NAMES[m] for m in models)}")
    print(f"  Channel        : {args.channel}")
    print(f"  Context length : {args.context_length}")
    print(f"  Pred. length   : {args.prediction_length}")
    print(f"  Max val windows: {args.max_windows if args.max_windows > 0 else 'all'}")
    print(f"  XAI enabled    : {args.xai_enabled}")
    if args.xai_enabled:
        print(f"  XAI methods    : {', '.join(args.xai_methods)}")
        print(f"  XAI output dir : {args.xai_dir}")

    print(f"\nLoading and windowing from {args.raw_dir} ...")
    print(f"  Window: {args.window_size} samples ({args.window_size // 100}s), overlap {args.window_overlap:.0%}")
    all_df = load_and_window_all_patients(args.raw_dir, args.window_size, args.window_overlap)

    patients = sorted(all_df["idPatient"].unique())
    print(f"  Patients ({len(patients)}): {patients}\n")

    all_model_results: Dict[str, List[Dict]] = {}
    all_aggregated: Dict[str, Dict] = {}

    for model_key in models:
        model_name = MODEL_DISPLAY_NAMES[model_key]
        print(f"\n{'=' * 60}")
        print(f"  {model_name.upper()} — LOSO  ({len(patients)} folds)")
        print("=" * 60)

        # Zero-shot models are loaded once and reused across folds
        if model_key != "tsmixer":
            print(f"  Loading {model_name} (zero-shot — loaded once for all folds)...")
            base_forecaster = create_forecaster(model_key, args)
        else:
            base_forecaster = None

        fold_results: List[Dict] = []

        for fold_idx, patient_id in enumerate(patients):
            print(f"\n  Fold {fold_idx + 1}/{len(patients)}  test={patient_id}")

            test_df = all_df[all_df["idPatient"] == patient_id]
            val_df = all_df[all_df["idPatient"] != patient_id]

            result = run_fold(
                patient_id=patient_id,
                test_df=test_df,
                val_df=val_df,
                model_key=model_key,
                base_forecaster=base_forecaster,
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
            print(f"  [WARN] No valid folds completed for {model_name}")
            continue

        aggregated = aggregate_metrics(fold_results)
        all_model_results[model_name] = fold_results
        all_aggregated[model_name] = aggregated

        print_loso_summary(model_name, fold_results, aggregated)

        cm_path = output_dir / "graphs" / f"loso_confusion_{model_key}.png"
        plot_loso_confusion_matrix(fold_results, model_name, cm_path)

        # Save concatenated predictions so plot_loso_results.py can draw ROC curves
        preds_dir = output_dir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "y_true":  np.concatenate([r["y_true"]  for r in fold_results]),
            "y_score": np.concatenate([r["y_score"] for r in fold_results]),
        }).to_csv(preds_dir / f"{model_key}_loso_predictions.csv", index=False)

        save_global_xai_averages(
            model_key=model_key,
            model_name=model_name,
            fold_results=fold_results,
            args=args,
        )

    if not all_aggregated:
        print("\n[ERROR] No results to aggregate.")
        return

    roc_path = output_dir / "graphs" / "loso_roc_curves.png"
    plot_loso_roc_curves(all_model_results, roc_path)

    metrics_path = output_dir / "loso_metrics_table.png"
    plot_loso_metrics_table(all_aggregated, metrics_path)

    save_fold_metrics_csv(all_model_results, output_dir)

    print("\n" + "=" * 60)
    print("  LOSO COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
