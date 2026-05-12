"""
OBSOLETO — sustituido por loso_foundation.ipynb.

LOSO (Leave-One-Subject-Out) with foundation models for EEG seizure classification.

For each fold one complete patient is left out as the test set; all remaining
patients are used for threshold tuning.

  - Zero-shot models (Chronos2, Moirai2): loaded ONCE, reused across every fold.

Outputs (under --output-dir, default images/results/loso/):
  loso_fold_metrics.csv          — per-fold, per-model metrics
  loso_metrics_table.png         — aggregated metrics table (mean ± std)
  graphs/loso_confusion_<model>.png  — summed confusion matrix across folds
  graphs/loso_roc_curves.png     — ROC curves (concatenated predictions)

Usage:
    python loso_foundation.py
    python loso_foundation.py --model chronos2 --channel "EEG F3"
    python loso_foundation.py --models chronos2 moirai2 --max-windows 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from foundation_models import (
    MODEL_DISPLAY_NAMES,
    SplitData,
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

    p.add_argument("--model", choices=["chronos2", "moirai2"], default=None)
    p.add_argument(
        "--models", nargs="+",
        choices=["chronos2", "moirai2"], default=None,
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

    forecaster = base_forecaster

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

    return {
        "patient_id": patient_id,
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score": test_scores,
        "window_ids": test_ids,
        "threshold": threshold,
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

    print("=" * 60)
    print("  LOSO — FOUNDATION MODELS — EEG SEIZURE CLASSIFICATION")
    print("=" * 60)
    print(f"  Models         : {', '.join(MODEL_DISPLAY_NAMES[m] for m in models)}")
    print(f"  Channel        : {args.channel}")
    print(f"  Context length : {args.context_length}")
    print(f"  Pred. length   : {args.prediction_length}")
    print(f"  Max val windows: {args.max_windows if args.max_windows > 0 else 'all'}")

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

        print(f"  Loading {model_name} (zero-shot — loaded once for all folds)...")
        base_forecaster = create_forecaster(model_key, args)

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
