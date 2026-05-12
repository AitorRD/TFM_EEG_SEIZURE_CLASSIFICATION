"""
Generate summary plots from LOSO results.

Reads:
  - loso_fold_metrics.csv      → metrics table (mean ± std)
  - predictions/*_loso_predictions.csv  → ROC curves for all models

Usage:
    python plot_loso_results.py
    python plot_loso_results.py --output-dir images/results/loso
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


METRICS_OF_INTEREST = ["Accuracy", "Precision", "Recall", "F1 Score", "F1 Macro", "ROC AUC"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LOSO results")
    p.add_argument("--output-dir", type=str, default="images/results/loso")
    p.add_argument("--csv", type=str, default=None,
                   help="Path to loso_fold_metrics.csv (overrides --output-dir for the CSV)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

_ROC_EXCLUDE = {"chronos", "moirai", "tsmixer"}


def plot_metrics_table(df: pd.DataFrame, output_dir: Path) -> None:
    available = [m for m in METRICS_OF_INTEREST if m in df.columns]
    agg = df.groupby("model")[available].agg(["mean", "std"])
    models = agg.index.tolist()

    table_data = [
        [f"{agg.loc[m, (metric, 'mean')]:.2f} ± {agg.loc[m, (metric, 'std')]:.2f}"
         for metric in available]
        for m in models
    ]

    # Index of best model per metric (highest mean)
    best_row = {
        metric: int(np.argmax([agg.loc[m, (metric, "mean")] for m in models]))
        for metric in available
    }

    fig, ax = plt.subplots(figsize=(max(10, len(available) * 1.8), len(models) * 0.75 + 1.4))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=available,
        rowLabels=models,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for j, metric in enumerate(available):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
        # Bold + highlight best cell in each column
        best_i = best_row[metric]
        cell = table[(best_i + 1, j)]
        cell.set_facecolor("#d5f5e3")
        cell.set_text_props(fontweight="bold")

    for i in range(len(models)):
        table[(i + 1, -1)].set_facecolor("#ecf0f1")
        table[(i + 1, -1)].set_text_props(fontweight="bold")

    ax.set_title("LOSO — mean ± std across patients", fontsize=11, pad=16, fontweight="bold")
    fig.tight_layout()

    path = output_dir / "loso_metrics_table.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(output_dir: Path) -> None:
    preds_dir = output_dir / "predictions"
    if not preds_dir.exists():
        print(f"  [SKIP] Predictions directory not found: {preds_dir}")
        return

    pred_files = sorted(preds_dir.glob("*_loso_predictions.csv"))
    if not pred_files:
        print(f"  [SKIP] No prediction files found in {preds_dir}")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random (AUC=0.500)")

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]

    for idx, pred_file in enumerate(pred_files):
        model_key = pred_file.name.replace("_loso_predictions.csv", "")
        if model_key.lower() in _ROC_EXCLUDE:
            print(f"  [SKIP ROC] {model_key} excluded")
            continue
        df = pd.read_csv(pred_file)

        y_true  = df["y_true"].values
        y_score = df["y_score"].values

        if np.unique(y_true).size < 2:
            print(f"  [SKIP] {model_key}: single class in concatenated labels")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, linewidth=2.2, color=color,
                label=f"{model_key} (AUC={roc_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — LOSO (all folds concatenated)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "loso_roc_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    csv_path = Path(args.csv) if args.csv else output_dir / "loso_fold_metrics.csv"
    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows — models: {df['model'].unique().tolist()}")

    print("\nGenerating plots...")
    plot_metrics_table(df, output_dir)
    plot_roc_curves(output_dir)

    print("\nDone. Files saved to:", output_dir)


if __name__ == "__main__":
    main()
