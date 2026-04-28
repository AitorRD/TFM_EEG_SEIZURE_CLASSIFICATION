"""
ML result visualization: ROC curves, confusion matrices, and metrics table.

For raw EEG signal visualization (seizure timelines, multi-channel traces)
see experimentation/eeg_plots.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
)

from .utils import get_model_name


def _get_test_data(model_key, ctx):
    if model_key in ctx.get('selectors', {}):
        X = ctx['selectors'][model_key].transform(ctx['X_test'])
    else:
        X = ctx['X_test']
    y = ctx['y_test']

    return X, y


def plot_roc_curves(pipelines, ctx, suffix=""):
    config = ctx['config']

    print(f"\n{'=' * 60}")
    print("  GENERATING ROC CURVES")
    print(f"{'=' * 60}\n")

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e', '#c0392b']
    color_idx = 0

    for model_key, pipeline in pipelines.items():
        model_name = get_model_name(config, model_key)

        if not hasattr(pipeline, 'predict_proba'):
            print(f"  [INFO] {model_name} has no predict_proba, skipping...")
            continue

        X_test_model, y_test_current = _get_test_data(model_key, ctx)

        try:
            y_proba = pipeline.predict_proba(X_test_model)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_current, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
                     linewidth=2.5, color=colors[color_idx % len(colors)])
            color_idx += 1
            print(f"  ✓ {model_name}: AUC = {roc_auc:.4f}")
        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.title('ROC Curves - Test Set', fontsize=24, fontweight='bold')
    plt.legend(loc='lower right', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path(f'images/graphs/roc_curves{suffix}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ ROC curves saved: {save_path}\n")


def plot_confusion_matrices(pipelines, ctx, suffix=""):
    config = ctx['config']

    print(f"\n{'=' * 60}")
    print("  GENERATING CONFUSION MATRICES")
    print(f"{'=' * 60}\n")

    n_models = len(pipelines)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    model_idx = 0
    for model_key, pipeline in pipelines.items():
        model_name = get_model_name(config, model_key)
        X_test_model, y_test_current = _get_test_data(model_key, ctx)

        try:
            y_pred = pipeline.predict(X_test_model)
            cm = confusion_matrix(y_test_current, y_pred)
            tn, fp, fn, tp = cm.ravel()

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['No Seizure (0)', 'Seizure (1)'],
            )

            ax = axes[model_idx]
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)

            ax.set_title(f'{model_name}\nTN={tn}, FP={fp}, FN={fn}, TP={tp}',
                         fontsize=18, fontweight='bold')
            ax.set_xlabel(ax.get_xlabel(), fontsize=17)
            ax.set_ylabel(ax.get_ylabel(), fontsize=17)
            ax.tick_params(axis='both', labelsize=15)
            ax.grid(False)

            for text_obj in disp.text_.ravel():
                text_obj.set_fontsize(20)

            ax.text(0, 0, '\nTN', ha='center', va='top', fontsize=13, color='darkblue', weight='bold')
            ax.text(1, 0, '\nFP', ha='center', va='top', fontsize=13, color='darkred', weight='bold')
            ax.text(0, 1, '\nFN', ha='center', va='top', fontsize=13, color='darkred', weight='bold')
            ax.text(1, 1, '\nTP', ha='center', va='top', fontsize=13, color='darkblue', weight='bold')

            print(f"{model_name}: TN={tn} FP={fp} FN={fn} TP={tp}")
            model_idx += 1
        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")

    for idx in range(model_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = Path(f'images/graphs/confusion_matrices{suffix}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Confusion matrices saved: {save_path}\n")


def plot_metrics_table(results_dict, save_path):
    df = pd.DataFrame(results_dict).T
    df_formatted = df.map(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else x
    )

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df_formatted.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center', cellLoc='center', colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Metrics table saved: {save_path}\n")
