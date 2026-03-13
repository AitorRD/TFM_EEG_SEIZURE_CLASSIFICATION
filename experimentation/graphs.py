"""
Visualization module: ML result plots (ROC curves, confusion matrices, metrics table)
and raw EEG signal visualization (seizure timelines, multi-channel traces).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
)

from .utils import get_model_name
from .models import is_raw_dl_model

SAMPLING_RATE = 100

SELECTED_CHANNELS = [
    "EEG Fp1", "EEG F3", "EEG C3", "EEG T3", "EEG P3", "EEG O1",
    "EEG Fp2", "EEG F4", "EEG C4", "EEG T4",
]


def _get_test_data(model_key, ctx):
    config = ctx['config']

    if is_raw_dl_model(model_key) and ctx.get('X_test_raw') is not None:
        return ctx['X_test_raw'].astype(np.float32), ctx['y_test_raw']

    if model_key in ctx.get('selectors', {}):
        X = ctx['selectors'][model_key].transform(ctx['X_test'])
    else:
        X = ctx['X_test']
    y = ctx['y_test']

    if model_key in config.get('dl_models', {}):
        if hasattr(X, 'values'):
            X = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float32)

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


def plot_seizure_timeline(df, file_name, save_dir):
    if "Seizure" not in df.columns:
        print(f"  [WARNING] 'Seizure' column not found in {file_name}")
        return

    df["Time (min)"] = df.index / (SAMPLING_RATE * 60)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Time (min)"], df["Seizure"], drawstyle='steps-post', color="red", linewidth=1)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Seizure (0 or 1)", fontsize=12)
    plt.title(f"Seizure Timeline - {file_name.replace('_clipped.csv', '')}", fontsize=14, fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    image_name = f"{file_name.replace('.csv', '')}_timeline.png"
    save_path = os.path.join(save_dir, image_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    total_minutes = len(df) / (SAMPLING_RATE * 60)
    print(f"  ✓ Timeline saved: {save_path} - Duration: {total_minutes:.2f} minutes")


def plot_raw_eeg_traces(df, file_name, save_dir, duration_minutes=3):
    if "Seizure" not in df.columns:
        print(f"  [WARNING] 'Seizure' column not found in {file_name}")
        return

    seizure_indices = df[df["Seizure"] == 1].index
    if len(seizure_indices) == 0:
        print(f"  No seizure detected in {file_name}")
        return

    seizure_onset_idx = seizure_indices[0]
    seizure_onset_time = seizure_onset_idx / SAMPLING_RATE

    preictal_duration = 120
    start_time = max(0, seizure_onset_time - preictal_duration)
    end_time = min(len(df) / SAMPLING_RATE, start_time + (duration_minutes * 60))

    start_idx = int(start_time * SAMPLING_RATE)
    end_idx = int(end_time * SAMPLING_RATE)

    df_window = df.iloc[start_idx:end_idx].copy()
    df_window["Time (s)"] = (df_window.index - start_idx) / SAMPLING_RATE

    available_channels = [ch for ch in SELECTED_CHANNELS if ch in df.columns]
    if len(available_channels) == 0:
        print(f"  No selected channels available in {file_name}")
        return

    fig, axes = plt.subplots(len(available_channels), 1, figsize=(14, 2 * len(available_channels)), sharex=True)
    if len(available_channels) == 1:
        axes = [axes]

    for i, channel in enumerate(available_channels):
        axes[i].plot(df_window["Time (s)"], df_window[channel], color='black', linewidth=0.5)
        axes[i].set_ylabel(f"{channel.replace('EEG ', '')} (µV)", fontsize=10)
        axes[i].grid(True, alpha=0.3)

        seizure_onset_relative = (seizure_onset_idx - start_idx) / SAMPLING_RATE
        if 0 <= seizure_onset_relative <= df_window["Time (s)"].max():
            axes[i].axvline(seizure_onset_relative, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            if i == 0:
                axes[i].text(seizure_onset_relative, axes[i].get_ylim()[1] * 0.9,
                             'Seizure Onset', color='red', fontsize=10, ha='right')

        if seizure_onset_relative > 0:
            axes[i].axvspan(0, seizure_onset_relative, alpha=0.1, color='blue', label='Preictal')
        if seizure_onset_relative < df_window["Time (s)"].max():
            axes[i].axvspan(seizure_onset_relative, df_window["Time (s)"].max(),
                            alpha=0.1, color='red', label='Ictal')

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_title(f"Raw EEG signals - {file_name.replace('_clipped.csv', '')}", fontsize=14, fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    save_path = os.path.join(save_dir, file_name.replace('.csv', '_raw_traces.png'))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Raw traces saved: {save_path} (Duration: {duration_minutes} min, Seizure onset: {seizure_onset_time:.1f}s)")


def process_all_eeg_files(data_dir="data/raw/csv-data",
                          timeline_dir="images/seizure_timelines",
                          traces_dir="images/raw_eeg_traces"):
    os.makedirs(timeline_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING EEG VISUALIZATIONS")
    print("=" * 80 + "\n")

    file_count = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("clipped.csv"):
                full_csv_path = os.path.join(root, file)
                file_count += 1

                print(f"\n[{file_count}] Processing: {file}")
                print("-" * 80)

                try:
                    df = pd.read_csv(full_csv_path)

                    if "Seizure" not in df.columns:
                        print(f"  [WARNING] 'Seizure' column not found in {file}")
                        continue

                    plot_seizure_timeline(df, file, timeline_dir)
                    plot_raw_eeg_traces(df, file, traces_dir, duration_minutes=3)

                except Exception as e:
                    print(f"  [ERROR] {file}: {e}")

    print("\n" + "=" * 80)
    print(f"COMPLETED: Processed {file_count} files")
    print("=" * 80 + "\n")
