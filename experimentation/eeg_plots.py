"""
Raw EEG signal visualization: seizure timelines and multi-channel traces.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLING_RATE = 100

SELECTED_CHANNELS = [
    "EEG Fp1", "EEG F3", "EEG C3", "EEG T3", "EEG P3", "EEG O1",
    "EEG Fp2", "EEG F4", "EEG C4", "EEG T4",
]


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


def plot_raw_eeg_traces(df, file_name, save_dir, duration_minutes=30):
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
                    plot_raw_eeg_traces(df, file, traces_dir, duration_minutes=30)

                except Exception as e:
                    print(f"  [ERROR] {file}: {e}")

    print("\n" + "=" * 80)
    print(f"COMPLETED: Processed {file_count} files")
    print("=" * 80 + "\n")
