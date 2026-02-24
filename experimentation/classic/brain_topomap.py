"""
Brain Topographic Map: Pre-Ictal vs Ictal EEG Activity
Generates a brain heatmap showing EEG power distribution before and during a seizure.

Patient: PN05 (selected for the most dramatic pre-ictal vs ictal contrast)
- Pre-ictal window: PN05-2_147000 (normal activity)
- Ictal window: PN05-3_180000 (seizure activity, ~x48 mean power increase)

Uses the 10-20 International System electrode positions.
"""

import pandas as pd
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
PATIENT = "PN05"
PRE_ICTAL_WINDOW = "PN05-2_147000"    # Normal activity (before seizure)
ICTAL_WINDOW = "PN05-3_180000"       # Seizure activity (during seizure)
DATA_PATH = "data/processed/windowed/dataset_windowed_train.csv"
OUTPUT_PATH = f"images/results/brain_topomap_{PATIENT}_preictal_vs_ictal.png"

# EEG channels in 10-20 system order
EEG_CHANNELS = [
    'EEG Fp1', 'EEG Fp2',
    'EEG F7', 'EEG F3', 'EEG Fz', 'EEG F4', 'EEG F8',
    'EEG T3', 'EEG C3', 'EEG Cz', 'EEG C4', 'EEG T4',
    'EEG T5', 'EEG P3', 'EEG Pz', 'EEG P4', 'EEG T6',
    'EEG O1', 'EEG O2'
]

# MNE standard channel names (10-20 system)
MNE_CHANNEL_NAMES = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Alternative names for MNE compatibility (T3->T7, T4->T8, T5->P7, T6->P8)
MNE_CHANNEL_NAMES_ALT = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]


def load_window_data(df, window_id):
    """Extract EEG data for a specific window"""
    window_data = df[df['window_id'] == window_id][EEG_CHANNELS]
    return window_data


def compute_band_power(data, sfreq=100):
    """
    Compute power in different frequency bands per channel.
    Returns total RMS power per channel (good for overall activity visualization).
    """
    power = np.sqrt((data ** 2).mean(axis=0))
    return power.values


def create_mne_info():
    """Create MNE Info object with 10-20 electrode positions"""
    # Try with alternative names first (MNE prefers T7/T8/P7/P8)
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=MNE_CHANNEL_NAMES_ALT, sfreq=100, ch_types='eeg')
        info.set_montage(montage)
        return info, MNE_CHANNEL_NAMES_ALT
    except Exception:
        pass
    
    # Fallback to original names
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=MNE_CHANNEL_NAMES, sfreq=100, ch_types='eeg')
        info.set_montage(montage)
        return info, MNE_CHANNEL_NAMES
    except Exception as e:
        print(f"Error creating montage: {e}")
        raise


def main():
    print("=" * 60)
    print("  BRAIN TOPOGRAPHIC MAP: Pre-Ictal vs Ictal")
    print(f"  Patient: {PATIENT}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Extract windows
    pre_ictal_data = load_window_data(df, PRE_ICTAL_WINDOW)
    ictal_data = load_window_data(df, ICTAL_WINDOW)
    
    print(f"  Pre-ictal window: {PRE_ICTAL_WINDOW} ({len(pre_ictal_data)} samples)")
    print(f"  Ictal window:     {ICTAL_WINDOW} ({len(ictal_data)} samples)")
    
    # Compute power per channel (RMS in volts)
    pre_power = compute_band_power(pre_ictal_data)
    ictal_power = compute_band_power(ictal_data)
    
    # Convert from Volts to microvolts (µV) — standard EEG unit
    pre_power_uv = pre_power * 1e6
    ictal_power_uv = ictal_power * 1e6
    
    # Compute ratio for difference map
    ratio = ictal_power_uv / (pre_power_uv + 1e-15)  # avoid division by zero
    
    # Create MNE info with electrode positions
    info, ch_names = create_mne_info()
    
    # ============================================================
    # CREATE FIGURE: 2 topomaps side by side
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('white')
    
    # Common colormap limits for pre-ictal and ictal (in µV)
    vmin_abs = min(pre_power_uv.min(), ictal_power_uv.min())
    vmax_abs = max(pre_power_uv.max(), ictal_power_uv.max())
    
    # --- 1. Pre-Ictal Topomap ---
    im1, _ = mne.viz.plot_topomap(
        pre_power_uv, info, axes=axes[0],
        cmap='YlOrRd', show=False,
        vlim=(vmin_abs, vmax_abs),
        contours=6,
        sensors=True,
        names=ch_names,
    )
    axes[0].set_title(
        'Pre-Ictal (Normal)',
        fontsize=12, fontweight='bold', pad=10
    )
    
    # --- 2. Ictal Topomap ---
    im2, _ = mne.viz.plot_topomap(
        ictal_power_uv, info, axes=axes[1],
        cmap='YlOrRd', show=False,
        vlim=(vmin_abs, vmax_abs),
        contours=6,
        sensors=True,
        names=ch_names,
    )
    axes[1].set_title(
        'Ictal (Seizure)',
        fontsize=12, fontweight='bold', pad=10
    )
    
    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.6, pad=0.08)
    cbar1.set_label('RMS Amplitude (µV)', fontsize=9)
    
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.6, pad=0.08)
    cbar2.set_label('RMS Amplitude (µV)', fontsize=9)
    
    # Suptitle
    fig.suptitle(
        'EEG Brain Activity Map\n'
        'Pre-Ictal vs Ictal Comparison (19 channels, 10-20 System) — RMS Amplitude in µV',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Topomap saved: {output_path}")
    
    # Print power summary
    print(f"\n{'='*60}")
    print(f"  POWER SUMMARY PER CHANNEL (µV)")
    print(f"{'='*60}")
    print(f"{'Channel':<10} {'Pre-Ictal (µV)':>15} {'Ictal (µV)':>15} {'Ratio':>8}")
    print("-" * 50)
    for i, ch in enumerate(EEG_CHANNELS):
        ch_short = ch.replace('EEG ', '')
        marker = " ▲" if ratio[i] > 1.5 else (" ▼" if ratio[i] < 0.7 else "")
        print(f"{ch_short:<10} {pre_power_uv[i]:>15.2f} {ictal_power_uv[i]:>15.2f} {ratio[i]:>7.1f}x{marker}")
    print(f"\nMean ratio: {ratio.mean():.2f}x")
    print(f"Max increase: {ratio.max():.1f}x ({EEG_CHANNELS[ratio.argmax()].replace('EEG ', '')})")


if __name__ == "__main__":
    main()
