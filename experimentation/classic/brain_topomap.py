import pandas as pd
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PATIENT = "PN14"
DATA_PATH = "data/processed/windowed/dataset_windowed_val.csv"
OUTPUT_PATH = f"images/results/brain_topomap_{PATIENT}_preictal_vs_ictal.png"

EEG_CHANNELS = [
    'EEG Fp1', 'EEG Fp2',
    'EEG F7', 'EEG F3', 'EEG Fz', 'EEG F4', 'EEG F8',
    'EEG T3', 'EEG C3', 'EEG Cz', 'EEG C4', 'EEG T4',
    'EEG T5', 'EEG P3', 'EEG Pz', 'EEG P4', 'EEG T6',
    'EEG O1', 'EEG O2'
]

MNE_CHANNEL_NAMES_ALT = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]

MNE_CHANNEL_NAMES = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]


def find_windows_for_patient(df, patient_id, eeg_channels):
    patient_mask = df['window_id'].str.startswith(patient_id)
    patient_df = df[patient_mask]

    if patient_df.empty:
        raise ValueError(f"No windows found for patient: {patient_id}")

    pre_ictal_windows = patient_df[patient_df['Seizure'] == 0]['window_id'].unique()
    ictal_windows     = patient_df[patient_df['Seizure'] == 1]['window_id'].unique()

    if len(pre_ictal_windows) == 0:
        raise ValueError(f"No pre-ictal windows found for patient: {patient_id}")
    if len(ictal_windows) == 0:
        raise ValueError(f"No ictal windows found for patient: {patient_id}")

    pre_ictal_power = {}
    for window_id in pre_ictal_windows:
        window_data = patient_df[patient_df['window_id'] == window_id][eeg_channels]
        if not window_data.empty:
            mean_abs = window_data.abs().mean(axis=0).mean()
            pre_ictal_power[window_id] = mean_abs

    pre_ictal_id = min(pre_ictal_power, key=pre_ictal_power.get)

    ictal_power = {}
    for window_id in ictal_windows:
        window_data = patient_df[patient_df['window_id'] == window_id][eeg_channels]
        if not window_data.empty:
            mean_abs = window_data.abs().mean(axis=0).mean()
            ictal_power[window_id] = mean_abs

    ictal_id = max(ictal_power, key=ictal_power.get)

    print(f"  Pre-ictal mean |µV|: {pre_ictal_power[pre_ictal_id]:.2f} µV")
    print(f"  Ictal mean |µV|:     {ictal_power[ictal_id]:.2f} µV")
    print(f"  Contraste:           x{ictal_power[ictal_id]/pre_ictal_power[pre_ictal_id]:.1f}")

    return pre_ictal_id, ictal_id


def load_window_data(df, window_id):
    return df[df['window_id'] == window_id][EEG_CHANNELS]


def compute_mean_abs(data):
    return data.abs().mean(axis=0).values


def create_mne_info():
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=MNE_CHANNEL_NAMES_ALT, sfreq=100, ch_types='eeg')
        info.set_montage(montage)
        return info, MNE_CHANNEL_NAMES_ALT
    except Exception:
        pass

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
    print(f"  BRAIN TOPOGRAPHIC MAP: Pre-Ictal vs Ictal")
    print(f"  Patient: {PATIENT}")
    print("=" * 60)

    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nSearching windows for patient {PATIENT}...")
    try:
        pre_ictal_window, ictal_window = find_windows_for_patient(df, PATIENT, EEG_CHANNELS)
        print(f"  ✓ Pre-ictal window: {pre_ictal_window}")
        print(f"  ✓ Ictal window:     {ictal_window}")
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print(f"  Available patients: {sorted(df['window_id'].str.extract(r'(PN\d+)')[0].unique())}")
        return

    pre_ictal_data = load_window_data(df, pre_ictal_window)
    ictal_data     = load_window_data(df, ictal_window)

    print(f"  Pre-ictal samples: {len(pre_ictal_data)}")
    print(f"  Ictal samples:     {len(ictal_data)}")

    if pre_ictal_data.empty or ictal_data.empty:
        print("[ERROR] One of the windows is empty.")
        return

    pre_power_uv  = compute_mean_abs(pre_ictal_data)
    ictal_power_uv = compute_mean_abs(ictal_data)
    ratio = ictal_power_uv / (pre_power_uv + 1e-15)

    info, ch_names = create_mne_info()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('white')

    vmin_abs = min(pre_power_uv.min(), ictal_power_uv.min())
    vmax_abs = max(pre_power_uv.max(), ictal_power_uv.max())

    im1, _ = mne.viz.plot_topomap(
        pre_power_uv, info, axes=axes[0],
        cmap='YlOrRd', show=False,
        vlim=(vmin_abs, vmax_abs),
        contours=6, sensors=True, names=None,
    )
    axes[0].set_title('Pre-Ictal (Normal)', fontsize=12, fontweight='bold', pad=10)
    for text in axes[0].texts:
        text.set_fontsize(6)

    im2, _ = mne.viz.plot_topomap(
        ictal_power_uv, info, axes=axes[1],
        cmap='YlOrRd', show=False,
        vlim=(vmin_abs, vmax_abs),
        contours=6, sensors=True, names=None,
    )
    axes[1].set_title('Ictal (Seizure)', fontsize=12, fontweight='bold', pad=10)
    for text in axes[1].texts:
        text.set_fontsize(6)

    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.6, pad=0.08)
    cbar1.set_label('Mean |Amplitude| (µV)', fontsize=9)
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.6, pad=0.08)
    cbar2.set_label('Mean |Amplitude| (µV)', fontsize=9)

    fig.suptitle(
        f'EEG Brain Activity Map'
        'Pre-Ictal vs Ictal Comparison',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Topomap saved: {output_path}")

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