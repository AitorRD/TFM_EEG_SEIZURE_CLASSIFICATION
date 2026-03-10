"""
EEG F3 Channel Comparison: Pre-Ictal vs Ictal per Patient
=========================================================
For each patient, plots the raw EEG F3 signal from:
  - A pre-ictal (no seizure) window
  - An ictal (seizure) window

Uses dataset_windowed_train.csv (windowed data with raw signal).
One figure per patient saved to images/F3/

Also generates a summary figure with all patients side by side.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
WINDOWED_PATHS = [
    "data/processed/windowed/dataset_windowed_train.csv",
    "data/processed/windowed/dataset_windowed_val.csv",
    "data/processed/windowed/dataset_windowed_test.csv",
]
OUTPUT_DIR = Path("images/F3")
TARGET_CHANNEL = "EEG F3"
SFREQ = 100  # Sampling frequency (Hz)
PEAK_DISTANCE = 3  # Same as tsfresh number_peaks(n=3)


def load_windowed_data():
    """Load all windowed datasets, keeping only relevant columns"""
    cols = ['Seizure', 'idPatient', 'idSession', TARGET_CHANNEL, 'window_id']
    dfs = []
    for path in WINDOWED_PATHS:
        p = Path(path)
        if p.exists():
            print(f"  Loading {p.name}...")
            df = pd.read_csv(p, usecols=cols)
            dfs.append(df)
        else:
            print(f"  [SKIP] {p} not found")
    return pd.concat(dfs, ignore_index=True)


def select_representative_windows(df, patient):
    """Select one seizure and one pre-ictal window for a patient.
    
    For the seizure window: picks the one with highest signal variance (most dramatic).
    For the pre-ictal window: picks a no-seizure window from the same session as the
    seizure (closest to the seizure onset), for a fair pre-ictal comparison.
    """
    patient_data = df[df['idPatient'] == patient]
    
    # Get seizure windows
    seizure_wids = patient_data[patient_data['Seizure'] == 1]['window_id'].unique()
    no_seizure_wids = patient_data[patient_data['Seizure'] == 0]['window_id'].unique()
    
    if len(seizure_wids) == 0 or len(no_seizure_wids) == 0:
        return None, None
    
    # Pick seizure window with highest variance (most dramatic seizure)
    best_sz_var = -1
    best_sz_wid = seizure_wids[0]
    for wid in seizure_wids:
        signal = patient_data[patient_data['window_id'] == wid][TARGET_CHANNEL].values
        v = np.var(signal)
        if v > best_sz_var:
            best_sz_var = v
            best_sz_wid = wid
    
    # Extract session from the seizure window_id (e.g. "PN05-3_180000" -> "PN05-3")
    sz_session = best_sz_wid.rsplit('_', 1)[0]
    
    # Try to find a no-seizure window from the same session (true pre-ictal)
    same_session_nosz = [w for w in no_seizure_wids if w.startswith(sz_session)]
    
    if same_session_nosz:
        # Pick the last no-seizure window before the seizure (closest pre-ictal)
        # Window IDs encode time offset, so sort numerically
        def get_offset(wid):
            return int(wid.rsplit('_', 1)[1])
        
        sz_offset = get_offset(best_sz_wid)
        pre_ictal_candidates = [(w, get_offset(w)) for w in same_session_nosz 
                                 if get_offset(w) < sz_offset]
        
        if pre_ictal_candidates:
            # Closest to seizure onset
            best_nosz_wid = max(pre_ictal_candidates, key=lambda x: x[1])[0]
        else:
            # Any from same session
            best_nosz_wid = same_session_nosz[0]
    else:
        # Fallback: pick a no-seizure window with lowest variance (calm baseline)
        best_nosz_var = float('inf')
        best_nosz_wid = no_seizure_wids[0]
        for wid in no_seizure_wids[:50]:  # limit search
            signal = patient_data[patient_data['window_id'] == wid][TARGET_CHANNEL].values
            v = np.var(signal)
            if v < best_nosz_var:
                best_nosz_var = v
                best_nosz_wid = wid
    
    return best_nosz_wid, best_sz_wid


def plot_patient_f3(df, patient, preictal_wid, ictal_wid, output_dir):
    """Generate a figure for one patient comparing pre-ictal vs ictal F3 signal"""
    patient_data = df[df['idPatient'] == patient]
    
    preictal_signal = patient_data[patient_data['window_id'] == preictal_wid][TARGET_CHANNEL].values
    ictal_signal = patient_data[patient_data['window_id'] == ictal_wid][TARGET_CHANNEL].values
    
    # Convert to µV for readability
    preictal_uv = preictal_signal * 1e6
    ictal_uv = ictal_signal * 1e6
    
    # Time axis
    time_pre = np.arange(len(preictal_uv)) / SFREQ
    time_ict = np.arange(len(ictal_uv)) / SFREQ
    
    # Detect peaks
    pre_peaks, _ = find_peaks(preictal_signal, distance=PEAK_DISTANCE)
    ict_peaks, _ = find_peaks(ictal_signal, distance=PEAK_DISTANCE)
    
    # Stats
    pre_rms = np.sqrt(np.mean(preictal_signal ** 2)) * 1e6
    ict_rms = np.sqrt(np.mean(ictal_signal ** 2)) * 1e6
    ratio = ict_rms / (pre_rms + 1e-15)
    
    # ── Figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.patch.set_facecolor('white')
    
    # Pre-ictal
    ax1 = axes[0]
    ax1.plot(time_pre, preictal_uv, color='#3498db', linewidth=0.5, alpha=0.9)
    # Mark peaks (subsample for visual clarity)
    peak_show = pre_peaks[::3] if len(pre_peaks) > 100 else pre_peaks
    ax1.scatter(peak_show / SFREQ, preictal_uv[peak_show],
                color='#2980b9', marker='v', s=15, zorder=5, alpha=0.6)
    ax1.set_title(f'Pre-Ictal (No Seizure) — {preictal_wid}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (µV)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.98, 0.95,
             f'RMS: {pre_rms:.2f} µV\nPeaks (n=3): {len(pre_peaks)}',
             transform=ax1.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#d6eaf8', alpha=0.9))
    
    # Ictal
    ax2 = axes[1]
    ax2.plot(time_ict, ictal_uv, color='#e74c3c', linewidth=0.5, alpha=0.9)
    peak_show = ict_peaks[::3] if len(ict_peaks) > 100 else ict_peaks
    ax2.scatter(peak_show / SFREQ, ictal_uv[peak_show],
                color='#c0392b', marker='v', s=15, zorder=5, alpha=0.6)
    ax2.set_title(f'Ictal (Seizure) — {ictal_wid}', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude (µV)', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.98, 0.95,
             f'RMS: {ict_rms:.2f} µV\nPeaks (n=3): {len(ict_peaks)}\n'
             f'RMS ratio: {ratio:.1f}x',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.9))
    
    fig.suptitle(f'EEG F3 Channel — Patient {patient}\nPre-Ictal vs Ictal Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    save_path = output_dir / f"F3_{patient}_preictal_vs_ictal.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'patient': patient,
        'preictal_wid': preictal_wid,
        'ictal_wid': ictal_wid,
        'pre_rms_uv': pre_rms,
        'ict_rms_uv': ict_rms,
        'rms_ratio': ratio,
        'pre_peaks': len(pre_peaks),
        'ict_peaks': len(ict_peaks),
    }


def plot_summary(all_stats, df, output_dir):
    """Generate a summary figure with all patients in a grid"""
    n_patients = len(all_stats)
    n_cols = 2
    n_rows = n_patients
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.patch.set_facecolor('white')
    
    for i, stats in enumerate(all_stats):
        patient = stats['patient']
        patient_data = df[df['idPatient'] == patient]
        
        # Pre-ictal
        pre_signal = patient_data[patient_data['window_id'] == stats['preictal_wid']][TARGET_CHANNEL].values * 1e6
        time_pre = np.arange(len(pre_signal)) / SFREQ
        
        ax_pre = axes[i, 0]
        ax_pre.plot(time_pre, pre_signal, color='#3498db', linewidth=0.4, alpha=0.8)
        ax_pre.set_ylabel(f'{patient}', fontsize=10, fontweight='bold', rotation=0, labelpad=40)
        ax_pre.grid(True, alpha=0.2)
        ax_pre.tick_params(labelsize=8)
        ax_pre.text(0.98, 0.92, f'{stats["pre_rms_uv"]:.1f} µV\n{stats["pre_peaks"]} peaks',
                    transform=ax_pre.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='#d6eaf8', alpha=0.8))
        
        if i == 0:
            ax_pre.set_title('Pre-Ictal (No Seizure)', fontsize=12, fontweight='bold')
        
        # Ictal
        ict_signal = patient_data[patient_data['window_id'] == stats['ictal_wid']][TARGET_CHANNEL].values * 1e6
        time_ict = np.arange(len(ict_signal)) / SFREQ
        
        ax_ict = axes[i, 1]
        ax_ict.plot(time_ict, ict_signal, color='#e74c3c', linewidth=0.4, alpha=0.8)
        ax_ict.grid(True, alpha=0.2)
        ax_ict.tick_params(labelsize=8)
        ax_ict.text(0.98, 0.92, f'{stats["ict_rms_uv"]:.1f} µV\n{stats["ict_peaks"]} peaks\n{stats["rms_ratio"]:.1f}x',
                    transform=ax_ict.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))
        
        if i == 0:
            ax_ict.set_title('Ictal (Seizure)', fontsize=12, fontweight='bold')
        
        # Set same Y limits for both panels per patient
        ymax = max(np.max(np.abs(pre_signal)), np.max(np.abs(ict_signal))) * 1.1
        ax_pre.set_ylim(-ymax, ymax)
        ax_ict.set_ylim(-ymax, ymax)
    
    # X labels only on bottom row
    axes[-1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=10)
    
    fig.suptitle('EEG F3 Channel — All Patients\nPre-Ictal vs Ictal Comparison',
                 fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    save_path = output_dir / "F3_all_patients_summary.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Summary figure saved: {save_path}")


def main():
    print("=" * 60)
    print("  EEG F3 CHANNEL: PRE-ICTAL vs ICTAL PER PATIENT")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading windowed data (this may take a moment)...")
    df = load_windowed_data()
    
    patients = sorted(df['idPatient'].unique())
    print(f"\nPatients found: {patients}")
    
    all_stats = []
    
    for patient in patients:
        print(f"\n{'─'*40}")
        print(f"  Patient: {patient}")
        print(f"{'─'*40}")
        
        preictal_wid, ictal_wid = select_representative_windows(df, patient)
        
        if preictal_wid is None or ictal_wid is None:
            print(f"  [SKIP] No seizure/no-seizure windows available")
            continue
        
        print(f"  Pre-ictal window: {preictal_wid}")
        print(f"  Ictal window:     {ictal_wid}")
        
        stats = plot_patient_f3(df, patient, preictal_wid, ictal_wid, OUTPUT_DIR)
        all_stats.append(stats)
        
        print(f"  ✓ Saved: F3_{patient}_preictal_vs_ictal.png")
        print(f"    RMS Pre-ictal: {stats['pre_rms_uv']:.2f} µV | Ictal: {stats['ict_rms_uv']:.2f} µV | Ratio: {stats['rms_ratio']:.1f}x")
        print(f"    Peaks Pre-ictal: {stats['pre_peaks']} | Ictal: {stats['ict_peaks']}")
    
    # Summary figure
    if all_stats:
        print("\nGenerating summary figure...")
        plot_summary(all_stats, df, OUTPUT_DIR)
    
    # Print stats table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Patient':<10} {'Pre-RMS (µV)':>14} {'Ict-RMS (µV)':>14} {'Ratio':>8} {'Pre-Peaks':>10} {'Ict-Peaks':>10}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s['patient']:<10} {s['pre_rms_uv']:>14.2f} {s['ict_rms_uv']:>14.2f} {s['rms_ratio']:>7.1f}x {s['pre_peaks']:>10} {s['ict_peaks']:>10}")
    
    print(f"\n✓ All figures saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
