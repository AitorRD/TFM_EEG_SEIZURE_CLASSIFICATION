import argparse
import sys
import time
import os
import re

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def _time_to_seconds(time_str):
    h, m, s = map(int, re.split('[:.]', time_str))
    return h * 3600 + m * 60 + s


def _clip_around_seizures(df, seizure_infos, minutes=30):
    context_seconds = minutes * 60
    times = df['Time (s)'].values
    t_min, t_max = times[0], times[-1]

    ranges = []
    for i, info in enumerate(seizure_infos, 1):
        clip_start = max(info['start_time'] - context_seconds, t_min)
        clip_end = min(info['end_time'] + context_seconds, t_max)
        duration = clip_end - clip_start
        print(f"    Seizure {i}: {clip_start:.1f}s – {clip_end:.1f}s ({duration/60:.1f} min)")
        ranges.append((clip_start, clip_end))

    # Merge overlapping ranges
    merged_ranges = []
    for start, end in sorted(ranges):
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)

    print(f"  Ranges after merging: {len(merged_ranges)}")

    t = df['Time (s)']
    clipped_df = pd.concat([
        df[(t >= start) & (t <= end)] for start, end in merged_ranges
    ])
    print(f"  Original size: {len(df)} -> Clipped size: {len(clipped_df)} rows ({len(clipped_df)/100/60:.1f} min)")
    return clipped_df


def run_conversion():
    print("\n" + "="*70)
    print("  STEP 1: EDF to CSV CONVERSION")
    print("="*70 + "\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("  Using GPU 1 (RTX 3090)")

    n_jobs = 16
    print(f"  Using {n_jobs} CPU cores for parallel processing\n")

    start_time = time.time()

    import mne
    import tqdm
    from mne.preprocessing import ICA

    file_path_read = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0")
    file_path_write = os.path.join("data", "processed", "dataset_clipped_30")
    os.makedirs(file_path_write, exist_ok=True)

    files_processed = 0
    files_skipped = 0
    files_failed = 0

    for patient_folder in tqdm.tqdm(sorted(os.listdir(file_path_read)), desc="Processing patients"):
        patient_path = os.path.join(file_path_read, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        seizures = {}

        txt_file = next((f for f in os.listdir(patient_path) if f.endswith(".txt")), None)

        if txt_file:
            txt_path = os.path.join(patient_path, txt_file)
            with open(txt_path, "r") as f:
                content = f.read()

            matches = re.findall(
                r"(?:File name:\s*([\w\-.]+\.edf))?\s*"
                r"(?:Registration start time:\s*(\d+[\.:]\d+[\.:]\d+))?\s*"
                r"(?:Registration end time:\s*\d+[\.:]\d+[\.:]\d+)?\s*"
                r"(?:Seizure start time|Start time):\s*(\d+[\.:]\d+[\.:]\d+)\s*"
                r"(?:Seizure end time|End time):\s*(\d+[\.:]\d+[\.:]\d+)",
                content, re.S
            )

            for match in matches:
                file_name, reg_start, seiz_start, seiz_end = match

                if not file_name:
                    file_name = txt_file.replace(".txt", ".edf")

                if file_name not in seizures:
                    seizures[file_name] = []

                if reg_start > seiz_start:
                    if reg_start:
                        seiz_start_sec = abs((_time_to_seconds(seiz_start.strip()) + 24*3600) - _time_to_seconds(reg_start.strip()))
                        seiz_end_sec = abs((_time_to_seconds(seiz_end.strip()) + 24*3600) - _time_to_seconds(reg_start.strip()))
                else:
                    if reg_start:
                        seiz_start_sec = abs(_time_to_seconds(seiz_start.strip()) - _time_to_seconds(reg_start.strip()))
                        seiz_end_sec = abs(_time_to_seconds(seiz_end.strip()) - _time_to_seconds(reg_start.strip()))

                if seiz_end_sec < seiz_start_sec:
                    time_diff = abs(seiz_start_sec - seiz_end_sec)
                else:
                    time_diff = abs(seiz_end_sec - seiz_start_sec)

                seizures[file_name].append({
                    "start_time": seiz_start_sec,
                    "end_time": seiz_start_sec + time_diff,
                })

        for file in os.listdir(patient_path):
            if file.endswith(".edf"):
                edf_path = os.path.join(patient_path, file)
                csv_patient_path = os.path.join(file_path_write, patient_folder)
                os.makedirs(csv_patient_path, exist_ok=True)
                csv_path = os.path.join(csv_patient_path, file.replace(".edf", "_clipped.csv"))

                #if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                #    files_skipped += 1
                #    continue

                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    lista_canales = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
                    raw.pick(lista_canales)
                    raw.filter(0.5, 70, method='iir', n_jobs=n_jobs)
                    raw.notch_filter(60, method='iir', n_jobs=n_jobs)
                    ica = ICA(n_components=len(raw.ch_names), method='fastica', random_state=0, max_iter=500)
                    ica.fit(raw, verbose=False)
                    raw = ica.apply(raw)
                    raw.resample(100)
                    data, times = raw.get_data(return_times=True)
                    df = pd.DataFrame(data.T, columns=raw.ch_names)
                    df.insert(0, "Time (s)", times)
                    seizure_flag = np.zeros(len(times), dtype=int)

                    if file in seizures:
                        for seizure_info in seizures[file]:
                            seiz_start = seizure_info["start_time"]
                            seiz_end = seizure_info["end_time"]
                            mask = (times >= seiz_start) & (times <= seiz_end)
                            seizure_flag[mask] = 1

                    df.insert(1, "Seizure", seizure_flag)
                    df.insert(2, "idPatient", patient_folder)
                    df.insert(3, "idSession", file.replace(".edf", ""))

                    # Apply clipping around seizures if seizures exist
                    if file in seizures and seizures[file]:
                        df = _clip_around_seizures(df, seizures[file], minutes=30)

                    df.fillna(0, inplace=True)
                    df.to_csv(csv_path, index=False)
                    files_processed += 1

                except Exception as e:
                    print(f"  Error processing {file}: {e}")
                    files_failed += 1

    elapsed_time = time.time() - start_time
    print(f"\nConversion completed in {elapsed_time:.2f} seconds")
    print(f"  Files processed: {files_processed}")
    print(f"  Files skipped (already existed): {files_skipped}")
    print(f"  Files failed: {files_failed}")
    print(f"  Files saved to: {file_path_write}\n")


def main():
    parser = argparse.ArgumentParser(
        description='EEG data processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data.py   # EDF to CSV (filters + clip ±30 min around seizures)
        """
    )
    parser.parse_args()
    run_conversion()


if __name__ == "__main__":
    main()
