"""
Complete EEG Data Processing Pipeline
======================================
This script executes the entire data preparation process:
1. EDF to CSV conversion (with preprocessing and clipping)
2. Concatenation and train/val/test split
3. Windowing for time series

Usage:
    python data/processed/data.py
    
Or run individual steps:
    python data/processed/data.py --step conversion
    python data/processed/data.py --step concat
    python data/processed/data.py --step window
"""

import argparse
import sys
import time
import os

# Add root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_conversion():
    """
    Step 1: EDF to CSV conversion
    - Reads patient EDF files
    - Applies preprocessing (filters, ICA, resample)
    - Marks seizures according to text files
    - Applies clipping around seizures
    - Saves CSV to data/raw/csv-data/
    """
    print("\n" + "="*70)
    print("  STEP 1: EDF to CSV CONVERSION")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Import conversion script dependencies
    import mne
    import pandas as pd
    import numpy as np
    import re
    import tqdm
    from mne.preprocessing import ICA
    
    file_path_read = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0")
    file_path_write = os.path.join("data", "raw", "csv-data")
    os.makedirs(file_path_write, exist_ok=True)

    def time_to_seconds(start_time_str):
        h, m, s = map(int, re.split('[:.]', start_time_str))
        start_seconds = h * 3600 + m * 60 + s
        return start_seconds

    def clipping(df, freq_hz=100, minutes=30):
        context_samples = minutes * 60 * freq_hz
        seizure_series = df['Seizure']
        seizure_indices = seizure_series[seizure_series == 1].index

        if len(seizure_indices) == 0:
            print("  No seizures detected. Returning original DataFrame without clipping.")
            return df

        # Detect continuous seizure blocks
        seizure_diff = seizure_indices.to_series().diff().fillna(1)
        block_ids = (seizure_diff != 1).cumsum()

        # Create extended ranges with context
        ranges = []
        for i, (block_id, block) in enumerate(seizure_indices.to_series().groupby(block_ids), 1):
            block = block.index if hasattr(block, 'index') else pd.Index([block])
            start = block.min() - context_samples
            end = block.max() + context_samples
            start = max(start, 0)
            end = min(end, df.index[-1])
            print(f"    Block {i}: Clipping from index {start} to {end}")
            ranges.append((start, end))

        # Merge overlapping ranges
        merged_ranges = []
        for start, end in sorted(ranges):
            if not merged_ranges or start > merged_ranges[-1][1]:
                merged_ranges.append([start, end])
            else:
                merged_ranges[-1][1] = max(merged_ranges[-1][1], end)

        print(f"  Ranges after merging: {len(merged_ranges)}")

        # Clip the DataFrame
        clipped_df = pd.concat([df.loc[start:end] for start, end in merged_ranges])
        print(f"  Original size: {len(df)} -> Clipped size: {len(clipped_df)} rows")
        return clipped_df

    # Counters
    files_processed = 0
    files_skipped = 0
    files_failed = 0

    # Process patients
    for patient_folder in tqdm.tqdm(os.listdir(file_path_read), desc="Procesando pacientes"):
        patient_path = os.path.join(file_path_read, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        seizures = {}
        
        # Search for seizures file
        txt_file = next((f for f in os.listdir(patient_path) if f.endswith(".txt")), None)
        
        if txt_file:
            txt_path = os.path.join(patient_path, txt_file)
            with open(txt_path, "r") as f:
                content = f.read()
            
            # Extract seizure information
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

                # Convert times to seconds
                if reg_start > seiz_start:
                    if reg_start:
                        seiz_start_sec = abs((time_to_seconds(seiz_start.strip()) + 24*3600) - time_to_seconds(reg_start.strip()))
                        seiz_end_sec = abs((time_to_seconds(seiz_end.strip()) + 24*3600) - time_to_seconds(reg_start.strip()))
                else:
                    if reg_start:
                        seiz_start_sec = abs(time_to_seconds(seiz_start.strip()) - time_to_seconds(reg_start.strip()))
                        seiz_end_sec = abs(time_to_seconds(seiz_end.strip()) - time_to_seconds(reg_start.strip()))
                
                if seiz_end_sec < seiz_start_sec:
                    time_diff = abs(seiz_start_sec - seiz_end_sec)
                else:   
                    time_diff = abs(seiz_end_sec - seiz_start_sec)
                    
                seizures[file_name].append({
                    "start_time": seiz_start_sec,
                    "end_time": seiz_start_sec + time_diff,
                })

        # Process EDF files
        for file in os.listdir(patient_path):
            if file.endswith(".edf"):
                edf_path = os.path.join(patient_path, file)
                csv_patient_path = os.path.join(file_path_write, patient_folder)
                os.makedirs(csv_patient_path, exist_ok=True)
                csv_path = os.path.join(csv_patient_path, file.replace(".edf", "_clipped.csv"))
                
                # Check if file already exists
                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    files_skipped += 1
                    continue
                
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    lista_canales = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
                    raw.pick(lista_canales)
                    raw.filter(0.5, 70, method='iir')
                    raw.notch_filter(60, method='iir')
                    ica = ICA(n_components=len(raw.ch_names), method='fastica', random_state=0, max_iter=500)
                    ica.fit(raw)
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
                    
                    if df["Time (s)"].max() >= (1800*2 + seizures.get(file, [{}])[0].get("end_time", 0) - seizures.get(file, [{}])[0].get("start_time", 0)):
                        df = clipping(df, freq_hz=100, minutes=30)
                    
                    # Only save if DataFrame has data
                    if len(df) > 0:
                        df.fillna(0, inplace=True)
                        df.to_csv(csv_path, index=False)
                        files_processed += 1
                    else:
                        print(f"  {file}: Empty DataFrame after clipping, not saved.")
                        files_failed += 1

                except Exception as e:
                    print(f"  Error processing {file}: {e}")
                    files_failed += 1
    
    elapsed_time = time.time() - start_time
    print(f"\nConversion completed in {elapsed_time:.2f} seconds")
    print(f"  Files processed: {files_processed}")
    print(f"  Files skipped (already existed): {files_skipped}")
    print(f"  Files failed: {files_failed}")
    print(f"  Files saved to: {file_path_write}\n")


def run_concat():
    """
    Step 2: Concatenation and train/val/test split
    - Reads all clipped CSVs
    - Groups by session
    - Splits into train (70%), val (15%), test (15%)
    - Saves to data/processed/dataset_clipped/
    """
    print("\n" + "="*70)
    print("  STEP 2: CONCATENATION AND TRAIN/VAL/TEST SPLIT")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    root_dir = os.path.join("data", "raw", "csv-data")
    output_dir = os.path.join("data", "processed", "dataset_clipped")
    os.makedirs(output_dir, exist_ok=True)

    channels = ["Time (s)", "Seizure", "idSession", "idPatient",
                "EEG Fp1", "EEG Fp2", "EEG F7", "EEG F3", "EEG Fz", "EEG F4", "EEG F8",
                "EEG T3", "EEG C3", "EEG Cz", "EEG C4", "EEG T4", "EEG T5", "EEG P3",
                "EEG Pz", "EEG P4", "EEG T6", "EEG O1", "EEG O2"]

    session_dfs = {}

    # Read and group by session
    print("Reading CSV files...")
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_clipped.csv'):
                file_path = os.path.join(subdir, file)
                try:
                    # Check if file is not empty
                    if os.path.getsize(file_path) == 0:
                        print(f"  Empty file, skipping: {file}")
                        continue
                    
                    df = pd.read_csv(file_path)
                    
                    # Rename inconsistent columns
                    rename_dict = {}
                    cols = df.columns
                    if "EEG CZ" in cols:
                        rename_dict["EEG CZ"] = "EEG Cz"
                    if "EEG FP2" in cols:
                        rename_dict["EEG FP2"] = "EEG Fp2"
                    df.rename(columns=rename_dict, inplace=True)
                    df.fillna(0, inplace=True)
                    df = df[[col for col in channels if col in df.columns]]

                    for session_id in df["idSession"].unique():
                        df_session = df[df["idSession"] == session_id]
                        session_dfs[session_id] = df_session

                except Exception as e:
                    print(f"  Skipping {file} (possibly empty or corrupted): {e}")
                    continue

    print(f"  Total sessions found: {len(session_dfs)}")

    # Split sessions
    session_ids = list(session_dfs.keys())
    train_ids, temp_ids = train_test_split(session_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"  Train: {len(train_ids)} sessions")
    print(f"  Val:   {len(val_ids)} sessions")
    print(f"  Test:  {len(test_ids)} sessions")

    # Concatenate and save
    train_df = pd.concat([session_dfs[sid] for sid in train_ids], ignore_index=True)
    val_df = pd.concat([session_dfs[sid] for sid in val_ids], ignore_index=True)
    test_df = pd.concat([session_dfs[sid] for sid in test_ids], ignore_index=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    elapsed_time = time.time() - start_time
    print(f"\nConcatenation completed in {elapsed_time:.2f} seconds")
    print(f"  Files saved to: {output_dir}\n")


def run_window():
    """
    Step 3: Windowing
    - Creates sliding windows of 10 seconds
    - 25% overlap
    - Generates windows per session
    - Saves to data/processed/windowed/
    """
    print("\n" + "="*70)
    print("  STEP 3: WINDOWING")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    import pandas as pd
    from tqdm import tqdm
    
    def window_and_save(df, set_name, window_size=1000, overlap=0.25, out_dir='data/processed/windowed'):
        os.makedirs(out_dir, exist_ok=True)
        step = int(window_size * (1 - overlap))
        all_windows = []
        sessions = df['idSession'].unique()
        
        print(f"  [{set_name.upper()}] Windowing {len(sessions)} sessions...")
        for session_id in tqdm(sessions, desc=f"  {set_name.upper()}", leave=False):
            session_df = df[df['idSession'] == session_id].reset_index(drop=True)
            max_start = len(session_df) - window_size
            for start in range(0, max_start + 1, step):
                window = session_df.iloc[start:start+window_size].copy()
                window['window_id'] = f"{session_id}_{start}"
                all_windows.append(window)
        
        if all_windows:
            result = pd.concat(all_windows, ignore_index=True)
            result.to_csv(os.path.join(out_dir, f"dataset_windowed_{set_name}.csv"), index=False)
            print(f"  [{set_name.upper()}] {len(result)} rows saved")

    # Parameters
    input_dir = os.path.join("data", "processed", "dataset_clipped")
    sampling_rate = 100
    window_seconds = 10
    window_size = sampling_rate * window_seconds
    overlap = 0.25

    print(f"  Parameters:")
    print(f"    - Window duration: {window_seconds}s ({window_size} samples)")
    print(f"    - Overlap: {overlap*100}%")
    print(f"    - Step: {int(window_size * (1 - overlap))} samples\n")

    # Read and process datasets
    datasets = {
        "train": pd.read_csv(os.path.join(input_dir, "train.csv")),
        "val": pd.read_csv(os.path.join(input_dir, "val.csv")),
        "test": pd.read_csv(os.path.join(input_dir, "test.csv")),
    }

    for set_name, df in datasets.items():
        window_and_save(df, set_name, window_size=window_size, overlap=overlap)

    elapsed_time = time.time() - start_time
    print(f"\nWindowing completed in {elapsed_time:.2f} seconds")
    print(f"  Files saved to: data/processed/windowed/\n")


def run_all():
    """Executes the complete pipeline"""
    print("\n" + "="*70)
    print("  COMPLETE EEG DATA PROCESSING PIPELINE")
    print("="*70)
    
    total_start = time.time()
    
    try:
        run_conversion()
        run_concat()
        run_window()
        
        total_time = time.time() - total_start
        print("\n" + "="*70)
        print("  PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline de procesamiento de datos EEG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python data/processed/data.py                 # Ejecutar todo el pipeline
  python data/processed/data.py --step conversion  # Solo conversión
  python data/processed/data.py --step concat      # Solo concatenación
  python data/processed/data.py --step window      # Solo ventanado
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['conversion', 'concat', 'window', 'all'],
        default='all',
        help='Specific step to execute (default: all)'
    )
    
    args = parser.parse_args()
    
    # Execute requested step
    if args.step == 'all':
        run_all()
    elif args.step == 'conversion':
        run_conversion()
    elif args.step == 'concat':
        run_concat()
    elif args.step == 'window':
        run_window()


if __name__ == "__main__":
    main()
