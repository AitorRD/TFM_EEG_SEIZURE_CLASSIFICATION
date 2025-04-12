import mne  # Para leer archivos EDF
import pandas as pd
import os  # Para manejar rutas
import re  # Para procesar el archivo de texto
import numpy as np  # Para operaciones numéricas
import tqdm  # Para mostrar la barra de progreso

file_path_read = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0")
file_path_write = os.path.join("data", "raw", "csv-data")
os.makedirs(file_path_write, exist_ok=True)

def time_to_seconds(start_time_str):
    h, m, s = map(int, re.split('[:.]', start_time_str))
    start_seconds = h * 3600 + m * 60 + s
    return start_seconds

def clipping(df, sec=1800):
    seizure_mask = df['Seizure'] == 1
    seiz_times = df.loc[seizure_mask, "Time (s)"]
    start_clip = max(seiz_times.min() - sec, df["Time (s)"].min())
    end_clip = min(seiz_times.max() + sec, df["Time (s)"].max())

    clipped_df = df[(df["Time (s)"] >= start_clip) & (df["Time (s)"] <= end_clip)].reset_index(drop=True)
    return clipped_df

# Iterar sobre cada carpeta de paciente
for patient_folder in tqdm.tqdm(os.listdir(file_path_read), desc="Procesando pacientes"):
    patient_path = os.path.join(file_path_read, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    seizures = {}
    
    # Buscar el archivo de texto con la información de convulsiones
    txt_file = next((f for f in os.listdir(patient_path) if f.endswith(".txt")), None)
    
    if txt_file:
        txt_path = os.path.join(patient_path, txt_file)
        with open(txt_path, "r") as f:
            content = f.read()
        
        # Extraer la información de cada convulsión
        matches = re.findall(
            r"(?:File name:\s*([\w\-.]+\.edf))?\s*"
            r"(?:Registration start time:\s*(\d+[\.:]\d+[\.:]\d+))?\s*"
            r"(?:Registration end time:\s*\d+[\.:]\d+[\.:]\d+)?\s*"
            r"(?:Seizure start time|Start time):\s*(\d+[\.:]\d+[\.:]\d+)\s*"
            r"(?:Seizure end time|End time):\s*(\d+[\.:]\d+[\.:]\d+)",
            content, re.S
        )

        print(f"\nMatches encontrados en {txt_file}:")
        for i, match in enumerate(matches, 1):
            print(match)
            file_name, reg_start, seiz_start, seiz_end = match
            print(f"  [{i}] File: {file_name}, Reg Start: {reg_start}, Seiz Start: {seiz_start}, Seiz End: {seiz_end}")
            
            if not file_name:
                file_name = txt_file.replace(".txt", ".edf")
                
            if file_name not in seizures:
                seizures[file_name] = []

            # Convertir tiempos a segundos absolutos
            if reg_start > seiz_start:
                if reg_start:
                    seiz_start_sec = abs((time_to_seconds(seiz_start.strip()) + 24*3600)  - time_to_seconds(reg_start.strip()))
                    seiz_end_sec = abs((time_to_seconds(seiz_end.strip()) + 24*3600) - time_to_seconds(reg_start.strip()))
            else:
                if reg_start:
                    seiz_start_sec = abs(time_to_seconds(seiz_start.strip()) - time_to_seconds(reg_start.strip()))
                    seiz_end_sec = abs(time_to_seconds(seiz_end.strip()) - time_to_seconds(reg_start.strip()))
            
            if seiz_end_sec < seiz_start_sec:
                time_diff = abs(seiz_start_sec - seiz_end_sec)
            else:   
                time_diff = abs(seiz_end_sec - seiz_start_sec)
            print(f"Tiempo de la convulsión (en segundos): {time_diff}")
                
            seizures[file_name].append({
                "start_time": seiz_start_sec,
                "end_time": seiz_start_sec + time_diff,
            })

        # Manejo especial para el paciente "PN12"
        if "PN12" in patient_folder:
            combined_seizures = {}
            for file_name, conv_list in seizures.items():
                if file_name in combined_seizures:
                    combined_seizures[file_name].extend(conv_list)
                else:
                    combined_seizures[file_name] = conv_list
            seizures = combined_seizures

        # Contar número de convulsiones detectadas (por líneas con "Seizure n")
        seizure_count = len(re.findall(r"Seizure n\s*\d+", content))
        
    print(f"Seizures detected for {patient_folder}: {seizure_count}.")
    print(f"Seizures: {seizures}")

    # Procesar los archivos EDF dentro de la carpeta del paciente
    for file in os.listdir(patient_path):
        if file.endswith(".edf"):
            edf_path = os.path.join(patient_path, file)
            csv_patient_path = os.path.join(file_path_write, patient_folder)
            os.makedirs(csv_patient_path, exist_ok=True)
            csv_path = os.path.join(csv_patient_path, file.replace(".edf", ".csv"))
            
            print(f"Processing {file} in {patient_folder}...")
            
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                print(f"Duración del archivo EDF: {raw.times[-1]} segundos")
                raw.filter(0.1, 70, method='iir')  # Filtro paso banda
                raw.notch_filter(60, method='iir')  # Filtro notch
                raw.resample(100)  # Remuestreo a 100 Hz
                
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
                #if raw.times >= (1800*2+time_diff):
                #    df = clipping(df, sec=1800)
                df.to_csv(csv_path, index=False)

                print(f"Saved: {csv_path}")

            except Exception as e:
                print(f"Error processing {file} in {patient_folder}: {e}")
print("Processing completed.")