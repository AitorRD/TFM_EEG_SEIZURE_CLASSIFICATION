import mne  # Para leer archivos EDF
import pandas as pd
import os  # Para manejar rutas
import re  # Para procesar el archivo de texto
import numpy as np  # Para operaciones numéricas
import tqdm  # Para mostrar la barra de progreso
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
        print("No se detectaron convulsiones. No se aplica clipping.")
        return pd.DataFrame(columns=df.columns)

    # Detectar bloques continuos de seizure usando diferencias
    seizure_diff = seizure_indices.to_series().diff().fillna(1)
    block_ids = (seizure_diff != 1).cumsum()

    # Crear rangos extendidos por contexto
    ranges = []
    for i, (block_id, block) in enumerate(seizure_indices.to_series().groupby(block_ids), 1):
        block = block.index if hasattr(block, 'index') else pd.Index([block])
        start = block.min() - context_samples
        end = block.max() + context_samples
        start = max(start, 0)
        end = min(end, df.index[-1])
        print(f"  Bloque {i}: Recorte desde índice {start} hasta {end}")
        ranges.append((start, end))


    # Fusionar rangos solapados
    merged_ranges = []
    for start, end in sorted(ranges):
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)

    print("Rangos después de la fusión de solapamientos:")
    for i, (start, end) in enumerate(merged_ranges, 1):
        print(f"  Rango {i}: {start} - {end}")

    # Recortar el DataFrame basado en los rangos fusionados
    clipped_df = pd.concat([df.loc[start:end] for start, end in merged_ranges])
    original_size = len(df)
    clipped_size = len(clipped_df)
    print(f"Recorte completado. Tamaño original: {original_size} filas, Tamaño recortado: {clipped_size} filas.")
    return clipped_df
#===============FUNCIONES  ARRIBA ===================
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
            csv_path = os.path.join(csv_patient_path, file.replace(".edf", "_clipped.csv"))
            
            print(f"Processing {file} in {patient_folder}...")
            
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                print(f"Duración del archivo EDF: {raw.times[-1]} segundos")
                lista_canales = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
                raw.pick(lista_canales)
                #raw.filter(0.5, 70, method='iir')  # Filtro paso banda
                #raw.notch_filter(60, method='iir')  # Filtro notch
                raw.resample(100)  # Remuestreo a 100 Hz
                #ica = ICA(n_components=len(raw.ch_names), method='fastica', random_state=0, max_iter=500) #Filtro ICA con fastica
                #ica.fit(raw)
                #raw = ica.apply(raw)
                #print("ICA de MNE aplicado correctamente.")
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
                if df["Time (s)"].max() >= (1800*2 + time_diff):
                    df = clipping(df, freq_hz=100, minutes=30)
                else:
                    print("No se aplica clipping por duración insuficiente.")
                df.fillna(0, inplace=True)
                df.to_csv(csv_path, index=False)

                print(f"Saved: {csv_path}")

            except Exception as e:
                print(f"Error processing {file} in {patient_folder}: {e}")
print("Processing completed.")