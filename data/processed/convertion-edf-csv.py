import mne  # Para leer archivos EDF
import pandas as pd
import os  # Para manejar rutas
import re  # Para procesar el archivo de texto
import numpy as np  # Para operaciones numéricas

# Rutas de entrada y salida
file_path_read = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0", "PN00")
file_path_write = os.path.join("data", "raw", "csv-data")

os.makedirs(file_path_write, exist_ok=True)

def time_to_seconds(time_str):
    """Convierte un tiempo en formato h.m.s a segundos."""
    h, m, s = map(int, time_str.split('.'))
    return h * 3600 + m * 60 + s

# Leer archivo de texto con información de las convulsiones
txt_file = next((f for f in os.listdir(file_path_read) if f.endswith(".txt")), None)
seizures = {}

if txt_file:
    txt_path = os.path.join(file_path_read, txt_file)

    with open(txt_path, "r") as f:
        content = f.read()

    # Extraer la información de cada EDF
    matches = re.findall(
        r"File name:\s*(PN\d+-\d+\.edf).*?"
        r"Registration start time:\s*(\d+\.\d+\.\d+).*?"
        r"Registration end time:\s*(\d+\.\d+\.\d+).*?"
        r"Seizure start time:\s*(\d+\.\d+\.\d+).*?"
        r"Seizure end time:\s*(\d+\.\d+\.\d+)", 
        content, re.S
    )

    for match in matches:
        file_name, reg_start, reg_end, seiz_start, seiz_end = match
        reg_start_sec = time_to_seconds(reg_start.strip())
        seiz_start_sec = time_to_seconds(seiz_start.strip()) - reg_start_sec
        seiz_end_sec = time_to_seconds(seiz_end.strip()) - reg_start_sec
        
        seizures[file_name.strip()] = {
            "start_time": seiz_start_sec,
            "end_time": seiz_end_sec
        }

print(f"Seizures detected for {len(seizures)} files.")

# Procesar los archivos EDF
for file in os.listdir(file_path_read):
    if file.endswith(".edf"):
        edf_path = os.path.join(file_path_read, file)
        csv_path = os.path.join(file_path_write, file.replace(".edf", ".csv"))

        print(f"Processing {file}...")

        try:
            # Cargar el archivo EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            data, times = raw.get_data(return_times=True)
            df = pd.DataFrame(data.T, columns=raw.ch_names)

            # Insertar columna de tiempo
            df.insert(0, "Time (s)", times)

            # Insertar columna de Seizure con 0 por defecto
            seizure_flag = np.zeros(len(times), dtype=int)

            if file in seizures:
                seizure_info = seizures[file]
                seiz_start = seizure_info["start_time"]
                seiz_end = seizure_info["end_time"]

                # Marcar los tiempos dentro del rango de la convulsión
                seizure_flag[(times >= seiz_start) & (times <= seiz_end)] = 1

            df.insert(1, "Seizure", seizure_flag)
            df.to_csv(csv_path, index=False)

            print(f"Saved: {csv_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("Conversion completed.")
