import mne  # Para leer archivos EDF
import pandas as pd
import os  # Para manejar rutas
import re  # Para procesar el archivo de texto
import numpy as np  # Para operaciones numéricas

file_path_read = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0")
file_path_write = os.path.join("data", "raw", "csv-data")
os.makedirs(file_path_write, exist_ok=True)

def time_to_seconds(start_time_str, end_time_str=None):
    """Convierte una hora en formato HH:MM:SS a segundos.
       Si el tiempo final es menor que el tiempo de inicio, asume un cruce de día y ajusta.
    """
    h, m, s = map(int, re.split('[:.]', start_time_str))
    start_seconds = h * 3600 + m * 60 + s

    if end_time_str:
        h, m, s = map(int, re.split('[:.]', end_time_str))
        end_seconds = h * 3600 + m * 60 + s
        
        # Si el tiempo de fin es menor que el de inicio, el evento pasó al día siguiente
        if end_seconds < start_seconds:
            end_seconds += 24 * 3600  # Añadimos 24 horas en segundos

        return start_seconds, end_seconds
    else:
        return start_seconds

# Iterar sobre cada carpeta de paciente
for patient_folder in os.listdir(file_path_read):
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
        
        # Extraer la información de cada convulsión y asignarla a su archivo correspondiente
        #\d - Se refiere a un dígito, \s* - Espacio en blanco, [\.:] - Punto o dos puntos
        #\s* - Espacio en blanco, (?:...) - Agrupación sin captura, (.+?) - Captura de grupo no codicioso
        #\s - Espacio en blanco, \w - Caracteres alfanuméricos, \. - Punto, \- - Guion, ()? - Agrupación, () 
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
                
            reg_start_sec = time_to_seconds(reg_start.strip()) if reg_start else 0
            seiz_start_sec = time_to_seconds(seiz_start.strip()) - reg_start_sec
            seiz_end_sec = time_to_seconds(seiz_end.strip()) - reg_start_sec
                
            seizures[file_name].append({
                "start_time": seiz_start_sec,
                "end_time": seiz_end_sec
            })
                
        # Manejo especial para el paciente "PN12"
        if "PN12" in patient_folder:
            combined_seizures = {}
            for file_name, conv_list in seizures.items():
                first_seizure = conv_list[0]
                if first_seizure in combined_seizures:
                    combined_seizures[first_seizure].extend(conv_list)
                else:
                    combined_seizures[first_seizure] = conv_list
            seizures = combined_seizures
            
        seizure_count = len(re.findall(r"Seizure n\s*+", content))
        
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
                data, times = raw.get_data(return_times=True)
                df = pd.DataFrame(data.T, columns=raw.ch_names)
                df.insert(0, "Time (s)", times)
                seizure_flag = np.zeros(len(times), dtype=int)

                if file in seizures:
                    for seizure_info in seizures[file]:
                        seiz_start = seizure_info["start_time"]
                        seiz_end = seizure_info["end_time"]
                        seizure_flag[(times >= seiz_start) & (times <= seiz_end)] = 1

                df.insert(1, "Seizure", seizure_flag)
                df.to_csv(csv_path, index=False)

                print(f"Saved: {csv_path}")

            except Exception as e:
                print(f"Error processing {file} in {patient_folder}: {e}")

print("Conversion completed.")
