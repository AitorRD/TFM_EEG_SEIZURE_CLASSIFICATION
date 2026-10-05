import mne  # Para leer archivos EDF
import pandas as pd
import os  # Para manejar rutas
import re  # Para procesar el archivo de texto
import numpy as np  # Para operaciones numéricas
import tqdm  # Para mostrar la barra de progreso
from mne.preprocessing import ICA

# v2: mismo pipeline que convertion-edf-csv.py pero para CHB-MIT (physionet.org)
file_path_read = os.path.join("data", "raw", "physionet.org", "files", "chbmit", "1.0.0")
file_path_write = os.path.join("data", "raw", "csv-data-v2")
os.makedirs(file_path_write, exist_ok=True)

# Canales que no son EEG de cuero cabelludo y deben excluirse antes del ICA
NON_EEG_SUBSTRINGS = ("ECG", "EKG", "VNS", "LOC", "ROC", "RAE", "LUE")

# Varios pacientes de CHB-MIT declaran canales "-" o "." como separadores de
# grupo de amplificador sin señal real (std ~0.005, prácticamente ruido de
# canal desconectado). MNE los renombra al duplicarse como "--0".."--4" o
# ".-0".."-4"; hay que excluirlos o contaminan el ICA.
PLACEHOLDER_CHANNEL_RE = re.compile(r'^(-|\.)(-\d+)?$')

# A diferencia de Siena, la mayoría de archivos de CHB-MIT (~80%) no tienen
# ninguna crisis y muchos son sesiones de 2-4h+ sin recortar. Sin límite,
# esas horas de interictal dominan el nº de ventanas/features en el LOSO
# posterior. Se recorta a los primeros N minutos, igual que el contexto que
# ya se usa alrededor de cada crisis.
INTERICTAL_MAX_MINUTES = 30


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


def parse_summary(txt_path):
    """Parsea un chbNN-summary.txt y devuelve {file_name.edf: [{start_time, end_time}, ...]}.

    Los tiempos de crisis en CHB-MIT ya vienen en segundos relativos al inicio
    del propio archivo (a diferencia de Siena, que usa horas de reloj), así
    que no hace falta restar contra ninguna hora de registro.
    """
    with open(txt_path, "r") as f:
        content = f.read()

    seizures = {}
    # Un bloque por cada "File Name:" hasta el siguiente (o fin de fichero)
    blocks = re.split(r'(?=File Name:)', content)

    for block in blocks:
        name_match = re.search(r'File Name:\s*(\S+\.edf)', block)
        if not name_match:
            continue
        file_name = name_match.group(1)

        # Cubre tanto "Seizure 1 Start Time:" (chb01-chb23) como
        # "Seizure Start Time:" sin numerar (chb24)
        starts = re.findall(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds', block)
        ends = re.findall(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds', block)

        if starts and ends:
            seizures[file_name] = [
                {"start_time": float(s), "end_time": float(e)}
                for s, e in zip(starts, ends)
            ]

    return seizures


# ===============FUNCIONES ARRIBA===================
# Iterar sobre cada carpeta de paciente/sesión (chb01, chb02, ...)
for patient_folder in tqdm.tqdm(sorted(os.listdir(file_path_read)), desc="Procesando pacientes"):
    patient_path = os.path.join(file_path_read, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    # Buscar el resumen de anotaciones del paciente (chbNN-summary.txt)
    txt_file = next((f for f in os.listdir(patient_path) if f.endswith(".txt")), None)

    seizures = {}
    if txt_file:
        seizures = parse_summary(os.path.join(patient_path, txt_file))

    print(f"\nSeizures detected for {patient_folder}: {sum(len(v) for v in seizures.values())}.")
    print(f"Seizures: {seizures}")

    # Procesar los archivos EDF dentro de la carpeta del paciente
    for file in sorted(os.listdir(patient_path)):
        if file.endswith(".edf"):
            edf_path = os.path.join(patient_path, file)
            csv_patient_path = os.path.join(file_path_write, patient_folder)
            os.makedirs(csv_patient_path, exist_ok=True)
            csv_path = os.path.join(csv_patient_path, file.replace(".edf", "_clipped.csv"))

            print(f"Processing {file} in {patient_folder}...")

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                print(f"Duración del archivo EDF: {raw.times[-1]} segundos")

                lista_canales = [
                    ch for ch in raw.ch_names
                    if not any(tag in ch.upper() for tag in NON_EEG_SUBSTRINGS)
                    and not PLACEHOLDER_CHANNEL_RE.match(ch)
                ]
                raw.pick(lista_canales)
                raw.filter(0.5, 70, method='iir')  # Filtro paso banda
                raw.notch_filter(60, method='iir')  # Filtro notch
                ica = ICA(n_components=len(raw.ch_names), method='fastica', random_state=0, max_iter=500)
                ica.fit(raw)
                raw = ica.apply(raw)
                print("ICA de MNE aplicado correctamente.")
                raw.resample(100)
                data, times = raw.get_data(return_times=True)
                df = pd.DataFrame(data.T, columns=raw.ch_names)
                df.insert(0, "Time (s)", times)
                seizure_flag = np.zeros(len(times), dtype=int)

                file_seizures = seizures.get(file, [])
                for seizure_info in file_seizures:
                    seiz_start = seizure_info["start_time"]
                    seiz_end = seizure_info["end_time"]
                    mask = (times >= seiz_start) & (times <= seiz_end)
                    seizure_flag[mask] = 1

                df.insert(1, "Seizure", seizure_flag)
                df.insert(2, "idPatient", patient_folder)
                df.insert(3, "idSession", file.replace(".edf", ""))

                if file_seizures:
                    max_time_diff = max(s["end_time"] - s["start_time"] for s in file_seizures)
                    if df["Time (s)"].max() >= (1800 * 2 + max_time_diff):
                        df = clipping(df, freq_hz=100, minutes=30)
                    else:
                        print("No se aplica clipping por duración insuficiente.")
                else:
                    max_rows = INTERICTAL_MAX_MINUTES * 60 * 100  # 100 Hz tras el resample
                    if len(df) > max_rows:
                        print(
                            f"  Interictal sin crisis: recortando a los primeros "
                            f"{INTERICTAL_MAX_MINUTES} min ({len(df)} -> {max_rows} filas)"
                        )
                        df = df.iloc[:max_rows]

                df.fillna(0, inplace=True)
                df.to_csv(csv_path, index=False)

                print(f"Saved: {csv_path}")

            except Exception as e:
                print(f"Error processing {file} in {patient_folder}: {e}")

print("Processing completed.")
