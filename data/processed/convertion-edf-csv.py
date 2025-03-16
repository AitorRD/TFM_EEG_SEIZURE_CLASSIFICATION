import mne
import pandas as pd
import os

file_path_read = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0", "PN00", "PN00-1.edf")
file_path_write = os.path.join("data", "processed", "eeg_pn001_data.csv")
raw = mne.io.read_raw_edf(file_path_read, preload=True)

# Obtener datos y tiempos
data, times = raw.get_data(return_times=True)

# Crear DataFrame
df = pd.DataFrame(data.T, columns=raw.ch_names)
df.insert(0, "Tiempo (s)", times)

# Guardar en CSV
df.to_csv(file_path_write, index=False)
print(df.head())
