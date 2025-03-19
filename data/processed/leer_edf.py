import mne
import os
# Ruta del archivo EDF
file_path = os.path.join("data", "raw", "siena-scalp-eeg-database-1.0.0", "PN00", "PN00-1.edf")

# Cargar el archivo EDF
raw = mne.io.read_raw_edf(file_path, preload=True)

# Mostrar información del archivo
print(raw.info)

# Obtener los nombres de los canales
print("Canales:", raw.ch_names)

# Obtener los datos de EEG y los tiempos
data, times = raw.get_data(return_times=True)

# Mostrar forma de los datos (n_canales, n_muestras)
print("Forma de los datos:", data.shape)

# Mostrar los primeros 5 valores de cada canal
print("Datos de EEG:", data[:, :5])
print("Tiempos:", times[:5])

"""
Lo que debe salir es:
La cabecera del archivo EDF nos da información sobre la grabación:

Canales (ch_names): Hay 35 canales, como EEG Fp1, EEG F3, EEG C3, etc. Estos representan diferentes electrodos colocados en el cuero cabelludo.
Número de canales (nchan): 35, lo que significa que hay 35 electrodos midiendo actividad eléctrica.
Frecuencia de muestreo (sfreq): 512 Hz, lo que indica que se registran 512 muestras por segundo para cada canal.
Fecha de la medición (meas_date): 1 de enero de 2016.
Filtros (highpass y lowpass): La señal no tiene filtro de paso alto (0.0 Hz), pero tiene un filtro de paso bajo en 256.0 Hz
Forma de los datos: (X columnas, Y filas), donde X es el número de canales y Y es el número de muestras.
Datos de EEG: Los primeros 5 valores de cada canal, que son los valores de voltaje en microvoltios.
Tiempos: Los primeros 5 tiempos de la señal en segundos.
"""

