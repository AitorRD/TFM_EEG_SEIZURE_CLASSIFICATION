import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join("data", "raw", "csv-data")
output_path = os.path.join("images", "seizure_timelines")
os.makedirs(output_path, exist_ok=True)

sampling_rate = 100  # Frecuencia de muestreo en Hz (100 muestras por segundo)

# Recorrer todos los subdirectorios y archivos CSV
for root, dirs, files in os.walk(file_path):
    for file in files:
        if file.endswith("clipped.csv"):
            full_csv_path = os.path.join(root, file)
            try:
                df = pd.read_csv(full_csv_path)

                if "Seizure" in df.columns:
                    # Crear columna de tiempo en minutos basado en índices y frecuencia
                    df["Time (min)"] = df.index / (sampling_rate * 60)

                    plt.figure(figsize=(12, 6))
                    plt.plot(df["Time (min)"], df["Seizure"], drawstyle='steps-post', color="red", linewidth=1)
                    plt.xlabel("Tiempo (minutos)")
                    plt.ylabel("Seizure (0 o 1)")
                    plt.title(f"Seizure Timeline - {file}")
                    plt.ylim(-0.1, 1.1)
                    plt.grid(True)

                    # Guardar imagen
                    image_name = f"{file.replace('.csv', '')}_timeline.png"
                    save_path = os.path.join(output_path, image_name)
                    plt.savefig(save_path)
                    plt.close()

                    # Imprimir duración total
                    total_minutes = len(df) / (sampling_rate * 60)
                    print(f"Guardado: {save_path} - Duración: {total_minutes:.2f} minutos")

                else:
                    print(f"Advertencia: 'Seizure' no está en {file}")
            except Exception as e:
                print(f"Error procesando {file}: {e}")



