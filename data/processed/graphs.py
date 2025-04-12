import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join("data", "raw", "csv-data")
output_path = os.path.join("images", "seizure_distributions")
os.makedirs(output_path, exist_ok=True)

# Recorrer todos los subdirectorios y archivos CSV
for root, dirs, files in os.walk(file_path):
    for file in files:
        if file.endswith(".csv"):
            full_csv_path = os.path.join(root, file)
            try:
                df = pd.read_csv(full_csv_path)

                # Solo si la columna 'Seizure' está presente
                if "Seizure" in df.columns:
                    plt.figure(figsize=(6, 4))
                    df["Seizure"].value_counts().plot(kind="bar", color=["blue", "red"])
                    plt.xticks(ticks=[0, 1], labels=["No Seizure (0)", "Seizure (1)"])
                    plt.xlabel("Seizure")
                    plt.ylabel("Count")
                    plt.title(f"Distribución de Seizures - {file}")

                    # Nombre de imagen con ruta relativa a su carpeta
                    image_name = f"{file.replace('.csv', '')}_seizure_distribution.png"
                    save_path = os.path.join(output_path, image_name)
                    plt.savefig(save_path)
                    plt.close()
                    print(f"Guardado: {save_path}")
                else:
                    print(f"Advertencia: 'Seizure' no está en {file}")
            except Exception as e:
                print(f"Error procesando {file}: {e}")

