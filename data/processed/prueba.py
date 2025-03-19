import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Definir rutas
file_path = os.path.join("data", "raw", "csv-data", "PN00-4.csv")
output_path = os.path.join("data", "processed")

# Cargar CSV
df = pd.read_csv(file_path)

# 1. Gráfico de distribución de Seizures
plt.figure(figsize=(6, 4))
df["Seizure"].value_counts().plot(kind="bar", color=["blue", "red"])
plt.xticks(ticks=[0, 1], labels=["No Seizure (0)", "Seizure (1)"])
plt.xlabel("Seizure")
plt.ylabel("Count")
plt.title("Distribución de Seizures")
plt.savefig(os.path.join(output_path, "seizure_distribution.png"))
plt.close()

"""
# 2. Gráfico de frecuencias por canal y tiempo
plt.figure(figsize=(10, 6))
for channel in df.columns[2:]:  # Saltar 'Time (s)' y 'Seizure'
    plt.plot(df["Time (s)"], df[channel], alpha=0.5, label=channel)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Frecuencia de Señales EEG por Canal")
plt.legend(loc="upper right", ncol=3)
plt.savefig(os.path.join(output_path, "frequency_distribution.png"))
plt.close()

print("Gráficos guardados en:", output_path)"""
