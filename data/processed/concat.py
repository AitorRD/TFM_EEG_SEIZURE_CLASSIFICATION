import os
import pandas as pd

root_dir = file_path_write = os.path.join("data", "raw", "csv-data")
path_saved = os.path.join("data", "processed", "dataset_clipped.csv")
all_data = []

# Recorremos carpetas y archivos
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('_clipped.csv'):
            file_path = os.path.join(subdir, file)
            try:
                df = pd.read_csv(file_path)
                print("Leyendo archivo:", file_path)
                if 'Time (s)' not in df.columns:
                    print(f"Saltando archivo sin 'Time (s)': {file_path}")
                    continue
                df['sesion'] = file
                df['paciente'] = os.path.basename(subdir)
                
                all_data.append(df)
            except Exception as e:
                print(f"Error al leer {file_path}: {e}")

final_df = pd.concat(all_data, ignore_index=True)

# Guardar en un solo archivo
final_df.to_csv(path_saved, index=False)