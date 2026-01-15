import os
import pandas as pd
from sklearn.model_selection import train_test_split

root_dir = os.path.join("data", "raw", "csv-data")
output_dir = os.path.join("data", "processed", "dataset_clipped")
os.makedirs(output_dir, exist_ok=True)

channels = ["Time (s)", "Seizure", "idSession", "idPatient",
            "EEG Fp1", "EEG Fp2", "EEG F7", "EEG F3", "EEG Fz", "EEG F4", "EEG F8",
            "EEG T3", "EEG C3", "EEG Cz", "EEG C4", "EEG T4", "EEG T5", "EEG P3",
            "EEG Pz", "EEG P4", "EEG T6", "EEG O1", "EEG O2"]

session_dfs = {}

# Leer y agrupar por idSession
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('_clipped.csv'):
            file_path = os.path.join(subdir, file)
            try:
                df = pd.read_csv(file_path)
                print("Leyendo archivo:", file_path)

                rename_dict = {}
                cols = df.columns
                if "EEG CZ" in cols:
                    rename_dict["EEG CZ"] = "EEG Cz"
                if "EEG FP2" in cols:
                    rename_dict["EEG FP2"] = "EEG Fp2"
                df.rename(columns=rename_dict, inplace=True)
                df.fillna(0, inplace=True)
                df = df[[col for col in channels if col in df.columns]]

                for session_id in df["idSession"].unique():
                    df_session = df[df["idSession"] == session_id]
                    session_dfs[session_id] = df_session

            except Exception as e:
                print(f"Error al leer {file_path}: {e}")

print(f"\nTotal sesiones encontradas: {len(session_dfs)}")

# Dividir sesiones en train/val/test
session_ids = list(session_dfs.keys())
train_ids, temp_ids = train_test_split(session_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Concatenar los datos por conjunto
train_df = pd.concat([session_dfs[sid] for sid in train_ids], ignore_index=True)
val_df = pd.concat([session_dfs[sid] for sid in val_ids], ignore_index=True)
test_df = pd.concat([session_dfs[sid] for sid in test_ids], ignore_index=True)

# Guardar a CSV
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print(f"\nDatos guardados en:\n- train.csv ({len(train_df)} filas)\n- val.csv ({len(val_df)} filas)\n- test.csv ({len(test_df)} filas)")

