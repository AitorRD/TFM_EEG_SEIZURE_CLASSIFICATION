import os
import pandas as pd

root_dir = file_path_write = os.path.join("data", "raw", "csv-data")
path_saved = os.path.join("data", "processed", "dataset_clipped.csv")
channels = ["Time (s)", "Seizure", "idSession", "idPatient" ,"EEG Fp1", "EEG Fp2", "EEG F7", "EEG F3", "EEG Fz", "EEG F4", "EEG F8",
           "EEG T3", "EEG C3", "EEG Cz", "EEG C4", "EEG T4", "EEG T5", "EEG P3",
           "EEG Pz", "EEG P4", "EEG T6", "EEG O1", "EEG O2"]
all_data = []

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

                all_data.append(df)
            except Exception as e:
                print(f"Error al leer {file_path}: {e}")

print("Archivos leídos:", len(all_data))
print("\nConcatenando archivos...")
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv(path_saved, index=False)
print(f"CSV final guardado en: {path_saved}")
