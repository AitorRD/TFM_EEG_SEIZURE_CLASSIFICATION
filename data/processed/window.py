import os
import pandas as pd
from tqdm import tqdm
import time

# Función de ventanado por sesión
def window_and_save(df, set_name, window_size=200, overlap=0.50, out_dir='data/processed/windowed'):
    os.makedirs(out_dir, exist_ok=True)
    step = int(window_size * (1 - overlap))
    all_windows = []
    sessions = df['idSession'].unique()
    
    print(f"\n[{set_name.upper()}] Ventanando {len(sessions)} sesiones...")
    for session_id in tqdm(sessions, desc=f"{set_name.upper()} sessions"):
        session_df = df[df['idSession'] == session_id].reset_index(drop=True)
        max_start = len(session_df) - window_size
        for start in range(0, max_start + 1, step):
            window = session_df.iloc[start:start+window_size].copy()
            window['window_id'] = f"{session_id}_{start}"
            all_windows.append(window)
    
    if all_windows:
        result = pd.concat(all_windows, ignore_index=True)
        result.to_csv(os.path.join(out_dir, f"dataset_windowed_{set_name}.csv"), index=False)
        print(f"[{set_name.upper()}] Guardado: {len(result)} filas")

# Rutas de entrada
input_dir = os.path.join("data", "processed", "dataset_clipped")
train_file = os.path.join(input_dir, "train.csv")
val_file = os.path.join(input_dir, "val.csv")
test_file = os.path.join(input_dir, "test.csv")

# Parámetros de ventanado
sampling_rate = 100
window_seconds = 10
window_size = sampling_rate * window_seconds
overlap = 0.25

# Temporizador
start_time = time.time()

# Leer datasets y aplicar ventanado
datasets = {
    "train": pd.read_csv(train_file),
    "val": pd.read_csv(val_file),
    "test": pd.read_csv(test_file),
}

for set_name, df in datasets.items():
    window_and_save(df, set_name, window_size=window_size, overlap=overlap)

elapsed = time.time() - start_time
print(f"\n Ventanado completado en {elapsed:.2f} segundos.")

