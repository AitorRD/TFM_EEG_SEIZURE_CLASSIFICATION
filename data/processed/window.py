import pandas as pd
import os

from sklearn.model_selection import train_test_split

def window_data(df, window_size=2000, overlap=0.25, id_col='idSession'): #En la primera convulsion dura una 7000 filas
    step = int(window_size * (1 - overlap))
    windows = []

    print(f"Ventana: {window_size} | Overlap: {overlap} | Paso: {step}")
    for session_id, session_df in df.groupby(id_col):
        print(f"\nProcesando sesión: {session_id} (filas: {len(session_df)})")
        session_df = session_df.reset_index(drop=True)
        max_start = len(session_df) - window_size
        for i, start in enumerate(range(0, max_start + 1, step)):
            window = session_df.iloc[start:start+window_size].copy()
            window['window_id'] = f"{session_id}_{start}"
            window['idSession'] = session_id
            windows.append(window)
            if i < 2:
                print(f" - Ventana {i+1}: filas {start} a {start+window_size}")
    
    return pd.concat(windows).reset_index(drop=True)

file_path_read = os.path.join("data", "processed", "dataset_clipped.csv")
path_saved_train = os.path.join("data", "processed", "dataset_windowed_train.csv")
path_saved_val = os.path.join("data", "processed", "dataset_windowed_val.csv")
path_saved_test = os.path.join("data", "processed", "dataset_windowed_test.csv")
df = pd.read_csv(file_path_read)
df_windowed = window_data(df, window_size=3000, overlap=0.25)
unique_sessions = df_windowed['idSession'].unique()
train_ids, temp_ids = train_test_split(unique_sessions, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_df = df_windowed[df_windowed['idSession'].isin(train_ids)]
val_df = df_windowed[df_windowed['idSession'].isin(val_ids)]
test_df = df_windowed[df_windowed['idSession'].isin(test_ids)]

train_df.to_csv(path_saved_train, index=False)
val_df.to_csv(path_saved_val, index=False)
test_df.to_csv(path_saved_test, index=False)