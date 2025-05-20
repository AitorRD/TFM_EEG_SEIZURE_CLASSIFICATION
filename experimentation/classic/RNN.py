import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.utils.class_weight import compute_class_weight
import shap
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# -- Carga y preparación de datos --
file_path_train = os.path.join("data", "processed", "dataset_windowed_train.csv")
file_path_val = os.path.join("data", "processed", "dataset_windowed_val.csv")
file_path_test = os.path.join("data", "processed", "dataset_windowed_test.csv")

channels = ['EEG Fp1','EEG Fp2','EEG F7','EEG F3','EEG Fz','EEG F4','EEG F8',
            'EEG T3','EEG C3','EEG Cz','EEG C4','EEG T4','EEG T5','EEG P3','EEG Pz',
            'EEG P4','EEG T6','EEG O1','EEG O2']

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[DEBUG] Cargando {csv_path} - shape original: {df.shape}")
    grouped = df.groupby('window_id')
    
    X_windows = []
    y_windows = []
    
    for window_id, group in grouped:
        data_window = group[channels].to_numpy().T  # (channels, tiempo)
        label = group['Seizure'].mode()[0]
        X_windows.append(data_window)
        y_windows.append(label)
    
    X = np.array(X_windows)
    y = np.array(y_windows)
    print(f"[DEBUG] Ventanas: {X.shape[0]}, Canales: {X.shape[1]}, Tiempo: {X.shape[2]}")
    print(f"[DEBUG] Etiquetas únicas: {np.unique(y)}")
    return X, y

X_train, y_train = load_and_prepare(file_path_train)
X_val, y_val = load_and_prepare(file_path_val)
X_test, y_test = load_and_prepare(file_path_test)
df_train = pd.read_csv('data/processed/dataset_windowed_train.csv')
train_labels = y_train

X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / (X_train.std(axis=2, keepdims=True) + 1e-8)
X_val = (X_val - X_val.mean(axis=2, keepdims=True)) / (X_val.std(axis=2, keepdims=True) + 1e-8)
X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / (X_test.std(axis=2, keepdims=True) + 1e-8)

# Dataset personalizado
class EEGDataset(Dataset):
    def __init__(self, X_windows, y_windows):
        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.y = torch.tensor(y_windows, dtype=torch.long)
        print(f"[DEBUG][Dataset] X shape: {self.X.shape}, y shape: {self.y.shape}")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, canales, tiempo)
        y = self.y[idx]
        return x, y

train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

class EEGRNN(nn.Module):
    def __init__(self, num_channels, time_points, num_classes, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_channels,   # features (canales)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 1, channels, time) → necesitamos (batch, time, channels)
        x = x.squeeze(1).permute(0, 2, 1)  # ahora (batch, time, channels)
        output, (hn, cn) = self.lstm(x)    # output: (batch, time, hidden)
        out = self.fc(hn[-1])              # usamos la última capa oculta
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGRNN(num_channels=19, time_points=X_train.shape[2], num_classes=2).to(device)
labels = train_labels
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
class_weights[1] *= 2
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"[DEBUG][Train] Batch {i}, Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(y_batch.numpy())
            if i % 10 == 0:
                print(f"[DEBUG][Eval] Batch {i}, Pred sample: {predicted[0].item()}, True sample: {y_batch[0].item()}")
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds,average='weighted', zero_division=0)
    rec = recall_score(trues, preds,average='weighted', zero_division=0)
    f1 = f1_score(trues, preds,average='weighted',  zero_division=0)
    print(f"[DEBUG][Eval] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1, preds, trues

# Entrenamiento con debug
history = {"acc": [], "prec": [], "rec": [], "f1": [], "loss": []}

for epoch in range(5): #<-------------------------------------------- EPOCAS !!!!!!!
    
    print(f"\n[INFO] Epoch {epoch+1}")
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_acc, val_prec, val_rec, val_f1, preds, trues = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")
    history["loss"].append(train_loss)
    history["acc"].append(val_acc)
    history["prec"].append(val_prec)
    history["rec"].append(val_rec)
    history["f1"].append(val_f1)
    print("Distribución real:", np.bincount(trues))
    print("Distribución predicha:", np.bincount(preds))
    print(classification_report(trues, preds, target_names=['No Seizure', 'Seizure']))
metrics_df = pd.DataFrame(history)
metrics_df.index = [f"Epoch {i+1}" for i in range(len(metrics_df))]
print("\nTabla de métricas por época:")
print(metrics_df)


# Gráfico comparativo de métricas
fig, ax = plt.subplots(figsize=(8, 2 + 0.5*len(metrics_df)))
ax.axis('off')
tbl = ax.table(cellText=np.round(metrics_df.values, 4),
               colLabels=metrics_df.columns,
               rowLabels=metrics_df.index,
               loc='center',
               cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.tight_layout()
plt.savefig("images/results/rnn_metricas_comparativas.png", dpi=300)
plt.close()
print("[INFO] Gráfico de métricas guardado en images/results/rnn_metricas_comparativas.png")

# LIME para explicar
time_points = X_test.shape[2]
X_test_flat = X_test.reshape(X_test.shape[0], -1)

def predict_fn(x_numpy):
    x_tensor = torch.tensor(x_numpy.reshape(-1, 1, 19, time_points), dtype=torch.float32).to('cpu')
    model_cpu = model.to('cpu')
    model_cpu.eval()
    with torch.no_grad():
        outputs = model_cpu(x_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

torch.cuda.empty_cache()
explainer = lime_tabular.LimeTabularExplainer(
    X_test_flat,
    feature_names=[f"C{c}_T{t}" for c in range(19) for t in range(time_points)],
    class_names=['No Seizure', 'Seizure'],
    discretize_continuous=False
)

exp = explainer.explain_instance(
    X_test_flat[0],
    predict_fn,
    num_features=50,
    top_labels=1
)

main_label = exp.available_labels()[0]
feature_importance = dict(exp.as_list(label=main_label))
lime_series = pd.Series(feature_importance).abs().sort_values(ascending=False)
lime_importancia = np.zeros(19)
for i, canal in enumerate(channels):
    canal_feats = [f"C{i}_T{t}" for t in range(time_points) if f"C{i}_T{t}" in lime_series.index]
    if canal_feats:
        lime_importancia[i] = lime_series[canal_feats].mean()
    else:
        lime_importancia[i] = 0  # Si no hay features, importancia 0
lime_canal_series = pd.Series(lime_importancia, index=channels).sort_values(ascending=False)


plt.figure(figsize=(8, 6))
print("LIME canal series:\n", lime_canal_series)
sns.barplot(x=lime_canal_series.values, y=lime_canal_series.index, palette="magma")
plt.title("Importancia LIME - RNN")
plt.xlabel("Importancia absoluta")
plt.tight_layout()
plt.savefig("images/xai/rnn_lime.png", dpi=300)
plt.close()
print("[INFO] Gráfico LIME guardado en images/xai/rnn_lime.png")

# SHAP para explicar
try:
    model_cpu = model.to('cpu')
    model_cpu.train()  # ← Esto es clave para evitar el error de cudnn
    background = torch.tensor(X_test[:50], dtype=torch.float32).unsqueeze(1)
    test_sample = torch.tensor(X_test[0:1], dtype=torch.float32).unsqueeze(1)

    explainer = shap.GradientExplainer(model_cpu, background)
    shap_values = explainer.shap_values(test_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)  # (1, 1, 19, tiempo)
    if shap_values.ndim == 4:
        shap_values = shap_values[0, 0]  # (19, tiempo)
        shap_importancia = np.abs(shap_values).mean(axis=1)
        shap_canal_series = pd.Series(shap_importancia, index=channels).sort_values(ascending=False)

        plt.figure(figsize=(8, 6))
        print("SHAP canal series:\n", shap_canal_series)
        sns.barplot(x=shap_canal_series.values, y=shap_canal_series.index, palette="viridis")
        plt.title("Importancia SHAP - RNN")
        plt.xlabel("Importancia media absoluta")
        plt.tight_layout()
        plt.savefig("images/xai/rnn_shap.png", dpi=300)
        plt.close()
        print("[INFO] Gráfico SHAP guardado en images/xai/rnn_shap.png")
    else:
        print(f"[ERROR] Dimensiones inesperadas de SHAP: {shap_values.shape}")
except Exception as e:
    print("[ERROR] al calcular SHAP:", str(e))