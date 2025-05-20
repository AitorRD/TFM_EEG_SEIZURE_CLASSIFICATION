import pandas as pd
import numpy as np
import torch
import os
from torch import device, nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
# ===================== DATASET =====================

class EEGWindowDataset(Dataset):
    def __init__(self, df, n_channels=19):
        self.data = []
        self.labels = []
        grouped = df.groupby("window_id")
        for window_id, group in grouped:
            X = group.iloc[:, 4:4 + n_channels].values  # EEG channels
            y = group["Seizure"].values[0]  # same label for all window rows
            if X.shape[0] == 3000:  # sanity check
                self.data.append(X)
                self.labels.append(y)
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ===================== MODEL =====================

class EEGTransformer(nn.Module):
    def __init__(self, input_dim=19, seq_len=3000, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.cls_head(x)   # (B, num_classes)
        return x

# ===================== TRAINING =====================

def train_model(model, train_loader, val_loader, epochs=5):
    class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Guardar métricas por época
# Guardar métricas por época
    history = {
        "epoch": [],
        "accuracy": [],
        "weighted_precision": [],
        "weighted_recall": [],
        "weighted_f1": [],
        "val_loss": [] 
    }

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation metrics
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        n_batches = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss_val = criterion(out, y)
                val_loss += loss_val.item()
                n_batches += 1
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        val_loss /= n_batches  # Promedio por batch

        acc = accuracy_score(all_labels, all_preds)
        weighted_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        weighted_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        history["epoch"].append(epoch + 1)
        history["accuracy"].append(acc)
        history["weighted_precision"].append(weighted_prec)
        history["weighted_recall"].append(weighted_rec)
        history["weighted_f1"].append(weighted_f1)
        history["val_loss"].append(val_loss)

        print(f"\nEpoch {epoch + 1}")
        print(f"Val Loss: {val_loss:.10f} | Accuracy: {acc:.10f} | Weighted Precision: {weighted_prec:.10f} | Weighted Recall: {weighted_rec:.10f} | Weighted F1: {weighted_f1:.10f}")

    # Mostrar tabla de métricas
    import pandas as pd
    metrics_df = pd.DataFrame(history)
    print("\nTabla de métricas por época:")
    print(metrics_df.to_string(index=False))

    # Guardar la tabla como imagen
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(metrics_df)))
    ax.axis('off')
    tbl = ax.table(cellText=np.round(metrics_df.values, 4),
                   colLabels=metrics_df.columns,
                   loc='center',
                   cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig("images/results/transformer_epoch_metrics.png", dpi=300)
    plt.close()
    print("[INFO] Tabla de métricas por época guardada en images/results/transformer_epoch_metrics.png")

# ===================== EVAL & PLOT =====================

def evaluate(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    report = classification_report(labels, preds, output_dict=True)
    df_metrics = pd.DataFrame(report).T.drop("accuracy")
    sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("images/results/transformer_confusion_matrix.png")

    # Save metrics to PNG
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(df_metrics.iloc[:-1, :-1], annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig("images/results/transformer_metrics.png")

# ===================== MAIN =====================

# Carga tus CSVs
file_path_train = os.path.join("data", "processed", "dataset_windowed_train.csv")
file_path_val = os.path.join("data", "processed", "dataset_windowed_val.csv")
file_path_test = os.path.join("data", "processed", "dataset_windowed_test.csv")
train_df = pd.read_csv(file_path_train)
val_df = pd.read_csv(file_path_val)
test_df = pd.read_csv(file_path_test)
# Crear datasets y loaders
train_dataset = EEGWindowDataset(train_df)
val_dataset = EEGWindowDataset(val_df)
test_dataset = EEGWindowDataset(test_df)
y_train = train_dataset.labels.numpy()
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Inicializar y entrenar
model = EEGTransformer()
train_model(model, train_loader, val_loader)
evaluate(model, test_loader)




