import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Configuración de rutas (Asegúrate de que coincidan con donde guardas tus datos tsfresh) ---
FEATURES_TRAIN_CSV = "data/processed/features_train.csv"
LABELS_TRAIN_CSV = "data/processed/labels_train.csv"
FEATURES_TEST_CSV = "data/processed/features_test.csv"
LABELS_TEST_CSV = "data/processed/labels_test.csv"
XAI_OUTPUT_DIR = "images/xai/"
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)

# --- 1. Definición del Modelo CNN (Igual que antes) ---
class CNN1DClassifier(nn.Module):
    def __init__(self, input_features_count, num_classes=2):
        super(CNN1DClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1_input_features = self._get_conv_output_size(input_features_count)

        self.fc1 = nn.Linear(self.fc1_input_features, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_features_count)
        # Necesitamos (batch_size, 1, input_features_count) para Conv1d
        x = x.unsqueeze(1) # Añade la dimensión de canal (1 canal)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Aplanar el tensor para la capa totalmente conectada
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _get_conv_output_size(self, input_length):
        # Calcula el tamaño de salida convolucional para el forward
        # Simula un paso hacia adelante con una entrada dummy
        dummy_input = torch.rand(1, 1, input_length) # Correctly creates (1, channel, length)
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

# --- 2. Preparación de Datos (Dataset y DataLoader) ---
class EEGFeaturesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 3. Función para Entrenar y Evaluar la CNN ---
def train_and_evaluate_cnn(X_train, y_train, X_test, y_test,
                           learning_rate=0.001, num_epochs=50, batch_size=64, random_state=42):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo para CNN: {device}")

    # Configurar semilla para reproducibilidad
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Escalado de características (Importante para CNNs)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Preparar DataLoaders
    train_dataset = EEGFeaturesDataset(X_train_scaled, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EEGFeaturesDataset(X_test_scaled, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inicializar el modelo
    input_features_count = X_train_scaled.shape[1]
    model = CNN1DClassifier(input_features_count=input_features_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Calcular pesos de clase para manejar desbalance de clases
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Entrenamiento
    print("\n--- Entrenando CNN ---")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")

    # Evaluación
    print("\n--- Evaluando CNN ---")
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_X, _ in test_dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(y_test, all_preds)
    precision = precision_score(y_test, all_preds)
    recall = recall_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds)
    f1_macro = f1_score(y_test, all_preds, average='macro')
    f1_micro = f1_score(y_test, all_preds, average='micro')
    roc_auc = roc_auc_score(y_test, np.array(all_probs)[:, 1])
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'F1 Macro': f1_macro,
        'F1 Micro': f1_micro,
        'AUC': roc_auc
    }
    print("\nMétricas de la CNN en Test:")
    for metric, value in metrics.items():
        print(f" - {metric}: {value:.4f}")
    
    return model, scaler, X_test_scaled, y_test, metrics

# --- 4. Funciones para XAI (SHAP y LIME) ---

def generate_xai_barplots_cnn(model, scaler, X_test_scaled, X_train_scaled, feature_names, save_dir, top_n=10):
    
    device = next(model.parameters()).device # Obtener el dispositivo actual del modelo

    # SHAP (KernelExplainer)
    try:
        # Usar una muestra más pequeña del conjunto de entrenamiento escalado como fondo para SHAP
        # Esto es crucial para la eficiencia con KernelExplainer
        background_data_for_shap = X_train_scaled.values[:min(50, len(X_train_scaled))] # Convertir a numpy
        
        # Wrap the model's predict_proba for SHAP
        def model_predict_proba(x):
            # x será un numpy array, puede ser (num_features,) para una sola instancia
            # o (batch_size, num_features) para un lote.
            # El modelo CNN1DClassifier espera (batch_size, 1, num_features).

            # Asegurarse de que x es al menos 2D
            if x.ndim == 1:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device) # (1, num_features)
            else: # x.ndim == 2
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device) # (batch_size, num_features)
            
            # El modelo CNN1DClassifier.forward() se encarga de unsqueeze(1) a (batch_size, 1, num_features)
            with torch.no_grad():
                model.eval() # Asegurarse de que el modelo está en modo evaluación
                outputs = model(x_tensor)
                probabilities = F.softmax(outputs, dim=1)
                return probabilities.cpu().numpy()

        explainer = shap.KernelExplainer(model_predict_proba, background_data_for_shap)
        
        test_sample_for_shap = X_test_scaled.values[:min(100, len(X_test_scaled))]
        shap_values_raw = explainer.shap_values(test_sample_for_shap)
        
        shap_values_for_plotting = None

        if isinstance(shap_values_raw, list):
            # This is the most common case for predict_proba: list of arrays, each (num_samples, num_features)
            if len(shap_values_raw) > 1:
                shap_values_for_plotting = shap_values_raw[1] # For the positive class (Seizure)
            else:
                print("[WARN] SHAP values list has only one element. Using the first one (index 0).")
                shap_values_for_plotting = shap_values_raw[0]
        else:
            # This handles cases where shap_values_raw itself might be a single array.
            # If it's (num_samples, num_features, num_classes), then select the class dimension.
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2:
                print("[INFO] SHAP values is a 3D array. Selecting values for class 1.")
                shap_values_for_plotting = shap_values_raw[:, :, 1] # Select values for the positive class (index 1)
            elif shap_values_raw.ndim == 2 and shap_values_raw.shape[1] == len(feature_names):
                print("[INFO] SHAP values is a 2D array. Assuming it's for a single class.")
                shap_values_for_plotting = shap_values_raw
            else:
                raise ValueError(f"Unexpected shape of shap_values_raw: {np.array(shap_values_raw).shape}")
        
        # Now, shap_values_for_plotting MUST be 2D: (num_samples, num_features)
        if shap_values_for_plotting is None:
            raise ValueError("Failed to determine shap_values for plotting.")

        print(f"Shape of shap_values_for_plotting (after selection): {shap_values_for_plotting.shape}")

        if shap_values_for_plotting.ndim == 2 and shap_values_for_plotting.shape[1] == len(feature_names):
            mean_abs_shap = np.abs(shap_values_for_plotting).mean(axis=0)
        else:
            raise ValueError(f"Final SHAP values array for plotting has unexpected shape: {shap_values_for_plotting.shape}. Expected (num_samples, num_features).")

        importances = pd.Series(mean_abs_shap, index=feature_names)
        importances = importances.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="Greys")
        plt.title(f"Importancia de características (SHAP) - CNN")
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Características")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"cnn_shap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[XAI] Gráfico SHAP guardado en: {plot_path}")

    except Exception as e:
        print(f"[ERROR] Fallo al generar gráfico SHAP para CNN: {e}")

   # --- LIME ---
    print("\n--- Generando explicaciones LIME para RNN ---")
    try:
        # Wrap the model's predict_proba for LIME
        def lime_predict_proba(x):
            device = next(model.parameters()).device 

            if x.ndim == 1:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            else: 
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            # --- NEW DEBUGGING: Check input for non-zero variance ---
            # If LIME sends constant inputs, model output might be constant too
            if x_tensor.std(dim=0).mean() < 1e-6: # Check if std is near zero across features
                print(f"[LIME DEBUG] WARNING: Very low variance in input x_tensor. Shape: {x_tensor.shape}, Sample (first 5 features): {x_tensor.cpu().numpy().flatten()[:5]}...")

            with torch.no_grad():
                model.eval() 
                outputs = model(x_tensor)
                
                if torch.isnan(outputs).any():
                    print(f"[LIME DEBUG] NaN detected in model outputs BEFORE softmax for LIME. Outputs: {outputs.cpu().numpy().flatten()[:5]}...")
                    outputs = torch.where(torch.isnan(outputs), torch.tensor(0.0).to(device), outputs)
                if torch.isinf(outputs).any():
                    print(f"[LIME DEBUG] Inf detected in model outputs BEFORE softmax for LIME. Outputs: {outputs.cpu().numpy().flatten()[:5]}...")
                    outputs = torch.where(torch.isinf(outputs), torch.tensor(1e6).to(device) * torch.sign(outputs), outputs) 
                    
                outputs = torch.clamp(outputs, min=-20.0, max=20.0)

                probabilities = F.softmax(outputs, dim=1)
                
                # --- NEW DEBUGGING: Check if probabilities are nearly constant ---
                # If probabilities are all the same, LIME can't find a linear relationship
                if probabilities.shape[0] > 1: # Only check if batch size is > 1
                    prob_std = probabilities.std(dim=0).mean()
                    if prob_std < 1e-6:
                        print(f"[LIME DEBUG] WARNING: Probabilities are almost constant for perturbed samples! Std: {prob_std:.6f}. Probs: {probabilities.cpu().numpy().flatten()[:4]}...")

                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    print(f"[LIME DEBUG] NaN/Inf detected in probabilities AFTER softmax. Probs: {probabilities.cpu().numpy().flatten()[:5]}...")
                if torch.min(probabilities) < 0 or torch.max(probabilities) > 1:
                    print(f"[LIME DEBUG] Probabilities out of [0,1] range. Min: {torch.min(probabilities):.4f}, Max: {torch.max(probabilities):.4f}")

                epsilon = 1e-8
                probabilities = torch.clamp(probabilities, epsilon, 1 - epsilon)
                
                return probabilities.cpu().numpy()

        # Check for constant features that might cause issues with LIME
        if (X_train_scaled.std(axis=0) == 0).any():
            constant_features = X_train_scaled.columns[X_train_scaled.std(axis=0) == 0].tolist()
            print(f"[WARNING] Se encontraron características constantes en X_train_scaled que pueden causar problemas con LIME: {constant_features}. LIME podría fallar si intenta perturbar estas.")


        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled.values, # Usar datos de entrenamiento escalados
            feature_names=feature_names.tolist(), # Convertir a lista si es un Index
            class_names=["No Seizure", "Seizure"],
            mode="classification",
            discretize_continuous=True, # <--- CAMBIO CRÍTICO: Establecer a True para mayor estabilidad
            # discretizer='quartile' # <--- Opcional: Prueba esto si True no funciona
        )
        
        all_lime_importances = []
        num_lime_samples = min(50, len(X_test_scaled)) # Usar 50 muestras o el tamaño del test set si es menor
        
        for idx in tqdm(range(num_lime_samples), desc="Generando explicaciones LIME"):
            data_row = X_test_scaled.iloc[idx].values # Obtener la fila como numpy array
            
            try:
                exp = explainer_lime.explain_instance(
                    data_row,
                    lime_predict_proba,
                    num_features=len(feature_names),
                    top_labels=1,
                    num_samples=5000 # <--- CAMBIO CRÍTICO: Restaurar un número alto de muestras para LIME
                )
                
                # LIME devuelve una lista de tuplas (feature_name, importance)
                lime_weights = dict(exp.as_list(label=1)) # Explicación para la clase positiva
                current_lime_series = pd.Series(lime_weights)
                current_lime_series = current_lime_series.reindex(feature_names, fill_value=0) # Asegurar todas las features
                all_lime_importances.append(current_lime_series)
            except Exception as e_inner:
                print(f"[ERROR] Fallo al explicar instancia {idx} con LIME: {e_inner}")
                # Si falla, añadir una serie de ceros para que el promedio no se rompa
                all_lime_importances.append(pd.Series(0.0, index=feature_names)) 
                continue # Continuar con la siguiente instancia si falla una

        # --- IMPORTANT: Check if all explanations were zeros or if list is empty ---
        if not all_lime_importances or all(s.sum() == 0 for s in all_lime_importances): 
            print(f"[ERROR] No se generaron explicaciones LIME válidas (o todas son cero) para CNN. Todas las instancias fallaron o no tuvieron impacto.")
            # Create an empty plot with a message to indicate no results
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No LIME explanations generated or all are zero.\nCheck console for LIME DEBUG messages.", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            plot_path = os.path.join(save_dir, f"rnn_lime.png") # Changed to rnn_lime.png
            plt.savefig(plot_path, dpi=300)
            plt.close()
            return # Exit the function if no valid explanations

        # Promediar las importancias absolutas de LIME
        avg_abs_lime_importances = pd.concat(all_lime_importances, axis=1).abs().mean(axis=1)
        lime_series = avg_abs_lime_importances.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=lime_series.values, y=lime_series.index, palette="Greys")
        plt.title(f"Importancia LIME (Promedio) - CNN") # Changed to CNN for consistency
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Características")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"rnn_lime.png") # Changed to rnn_lime.png
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[XAI] Gráfico LIME guardado en: {plot_path}")

    except Exception as e:
        print(f"[ERROR] Fallo al generar gráfico LIME para CNN: {e}")

    
# --- Función Principal para Ejecutar el Módulo ---
def main():
    print("--- Cargando datos para la CNN y XAI ---")
    X_train_raw = pd.read_csv(FEATURES_TRAIN_CSV, index_col=0)
    y_train = pd.read_csv(LABELS_TRAIN_CSV, index_col=0).squeeze()
    X_test_raw = pd.read_csv(FEATURES_TEST_CSV, index_col=0).squeeze()
    y_test = pd.read_csv(LABELS_TEST_CSV, index_col=0).squeeze()

    # Seleccionar solo las características comunes si ya se hizo una selección previa
    # Asume que `selected_features.csv` existe y tiene las columnas seleccionadas
    selected_features_path = "data/processed/selected_features.csv"
    if os.path.exists(selected_features_path):
        selected_columns_df = pd.read_csv(selected_features_path, header=None)
        selected_columns = selected_columns_df.iloc[:, 0].tolist()
        
        # Filtrar solo las columnas seleccionadas que existen en los DataFrames raw
        X_train = X_train_raw[[col for col in selected_columns if col in X_train_raw.columns]]
        X_test = X_test_raw[[col for col in selected_columns if col in X_test_raw.columns]]
        print(f"Usando {X_train.shape[1]} características seleccionadas de TSFRESH.")
    else:
        print("[ADVERTENCIA] No se encontró el archivo de características seleccionadas. Usando todas las características.")
        X_train = X_train_raw
        X_test = X_test_raw
    
    # Asegúrate de que las columnas sean idénticas y en el mismo orden para X_train y X_test
    # Esto es crucial para SHAP/LIME que dependen del orden de las características
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Reindexar para asegurar el orden
    X_train = X_train.reindex(columns=sorted(X_train.columns))
    X_test = X_test.reindex(columns=sorted(X_test.columns))

    print(f"Shape de X_train final: {X_train.shape}")
    print(f"Shape de X_test final: {X_test.shape}")

    # Entrenar y evaluar la CNN
    model, scaler, X_test_scaled, y_test_cnn, metrics = train_and_evaluate_cnn(X_train, y_train, X_test, y_test)

    # Guardar métricas en un PNG
    # Redondear las métricas a 2 decimales para una mejor visualización
    rounded_metrics = {k: round(v, 2) for k, v in metrics.items()}
    metrics_df = pd.DataFrame([rounded_metrics])

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    
    # Formatear el texto de las celdas para asegurar que los números se muestren limpiamente con 2 decimales
    cell_text = [[f"{value:.2f}" for value in row] for row in metrics_df.values]

    table = ax.table(cellText=cell_text, # Usar el texto de las celdas formateado
                     colLabels=metrics_df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    metrics_png_path = os.path.join("images","results", 'cnn_metrics.png')
    plt.savefig(metrics_png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Métricas de la CNN guardadas en: {metrics_png_path}")

    # Necesitamos X_train_scaled para el background de los explainers
    # Asegurarse de que X_train_scaled tiene las mismas columnas y orden que X_train
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

    # Generar gráficos XAI
    print("\n--- Generando gráficos XAI para la CNN ---")
    generate_xai_barplots_cnn(model, scaler, X_test_scaled, X_train_scaled, X_train.columns, XAI_OUTPUT_DIR)

    print("\nProceso de CNN y XAI completado.")
    
if __name__ == "__main__":
    main()