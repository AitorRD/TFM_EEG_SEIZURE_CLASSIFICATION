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

# --- Configuración de rutas ---
# Asegúrate de que estas rutas coincidan con la ubicación de tus archivos tsfresh
FEATURES_TRAIN_CSV = "data/processed/features_train.csv"
LABELS_TRAIN_CSV = "data/processed/labels_train.csv"
FEATURES_TEST_CSV = "data/processed/features_test.csv"
LABELS_TEST_CSV = "data/processed/labels_test.csv"

# Directorios de salida para los gráficos XAI y métricas
XAI_OUTPUT_DIR = "images/xai/"  # Subdirectorio específico para XAI de RNN
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)
METRICS_OUTPUT_DIR = "images/results/" # Directorio para métricas generales
os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)


# --- 1. Definición del Modelo RNN Simple ---
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        """
        Clasificador RNN simple para datos tabulares.
        input_size: En nuestro caso, será 1, ya que cada característica de tsfresh es un "paso de tiempo".
        hidden_size: Número de características en el estado oculto.
        num_layers: Número de capas recurrentes.
        num_classes: Número de clases de salida.
        """
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # nn.RNN: (input_size, hidden_size, num_layers, batch_first=True)
        # batch_first=True indica que el tensor de entrada/salida es (batch_size, seq_len, feature_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Capa completamente conectada para clasificar el estado oculto final
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Paso hacia adelante de la RNN.
        x: Tensor de entrada con forma (batch_size, num_features_tsfresh).
           Será remodelado a (batch_size, num_features_tsfresh, 1) para la RNN.
        """
        # x.shape: (batch_size, num_features_tsfresh)
        # Necesitamos remodelar a (batch_size, seq_len, input_size)
        # Aquí, seq_len = num_features_tsfresh, input_size = 1
        x = x.unsqueeze(2) # Transforma (batch_size, N) a (batch_size, N, 1)

        # Inicializar el estado oculto con ceros
        # h0.shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Propagar la RNN
        # out: Salidas de cada paso de tiempo: (batch_size, seq_len, hidden_size)
        # h_n: Estado oculto final para cada capa: (num_layers, batch_size, hidden_size)
        out, h_n = self.rnn(x, h0)
        
        # Para la clasificación, usamos el estado oculto de la última capa (h_n[-1])
        # h_n[-1, :, :] tiene forma (batch_size, hidden_size)
        out = self.fc(h_n[-1, :, :])
        return out

# --- 2. Preparación de Datos (Dataset y DataLoader) ---
class EEGFeaturesDataset(Dataset):
    def __init__(self, X, y):
        """
        Conjunto de datos para las características de EEG.
        X: DataFrame de Pandas con las características.
        y: Serie de Pandas con las etiquetas.
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        """Devuelve el número total de muestras."""
        return len(self.X)

    def __getitem__(self, idx):
        """Devuelve una muestra y su etiqueta en el índice dado."""
        return self.X[idx], self.y[idx]

# --- 3. Función para Entrenar y Evaluar el RNN ---
def train_and_evaluate_rnn(X_train, y_train, X_test, y_test,
                           learning_rate=0.001, num_epochs=50, batch_size=64,
                           hidden_size=64, num_layers=1, random_state=42):
    """
    Entrena y evalúa un modelo RNN.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo para RNN: {device}")

    # Configurar semilla para reproducibilidad
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Escalado de características (Importante para redes neuronales)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Preparar DataLoaders
    train_dataset = EEGFeaturesDataset(X_train_scaled, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EEGFeaturesDataset(X_test_scaled, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inicializar el modelo RNN
    # input_size es 1 porque cada característica de tsfresh es un "paso de tiempo"
    model = RNNClassifier(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Calcular pesos de clase para manejar desbalance de clases
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    # Entrenamiento
    print("\n--- Entrenando RNN ---")
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
        # Puedes descomentar la siguiente línea para ver el progreso de la pérdida por época
        # print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")

    # Evaluación
    print("\n--- Evaluando RNN ---")
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_X, _ in test_dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1) # Probabilidades para XAI
            _, predicted = torch.max(outputs.data, 1) # Clases predichas
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(y_test, all_preds)
    precision = precision_score(y_test, all_preds)
    recall = recall_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds)
    f1_micro = f1_score(y_test, all_preds, average='micro')
    f1_macro = f1_score(y_test, all_preds, average='macro')
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
    print("\nMétricas del RNN en Test:")
    for metric, value in metrics.items():
        print(f" - {metric}: {value:.4f}")
    
    return model, scaler, X_test_scaled, y_test, metrics

# --- 4. Funciones para XAI (SHAP y LIME) ---

def generate_xai_barplots_rnn(model, scaler, X_test_scaled, X_train_scaled, feature_names, save_dir, top_n=10):
    
    device = next(model.parameters()).device 

    # SHAP (KernelExplainer)
    try:
        # Usar una muestra más pequeña del conjunto de entrenamiento escalado como fondo para SHAP
        background_data_for_shap = X_train_scaled.values[:min(50, len(X_train_scaled))] # Convertir a numpy
        
        # Wrap the model's predict_proba for SHAP
        def model_predict_proba_for_shap(x):
            # Asegurar que la entrada x sea 2D (batch_size, num_features) para el modelo
            if x.ndim == 1:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device) 
            else: 
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device) 
            
            with torch.no_grad():
                model.eval()
                outputs = model(x_tensor)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[WARN] NaN o Inf detectado en las salidas del modelo antes de softmax para SHAP. Outputs: {outputs.cpu().numpy().flatten()[:5]}...")
                    outputs = torch.where(torch.isnan(outputs), torch.tensor(0.0).to(device), outputs)
                    outputs = torch.where(torch.isinf(outputs), torch.tensor(1e-6).to(device), outputs)
                    
                probabilities = F.softmax(outputs, dim=1)
                
                epsilon = 1e-8
                probabilities = torch.clamp(probabilities, epsilon, 1 - epsilon)

                return probabilities.cpu().numpy()

        explainer_shap = shap.KernelExplainer(model_predict_proba_for_shap, background_data_for_shap)
        
        test_sample_for_shap = X_test_scaled.values[:min(100, len(X_test_scaled))]

        print("Cálculo de valores SHAP completado.")
        shap_values_raw = explainer_shap.shap_values(test_sample_for_shap)
        if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
            shap_values_for_plotting = shap_values_raw[1] # Select SHAP values for class 1 (Seizure)
        else:
            shap_values_for_plotting = shap_values_raw 

        # --- NUEVA CORRECCIÓN PARA SHAP ---
        # Reshape shap_values_for_plotting para que sea 2D (num_samples, num_total_features)
        # Esto es crucial si SHAP lo devuelve como (samples, sub_dimension_1, sub_dimension_2)
        # Aseguramos que la última dimensión de feature_names coincida con la segunda dimensión de shap_values_for_plotting
        target_num_features = len(feature_names)
        current_shape = shap_values_for_plotting.shape

        # If it's 3D and the total number of elements matches expected features * samples
        if shap_values_for_plotting.ndim == 3:
        # Assumes the structure is (samples, time_points, channels_or_some_other_dim)
        # And you want to flatten these into a single dimension per sample
        # The total number of features after flattening should be len(feature_names)

        # First, check if the total number of features (excluding samples) matches feature_names length
            if np.prod(current_shape[1:]) == target_num_features:
                shap_values_for_plotting = shap_values_for_plotting.reshape(current_shape[0], target_num_features)
                print(f"SHAP values reshaped from {current_shape} to {shap_values_for_plotting.shape}")
            else:
                if not isinstance(shap_values_raw, list) and shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2:
                    shap_values_for_plotting = shap_values_raw[:, :, 1] # Take only the class 1 SHAP values
                    print(f"Reshaped 3D SHAP values by slicing: {shap_values_for_plotting.shape}")
                else:
                    raise ValueError(f"Formato inesperado de shap_values_for_plotting: {shap_values_for_plotting.shape}. Se esperaba 2D (muestras, características) o un 3D (muestras, intermedio, clases) donde la última dimensión sea el número de clases.")


        if shap_values_for_plotting.ndim != 2 or shap_values_for_plotting.shape[1] != target_num_features:
            raise ValueError(f"Después del procesamiento, shap_values_for_plotting debe ser 2D con {target_num_features} características. Shape actual: {shap_values_for_plotting.shape}")


        avg_abs_shap_importances = np.abs(shap_values_for_plotting).mean(axis=0)
        
        shap_series = pd.Series(avg_abs_shap_importances, index=feature_names)
        shap_series = shap_series.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=shap_series.values, y=shap_series.index, palette="Greys") 
        plt.title(f"Importancia SHAP (Promedio Absoluto) - RNN")
        plt.xlabel("Importancia SHAP media (absoluta)")
        plt.ylabel("Característica")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"rnn_shap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[XAI] Gráfico SHAP guardado en: {plot_path}")
        
    except Exception as e:
        print(f"[ERROR] Fallo al generar gráfico SHAP para RNN: {e}")

    # LIME
    try:
        # Wrap the model's predict_proba for LIME
        def lime_predict_proba(x):
            if x.ndim == 1:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            with torch.no_grad():
                model.eval()
                outputs = model(x_tensor)
                
                # --- MODIFICADO: Mensajes de WARN más específicos ---
                if torch.isnan(outputs).any():
                    print(f"[WARN] NaN detectado en las salidas del modelo ANTES de softmax para LIME. Outputs: {outputs.cpu().numpy().flatten()[:5]}...") # Mostrar primeros 5 elementos
                    outputs = torch.where(torch.isnan(outputs), torch.tensor(0.0).to(device), outputs)
                if torch.isinf(outputs).any():
                    print(f"[WARN] Inf detectado en las salidas del modelo ANTES de softmax para LIME. Outputs: {outputs.cpu().numpy().flatten()[:5]}...") # Mostrar primeros 5 elementos
                    outputs = torch.where(torch.isinf(outputs), torch.tensor(1e-6).to(device), outputs)
                    
                # Si, después de manejar NaN/Inf, las salidas aún son muy extremas
                # O si queremos ser ultra-seguros, podemos clippear los logits antes del softmax
                # outputs = torch.clamp(outputs, min=-10.0, max=10.0) # Esto es un "hard clamp" a los logits. Prueba esto si los errores persisten.

                probabilities = F.softmax(outputs, dim=1)
                
                # Asegurar que las probabilidades no sean exactamente 0 o 1
                epsilon = 1e-8
                probabilities = torch.clamp(probabilities, epsilon, 1 - epsilon)
                
                return probabilities.cpu().numpy()

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled.values,
            feature_names=feature_names.tolist(),
            class_names=["No Seizure", "Seizure"],
            mode="classification",
            discretize_continuous=False
        )
        
        all_lime_importances = []
        num_lime_samples = min(50, len(X_test_scaled)) # Número de instancias del test set a explicar
        
        for idx in tqdm(range(num_lime_samples), desc="Generando explicaciones LIME para RNN"):
            data_row = X_test_scaled.iloc[idx].values
            
            try:
                # --- AUMENTAR NUM_SAMPLES para explain_instance (NUEVO) ---
                # Esto le da a LIME más datos para construir su modelo local y puede mejorar la estabilidad.
                exp = explainer_lime.explain_instance(
                    data_row,
                    lime_predict_proba,
                    num_features=len(feature_names),
                    top_labels=1,
                    num_samples=5000 # Aumentado de 1000 a 5000, un valor común para mayor robustez
                )
                
                lime_weights = dict(exp.as_list(label=1))
                current_lime_series = pd.Series(lime_weights)
                current_lime_series = current_lime_series.reindex(feature_names, fill_value=0)
                all_lime_importances.append(current_lime_series)
            except Exception as e_inner:
                print(f"[ERROR] Fallo al explicar instancia {idx} con LIME para RNN: {e_inner}")
                # En caso de fallo, podemos añadir una serie de ceros para no detener el procesamiento.
                # Esto hace que la media de LIME sea menos precisa para esa instancia, pero permite que el script continúe.
                all_lime_importances.append(pd.Series(0.0, index=feature_names)) # Añadir ceros
                continue 

        if not all_lime_importances:
            print(f"[ERROR] No se generaron explicaciones LIME válidas para RNN. Todas las instancias fallaron.")
            return

        avg_abs_lime_importances = pd.concat(all_lime_importances, axis=1).abs().mean(axis=1)
        lime_series = avg_abs_lime_importances.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=lime_series.values, y=lime_series.index, palette="Greys")
        plt.title(f"Importancia LIME (Promedio) - RNN")
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Característica")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"rnn_lime.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[XAI] Gráfico LIME guardado en: {plot_path}")

    except Exception as e:
        print(f"[ERROR] Fallo al generar gráfico LIME para RNN: {e}")

    
# --- Función Principal para Ejecutar el Módulo ---
def main():
    print("--- Cargando datos para el RNN y XAI ---")
    # Cargar las características y etiquetas generadas por tsfresh
    X_train_raw = pd.read_csv(FEATURES_TRAIN_CSV, index_col=0)
    y_train = pd.read_csv(LABELS_TRAIN_CSV, index_col=0).squeeze() # .squeeze() convierte a Serie si es un DataFrame de 1 columna
    X_test_raw = pd.read_csv(FEATURES_TEST_CSV, index_col=0).squeeze()
    y_test = pd.read_csv(LABELS_TEST_CSV, index_col=0).squeeze()

    # Seleccionar solo las características comunes si ya se hizo una selección previa
    # Este paso es importante si usaste un método de selección de características (e.g., de tsfresh)
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
    # Esto es crucial para SHAP/LIME y para la consistencia del modelo.
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Reindexar para asegurar el orden alfabético (o cualquier orden consistente)
    X_train = X_train.reindex(columns=sorted(X_train.columns))
    X_test = X_test.reindex(columns=sorted(X_test.columns))

    print(f"Shape de X_train final: {X_train.shape}")
    print(f"Shape de X_test final: {X_test.shape}")

    # Entrenar y evaluar el RNN
    # Se pasan los DataFrames originales, la función de entrenamiento los escala internamente.
    model, scaler, X_test_scaled, y_test_rnn, metrics = train_and_evaluate_rnn(X_train, y_train, X_test, y_test)

    # Guardar métricas en un PNG
    # Redondear las métricas a 2 decimales para una mejor visualización en la tabla
    rounded_metrics = {k: round(v, 2) for k, v in metrics.items()}
    metrics_df = pd.DataFrame([rounded_metrics])

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off') # Ocultar ejes
    
    # Formatear el texto de las celdas para asegurar que los números se muestren limpiamente
    cell_text = [[f"{value:.2f}" for value in row] for row in metrics_df.values]

    table = ax.table(cellText=cell_text, # Usar el texto de las celdas formateado
                     colLabels=metrics_df.columns, # Nombres de las columnas
                     loc='center', cellLoc='center') # Posición y alineación del texto en las celdas
    table.auto_set_font_size(False)
    table.set_fontsize(12) # Tamaño de fuente
    table.scale(1.2, 1.2) # Escalar la tabla
    plt.tight_layout() # Ajustar el diseño para que no haya superposiciones
    metrics_png_path = os.path.join(METRICS_OUTPUT_DIR, 'rnn_metrics.png')
    plt.savefig(metrics_png_path, bbox_inches='tight', dpi=300) # Guardar la imagen
    plt.close()
    print(f"[INFO] Métricas del RNN guardadas en: {metrics_png_path}")

    # Necesitamos X_train_scaled para el background de los explainers de XAI.
    # Se escala X_train_raw con el mismo scaler usado en el entrenamiento.
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

    # Generar gráficos XAI
    print("\n--- Generando gráficos XAI para el RNN ---")
    generate_xai_barplots_rnn(model, scaler, X_test_scaled, X_train_scaled, X_train.columns, XAI_OUTPUT_DIR)

    print("\nProceso de RNN y XAI completado.")


if __name__ == "__main__":
    main()