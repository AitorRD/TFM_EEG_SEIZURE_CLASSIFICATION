"""
Ejemplo: Deep Learning con Optimización de Hiperparámetros y Validación Avanzada
Insertar en transformers.py, CNN.py, RNN.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

# ============ VALIDACIÓN CRUZADA PARA DEEP LEARNING ============

def k_fold_cross_validation(dataset, model_class, k_folds=5, epochs=10, batch_size=32, 
                             lr=1e-4, device='cuda', random_state=42):
    """
    Validación cruzada K-Fold para modelos de deep learning.
    
    DÓNDE INSERTAR: En transformers.py, después de definir EEGTransformer
    
    IMPORTANTE: Para datos temporales/sesiones EEG, usar GroupKFold o StratifiedGroupKFold
    para evitar data leakage (ventanas de la misma sesión en train y test).
    
    Args:
        dataset: Dataset completo (train + val combinado)
        model_class: Clase del modelo (EEGTransformer, CNN, RNN)
        k_folds: Número de folds
        epochs: Épocas por fold
        batch_size: Tamaño del batch
        lr: Learning rate
        device: 'cuda' o 'cpu'
    
    Returns:
        fold_results: Lista con métricas de cada fold
        avg_metrics: Promedios y desviaciones estándar
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight
    
    # Obtener todos los datos y labels
    all_data = dataset.data  # (N, T, C)
    all_labels = dataset.labels  # (N,)
    
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_results = []
    
    print(f"\n{'='*70}")
    print(f"  K-FOLD CROSS VALIDATION (K={k_folds})")
    print(f"{'='*70}\n")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data, all_labels), 1):
        print(f"\n--- FOLD {fold}/{k_folds} ---")
        
        # Dividir datos
        train_data = all_data[train_idx]
        train_labels = all_labels[train_idx]
        val_data = all_data[val_idx]
        val_labels = all_labels[val_idx]
        
        # Crear datasets y loaders
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Crear modelo
        model = model_class().to(device)
        
        # Class weights
        class_weights = compute_class_weight('balanced', 
                                             classes=np.array([0, 1]), 
                                             y=train_labels.numpy())
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Entrenar
        for epoch in range(epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        
        # Evaluar en validación del fold
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())
        
        # Métricas
        acc = accuracy_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds, average='weighted')
        
        fold_results.append({'fold': fold, 'accuracy': acc, 'f1': f1})
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Calcular promedios
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    std_f1 = np.std([r['f1'] for r in fold_results])
    
    print(f"\n{'='*70}")
    print(f"  RESULTADOS CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"  Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
    
    avg_metrics = {
        'avg_accuracy': avg_acc,
        'std_accuracy': std_acc,
        'avg_f1': avg_f1,
        'std_f1': std_f1
    }
    
    return fold_results, avg_metrics


# ============ OPTIMIZACIÓN CON OPTUNA ============

def objective(trial, train_dataset, val_dataset, model_class, n_epochs=10, device='cuda'):
    """
    Función objetivo para Optuna: sugiere hiperparámetros y devuelve métrica objetivo.
    
    DÓNDE INSERTAR: En transformers.py, antes de la función main
    
    Args:
        trial: Trial de Optuna
        train_dataset, val_dataset: Datasets de entrenamiento y validación
        model_class: Clase del modelo (EEGTransformer)
        n_epochs: Épocas para entrenar
        device: 'cuda' o 'cpu'
    
    Returns:
        val_f1: Métrica objetivo (F1 en validación)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Sugerir hiperparámetros
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    
    # Crear loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo con hiperparámetros sugeridos
    # Nota: Necesitas modificar EEGTransformer para aceptar dropout
    model = model_class(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    
    # Class weights
    train_labels = train_dataset.labels.numpy()
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Entrenar
    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Validación intermedia para pruning
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())
        
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        # Reportar métrica intermedia para pruning (detener trials malos temprano)
        trial.report(val_f1, epoch)
        
        # Chequear si el trial debe ser podado
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_f1


def optimize_hyperparameters_optuna(train_dataset, val_dataset, model_class, 
                                     n_trials=50, n_epochs=10, device='cuda', 
                                     study_name='transformer_optimization'):
    """
    Optimiza hiperparámetros usando Optuna (Bayesian Optimization).
    
    DÓNDE INSERTAR: En transformers.py, después de cargar datasets
    CUÁNDO USAR: Antes de entrenar el modelo final
    
    Args:
        train_dataset, val_dataset: Datasets
        model_class: Clase del modelo
        n_trials: Número de combinaciones de hiperparámetros a probar
        n_epochs: Épocas por trial
        device: 'cuda' o 'cpu'
        study_name: Nombre del estudio (para guardar/reanudar)
    
    Returns:
        best_params: Mejores hiperparámetros encontrados
        study: Objeto study de Optuna (con historial completo)
    """
    print(f"\n{'='*70}")
    print(f"  OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    print(f"  Trials: {n_trials} | Épocas por trial: {n_epochs}")
    print(f"{'='*70}\n")
    
    # Crear o cargar estudio
    study = optuna.create_study(
        direction='maximize',  # Maximizar F1
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # Optimizar
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, model_class, n_epochs, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Resultados
    print(f"\n{'='*70}")
    print(f"  MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print(f"{'='*70}")
    print(f"  Mejor F1 Score: {study.best_value:.4f}")
    print(f"  Trial #: {study.best_trial.number}")
    print(f"\n  Hiperparámetros:")
    for param, value in study.best_params.items():
        print(f"    {param}: {value}")
    
    # Guardar resultados
    import pandas as pd
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"data/processed/optuna_{study_name}.csv", index=False)
    print(f"\n  → Resultados guardados en: data/processed/optuna_{study_name}.csv\n")
    
    # Visualizar (opcional, requiere plotly)
    try:
        import optuna.visualization as vis
        import plotly
        
        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)
        fig3 = vis.plot_slice(study)
        
        fig1.write_html(f"images/results/optuna_history_{study_name}.html")
        fig2.write_html(f"images/results/optuna_importance_{study_name}.html")
        fig3.write_html(f"images/results/optuna_slice_{study_name}.html")
        print(f"  → Visualizaciones guardadas en images/results/\n")
    except ImportError:
        print("  (Instala plotly para generar visualizaciones)\n")
    
    return study.best_params, study


# ============ EARLY STOPPING Y LEARNING RATE SCHEDULING ============

class EarlyStopping:
    """
    Early Stopping para evitar overfitting.
    
    DÓNDE USAR: En el loop de entrenamiento de transformers.py
    """
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        """
        Args:
            patience: Épocas sin mejora antes de detener
            min_delta: Cambio mínimo para considerar mejora
            mode: 'max' para métricas a maximizar (F1), 'min' para loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
        elif self._is_improvement(current_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def _is_improvement(self, current_value):
        if self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            return current_value < self.best_value - self.min_delta


def train_with_enhancements(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda'):
    """
    Entrenamiento mejorado con Early Stopping y LR Scheduling.
    
    DÓNDE USAR: Reemplazar la función train_model() actual en transformers.py
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Obtener labels para class weights
    train_labels = []
    for _, y in train_loader:
        train_labels.extend(y.numpy())
    train_labels = np.array(train_labels)
    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001, mode='max')
    
    model.to(device)
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}
    
    print(f"\n{'='*70}")
    print(f"  ENTRENAMIENTO CON EARLY STOPPING Y LR SCHEDULING")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        # Update scheduler
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Guardar historial
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping check
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"\n⚠ Early Stopping activado en época {epoch+1}")
            break
    
    print(f"\n✓ Entrenamiento completado!")
    print(f"  Mejor F1 en validación: {early_stopping.best_value:.4f}\n")
    
    return history


# ============ EJEMPLO DE USO COMPLETO ============

if __name__ == "__main__":
    """
    INTEGRACIÓN EN transformers.py:
    
    1. VALIDACIÓN CRUZADA (antes de entrenar modelo final):
       - Combinar train + val datasets
       - Llamar k_fold_cross_validation() para evaluar estabilidad
    
    2. OPTIMIZACIÓN DE HIPERPARÁMETROS (opcional, computacionalmente costoso):
       - Llamar optimize_hyperparameters_optuna()
       - Usar mejores hiperparámetros para modelo final
    
    3. ENTRENAMIENTO FINAL:
       - Usar train_with_enhancements() en lugar de train_model()
       - Incluye Early Stopping y LR Scheduling automáticamente
    """
    
    # Ejemplo de flujo completo
    # 
    # # Cargar datasets
    # train_df = pd.read_csv("data/processed/windowed/dataset_windowed_train.csv")
    # val_df = pd.read_csv("data/processed/windowed/dataset_windowed_val.csv")
    # test_df = pd.read_csv("data/processed/windowed/dataset_windowed_test.csv")
    # 
    # train_dataset = EEGWindowDataset(train_df)
    # val_dataset = EEGWindowDataset(val_df)
    # test_dataset = EEGWindowDataset(test_df)
    # 
    # # Opción 1: Validación Cruzada (evaluar estabilidad)
    # combined_df = pd.concat([train_df, val_df])
    # combined_dataset = EEGWindowDataset(combined_df)
    # fold_results, avg_metrics = k_fold_cross_validation(
    #     combined_dataset, EEGTransformer, k_folds=5, epochs=10
    # )
    # 
    # # Opción 2: Optimización de Hiperparámetros (computacionalmente costoso)
    # best_params, study = optimize_hyperparameters_optuna(
    #     train_dataset, val_dataset, EEGTransformer, 
    #     n_trials=50, n_epochs=10
    # )
    # 
    # # Opción 3: Entrenar modelo final con mejoras
    # model = EEGTransformer(**best_params)  # Usar mejores hiperparámetros
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # 
    # history = train_with_enhancements(model, train_loader, val_loader, epochs=50)
    # 
    # # Evaluar en test
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # test_metrics = evaluate(model, test_loader)
    
    print("\n✓ Archivo creado: dl_with_cv_and_tuning.py")
    print("  Integra estas funciones en transformers.py, CNN.py, RNN.py")
