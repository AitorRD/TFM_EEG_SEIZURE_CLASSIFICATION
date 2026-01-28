"""
Experimentos de Machine Learning y Deep Learning para Clasificación de Convulsiones en EEG
Configuración basada en YAML + Optimización con Optuna
Soporte para ML tradicional y DL con skorch
"""

import pandas as pd
import numpy as np
import os
import yaml
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Feature extraction
from tsfresh import extract_features

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, make_scorer)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
import xgboost as xgb

# Optuna
import optuna
from optuna.samplers import TPESampler

# PyTorch y skorch (para DL)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping, LRScheduler
    from dl_models import (EEGTransformer, CNN1DClassifier, LSTMClassifier, 
                           GRUClassifier, EEGWindowDataset, EEGFeaturesDataset)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠ PyTorch/skorch no disponible. Solo se pueden usar modelos ML tradicionales.")
    print("  Instalar con: pip install torch skorch")


class MLExperiment:
    """Clase principal para experimentos de ML configurables"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Inicializa el experimento cargando la configuración
        
        Args:
            config_path (str): Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.random_state = self.config['experiment']['random_state']
        self.n_jobs = self.config['experiment'].get('n_jobs', -1)
        
        # Estructuras para almacenar resultados
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.selected_features = None
        self.pipelines = {}
        self.results = {}
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _load_config(self, config_path):
        """Carga el archivo de configuración YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuración cargada desde: {config_path}")
        print(f"  Experimento: {config['experiment']['name']}\n")
        return config
    
    def _create_directories(self):
        """Crea los directorios necesarios para resultados"""
        dirs = [
            Path(self.config['paths']['results']['xai_dir']),
            Path(self.config['paths']['results']['optuna_dir']),
            Path(self.config['paths']['results']['metrics']).parent
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def extract_tsfresh_features(self, file_path, mode="train"):
        """
        Extrae características usando TSFRESH
        
        Args:
            file_path (str): Ruta al archivo CSV con datos de ventanas
            mode (str): 'train', 'val' o 'test'
            
        Returns:
            tuple: (features_df, labels_series)
        """
        print(f"\n[{mode.upper()}] Extrayendo características con TSFRESH...")
        
        df = pd.read_csv(file_path)
        df['Time (s)'] = df.groupby('window_id').cumcount()
        df_windowed = df.rename(columns={'window_id': 'id'})
        
        signal_cols = [col for col in df_windowed.columns 
                      if col not in ['id', 'Time (s)', 'Seizure', 'idSession', 'idPatient']]
        
        # Obtener parámetros desde config
        custom_fc_parameters = self.config['feature_extraction']['custom_fc_parameters']
        
        features = extract_features(
            df_windowed[['id', 'Time (s)'] + signal_cols],
            column_id="id",
            column_sort="Time (s)",
            disable_progressbar=False,
            n_jobs=self.n_jobs,
            default_fc_parameters=custom_fc_parameters
        )
        
        # Imputación de valores faltantes
        features = features.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        imputer.fit(features)
        
        features_imputed = pd.DataFrame(
            imputer.transform(features),
            columns=imputer.get_feature_names_out(features.columns),
            index=features.index
        )
        
        labels = df_windowed.groupby('id')['Seizure'].max()
        
        # Guardar features y labels
        features_path = self.config['paths']['features'][mode]
        labels_path = self.config['paths']['labels'][mode]
        features_imputed.to_csv(features_path)
        labels.to_csv(labels_path, header=True)
        print(f"  ✓ Features guardadas: {features_path}")
        print(f"  ✓ Labels guardadas: {labels_path}")
        
        return features_imputed, labels
    
    def load_or_extract_features(self):
        """Carga features desde CSV o las extrae si no existen"""
        modes = ['train', 'val', 'test']
        datasets = {}
        
        for mode in modes:
            features_path = self.config['paths']['features'][mode]
            labels_path = self.config['paths']['labels'][mode]
            data_path = self.config['paths']['data'][mode]
            
            # Verificar si existen los archivos de features
            if (os.path.exists(features_path) and os.path.exists(labels_path) 
                and not self.config['feature_extraction']['enabled']):
                print(f"\n[{mode.upper()}] Cargando features desde CSV...")
                X = pd.read_csv(features_path, index_col=0)
                y = pd.read_csv(labels_path, index_col=0).squeeze()
            else:
                X, y = self.extract_tsfresh_features(data_path, mode)
            
            datasets[mode] = (X, y)
            print(f"  Shape: {X.shape}")
        
        self.X_train, self.y_train = datasets['train']
        self.X_val, self.y_val = datasets['val']
        self.X_test, self.y_test = datasets['test']
    
    def select_features(self):
        """Selecciona las mejores k características"""
        if not self.config['feature_selection']['enabled']:
            print("\n[FEATURE SELECTION] Deshabilitada en config")
            self.selected_features = self.X_train.columns
            return
        
        k = self.config['feature_selection']['k']
        k = min(k, self.X_train.shape[1])
        
        print(f"\n[FEATURE SELECTION] Seleccionando {k} mejores features...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(self.X_train, self.y_train)
        
        self.selected_features = self.X_train.columns[selector.get_support()]
        self.X_train = pd.DataFrame(X_selected, columns=self.selected_features)
        self.X_val = self.X_val[self.selected_features]
        self.X_test = self.X_test[self.selected_features]
        
        # Guardar features seleccionadas
        selected_path = self.config['paths']['selected_features']
        pd.Series(self.selected_features, name="feature").to_csv(
            selected_path, index=False, header=True
        )
        print(f"  ✓ {len(self.selected_features)} features seleccionadas")
        print(f"  ✓ Guardadas en: {selected_path}")
    
    def get_enabled_models(self):
        """Retorna lista de modelos habilitados en la configuración"""
        enabled = []
        for model_key, model_config in self.config['models'].items():
            if model_config.get('enabled', False):
                enabled.append(model_key)
        return enabled
    
    def create_default_pipeline(self, model_key):
        """
        Crea un pipeline con parámetros por defecto para un modelo
        
        Args:
            model_key (str): Clave del modelo (lr, rf, svc, knn, xgb)
            
        Returns:
            Pipeline: Pipeline de sklearn
        """
        model_config = self.config['models'][model_key]
        default_params = model_config.get('default_params', {})
        default_params['random_state'] = self.random_state
        
        if model_key == "lr":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(**default_params))
            ])
        elif model_key == "rf":
            return Pipeline([
                ('rf', RandomForestClassifier(**default_params))
            ])
        elif model_key == "svc":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(**default_params))
            ])
        elif model_key == "knn":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(**default_params))
            ])
        elif model_key == "xgb":
            return Pipeline([
                ('xgb', xgb.XGBClassifier(**default_params, random_state=self.random_state))
            ])
        else:
            raise ValueError(f"Modelo no soportado: {model_key}")
    
    def get_enabled_dl_models(self):
        """Retorna lista de modelos DL habilitados en la configuración"""
        if not PYTORCH_AVAILABLE or not self.config.get('deep_learning', {}).get('enabled', False):
            return []
        
        enabled = []
        for model_key, model_config in self.config.get('dl_models', {}).items():
            if model_config.get('enabled', False):
                enabled.append(model_key)
        return enabled
    
    def create_dl_model(self, model_key, **params):
        """
        Crea un modelo de DL envuelto en skorch NeuralNetClassifier
        
        Args:
            model_key (str): Clave del modelo DL (transformer, cnn, lstm, gru)
            **params: Parámetros del modelo
            
        Returns:
            NeuralNetClassifier: Modelo envuelto en skorch
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch/skorch no está disponible")
        
        model_config = self.config['dl_models'][model_key]
        default_params = model_config.get('default_params', {})
        dl_config = self.config['deep_learning']
        
        # Combinar parámetros
        model_params = {**default_params, **params}
        
        # Seleccionar arquitectura
        if model_key == "transformer":
            module = EEGTransformer
        elif model_key == "cnn":
            module = CNN1DClassifier
            # Para CNN, necesitamos el número de features
            model_params['input_features'] = self.X_train.shape[1] if self.X_train is not None else 50
        elif model_key == "lstm":
            module = LSTMClassifier
        elif model_key == "gru":
            module = GRUClassifier
        else:
            raise ValueError(f"Modelo DL no soportado: {model_key}")
        
        # Calcular class weights
        if self.y_train is not None:
            y_train_np = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights_tensor = None
        
        # Configurar callbacks
        callbacks = []
        
        # Early Stopping
        if dl_config['early_stopping']['enabled']:
            callbacks.append(
                EarlyStopping(
                    monitor='valid_loss',
                    patience=dl_config['early_stopping']['patience'],
                    threshold=dl_config['early_stopping']['min_delta']
                )
            )
        
        # LR Scheduler
        if dl_config['lr_scheduler']['enabled']:
            if dl_config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
                callbacks.append(
                    LRScheduler(
                        policy='ReduceLROnPlateau',
                        monitor='valid_loss',
                        mode=dl_config['lr_scheduler']['mode'],
                        factor=dl_config['lr_scheduler']['factor'],
                        patience=dl_config['lr_scheduler']['patience']
                    )
                )
        
        # Determinar device
        device = self.config['experiment'].get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("  ⚠ CUDA no disponible, usando CPU")
        
        # Preparar parámetros del módulo (filtrar lr y weight_decay que van al optimizer)
        module_params = {k: v for k, v in model_params.items() if k not in ['lr', 'weight_decay']}
        
        # Crear kwargs para NeuralNetClassifier
        net_kwargs = {
            'module': module,
            'max_epochs': dl_config['epochs'],
            'batch_size': dl_config['batch_size'],
            'lr': params.get('lr', 1e-4),
            'optimizer': torch.optim.Adam,
            'optimizer__weight_decay': params.get('weight_decay', 0.0),
            'criterion': nn.CrossEntropyLoss,
            'iterator_train__shuffle': True,
            'callbacks': callbacks,
            'device': device,
            'verbose': 1
        }
        
        # Agregar parámetros del módulo con prefijo module__
        for key, value in module_params.items():
            net_kwargs[f'module__{key}'] = value
        
        # Agregar class weights si existen
        if class_weights_tensor is not None:
            net_kwargs['criterion__weight'] = class_weights_tensor
        
        # Crear NeuralNetClassifier
        net = NeuralNetClassifier(**net_kwargs)
        
        return net
    
    def create_dl_pipeline(self, model_key, **params):
        """
        Crea un pipeline que incluye preprocesamiento y modelo DL
        
        Args:
            model_key (str): Clave del modelo DL
            **params: Parámetros del modelo
            
        Returns:
            Pipeline: Pipeline con preprocesamiento y modelo DL
        """
        # Para CNN que trabaja con features, agregar scaler
        if model_key == "cnn":
            return Pipeline([
                ('scaler', StandardScaler()),
                (model_key, self.create_dl_model(model_key, **params))
            ])
        else:
            # Transformer, LSTM, GRU trabajan con datos raw (no necesitan scaler)
            return Pipeline([
                (model_key, self.create_dl_model(model_key, **params))
            ])
    
    def load_dl_data(self):
        """
        Carga datos en formato apropiado para DL
        Retorna datasets de PyTorch según el formato configurado
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch no está disponible")
        
        dl_config = self.config['deep_learning']
        data_format = dl_config.get('data_format', 'features')
        
        if data_format == 'raw':
            # Cargar ventanas raw para Transformer/LSTM/GRU
            print("\n[DL] Cargando datos raw (ventanas EEG)...")
            
            train_df = pd.read_csv(self.config['paths']['data']['train'])
            val_df = pd.read_csv(self.config['paths']['data']['val'])
            test_df = pd.read_csv(self.config['paths']['data']['test'])
            
            train_dataset = EEGWindowDataset(
                train_df,
                n_channels=dl_config['input_channels'],
                seq_len=dl_config['sequence_length']
            )
            val_dataset = EEGWindowDataset(
                val_df,
                n_channels=dl_config['input_channels'],
                seq_len=dl_config['sequence_length']
            )
            test_dataset = EEGWindowDataset(
                test_df,
                n_channels=dl_config['input_channels'],
                seq_len=dl_config['sequence_length']
            )
            
            # Para skorch, necesitamos X, y en formato numpy
            self.X_train = train_dataset.data.numpy()
            self.y_train = train_dataset.labels.numpy()
            self.X_val = val_dataset.data.numpy()
            self.y_val = val_dataset.labels.numpy()
            self.X_test = test_dataset.data.numpy()
            self.y_test = test_dataset.labels.numpy()
            
            print(f"  Train shape: {self.X_train.shape}")
            print(f"  Val shape: {self.X_val.shape}")
            print(f"  Test shape: {self.X_test.shape}")
            
        elif data_format == 'features':
            # Usar features ya extraídas (para CNN sobre features)
            print("\n[DL] Usando features extraídas...")
            # X_train, X_val, X_test ya están cargados por load_or_extract_features
            pass
        
        else:
            raise ValueError(f"Formato de datos no soportado: {data_format}")
    
    def cross_validate_models(self):
        """Realiza validación cruzada en todos los modelos habilitados"""
        if not self.config['cross_validation']['enabled']:
            print("\n[CROSS VALIDATION] Deshabilitada en config")
            return None
        
        cv_config = self.config['cross_validation']
        n_folds = cv_config['n_folds']
        scoring = cv_config['scoring']
        
        print(f"\n{'='*60}")
        print(f"  VALIDACIÓN CRUZADA ({n_folds}-Fold)")
        print(f"  Métrica: {scoring}")
        print(f"{'='*60}\n")
        
        cv_strategy = StratifiedKFold(
            n_splits=n_folds,
            shuffle=cv_config['shuffle'],
            random_state=self.random_state
        )
        
        enabled_models = self.get_enabled_models()
        cv_results = {}
        
        for model_key in enabled_models:
            model_name = self.config['models'][model_key]['name']
            print(f"Evaluando {model_name}...")
            
            pipeline = self.create_default_pipeline(model_key)
            
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=self.n_jobs
            )
            
            cv_results[model_key] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            print(f"  → Score: {scores.mean():.4f} (±{scores.std():.4f})")
            print(f"  → Scores por fold: {scores}\n")
        
        best_model = max(cv_results, key=lambda x: cv_results[x]['mean'])
        best_name = self.config['models'][best_model]['name']
        print(f"🏆 Mejor modelo (CV): {best_name} - Score: {cv_results[best_model]['mean']:.4f}\n")
        
        return cv_results, best_model
    
    def create_optuna_objective(self, model_key):
        """
        Crea función objetivo para optimización con Optuna
        
        Args:
            model_key (str): Clave del modelo a optimizar
            
        Returns:
            callable: Función objetivo para Optuna
        """
        model_config = self.config['models'][model_key]
        search_space = model_config['optuna_search_space']
        default_params = model_config.get('default_params', {})
        
        cv_config = self.config['cross_validation']
        cv_strategy = StratifiedKFold(
            n_splits=cv_config['n_folds'],
            shuffle=cv_config['shuffle'],
            random_state=self.random_state
        )
        
        def objective(trial):
            # Construir parámetros desde el espacio de búsqueda
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config['type']
                
                if param_type == 'loguniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], log=True
                    )
                elif param_type == 'uniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Combinar con parámetros por defecto
            all_params = {**default_params, **params, 'random_state': self.random_state}
            
            # Crear pipeline
            if model_key == "lr":
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(**all_params))
                ])
            elif model_key == "rf":
                pipeline = Pipeline([
                    ('rf', RandomForestClassifier(**all_params))
                ])
            elif model_key == "svc":
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svc', SVC(**all_params))
                ])
            elif model_key == "knn":
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier(**params))
                ])
            elif model_key == "xgb":
                pipeline = Pipeline([
                    ('xgb', xgb.XGBClassifier(**all_params))
                ])
            
            # Evaluar con CV
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv_strategy,
                scoring=cv_config['scoring'],
                n_jobs=self.n_jobs
            )
            
            return scores.mean()
        
        return objective
    
    def optimize_with_optuna(self, model_key):
        """
        Optimiza hiperparámetros usando Optuna
        
        Args:
            model_key (str): Clave del modelo a optimizar
            
        Returns:
            tuple: (best_params, best_pipeline, study)
        """
        optuna_config = self.config['optuna']
        model_name = self.config['models'][model_key]['name']
        
        print(f"\n{'='*60}")
        print(f"  OPTIMIZACIÓN CON OPTUNA - {model_name}")
        print(f"  Trials: {optuna_config['n_trials']}")
        print(f"{'='*60}\n")
        
        objective = self.create_optuna_objective(model_key)
        
        # Crear estudio
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=optuna_config['n_startup_trials'],
                n_warmup_steps=optuna_config['n_warmup_steps']
            ),
            study_name=f'{model_key}_optimization'
        )
        
        # Optimizar
        study.optimize(
            objective,
            n_trials=optuna_config['n_trials'],
            timeout=optuna_config.get('timeout'),
            show_progress_bar=optuna_config['show_progress_bar']
        )
        
        print(f"\n✓ Optimización completada!")
        print(f"  → Mejor Score (CV): {study.best_value:.4f}")
        print(f"  → Trial #: {study.best_trial.number}")
        print(f"  → Mejores hiperparámetros:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")
        
        # Guardar resultados
        trials_df = study.trials_dataframe()
        optuna_dir = Path(self.config['paths']['results']['optuna_dir'])
        trials_path = optuna_dir / f"optuna_results_{model_key}.csv"
        trials_df.to_csv(trials_path, index=False)
        print(f"\n  → Resultados guardados: {trials_path}")
        
        # Guardar visualizaciones
        try:
            import optuna.visualization as vis
            
            fig1 = vis.plot_optimization_history(study)
            fig2 = vis.plot_param_importances(study)
            fig3 = vis.plot_slice(study)
            
            results_dir = Path(self.config['paths']['results']['metrics']).parent
            fig1.write_html(results_dir / f"optuna_history_{model_key}.html")
            fig2.write_html(results_dir / f"optuna_importance_{model_key}.html")
            fig3.write_html(results_dir / f"optuna_slice_{model_key}.html")
            print(f"  → Visualizaciones guardadas en {results_dir}/\n")
        except ImportError:
            print(f"  (Instala plotly para visualizaciones: pip install plotly)\n")
        
        # Crear pipeline con mejores parámetros
        best_params = study.best_params
        model_config = self.config['models'][model_key]
        default_params = model_config.get('default_params', {})
        all_params = {**default_params, **best_params, 'random_state': self.random_state}
        
        if model_key == "lr":
            best_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(**all_params))
            ])
        elif model_key == "rf":
            best_pipeline = Pipeline([
                ('rf', RandomForestClassifier(**all_params))
            ])
        elif model_key == "svc":
            best_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(**all_params))
            ])
        elif model_key == "knn":
            best_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(**best_params))
            ])
        elif model_key == "xgb":
            best_pipeline = Pipeline([
                ('xgb', xgb.XGBClassifier(**all_params))
            ])
        
        best_pipeline.fit(self.X_train, self.y_train)
        
        return best_params, best_pipeline, study
    
    def train_models(self):
        """Entrena todos los modelos habilitados (ML y DL, con o sin optimización)"""
        # Determinar tipo de experimento
        experiment_type = self.config['experiment'].get('type', 'ml')
        
        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO DE MODELOS ({experiment_type.upper()})")
        print(f"{'='*60}\n")
        
        if experiment_type == 'ml' or not self.config.get('deep_learning', {}).get('enabled', False):
            # Entrenar modelos ML tradicionales
            enabled_models = self.get_enabled_models()
            
            for model_key in enabled_models:
                model_name = self.config['models'][model_key]['name']
                print(f"\nEntrenando: {model_name}")
                
                if self.config['optuna']['enabled']:
                    # Optimizar con Optuna
                    best_params, pipeline, study = self.optimize_with_optuna(model_key)
                    self.pipelines[model_key] = pipeline
                else:
                    # Usar parámetros por defecto
                    pipeline = self.create_default_pipeline(model_key)
                    pipeline.fit(self.X_train, self.y_train)
                    self.pipelines[model_key] = pipeline
                    print(f"  ✓ Entrenado con parámetros por defecto")
        
        elif experiment_type == 'dl':
            # Entrenar modelos DL
            if not PYTORCH_AVAILABLE:
                print("  ⚠ PyTorch/skorch no disponible. No se pueden entrenar modelos DL.")
                return
            
            enabled_dl_models = self.get_enabled_dl_models()
            
            if not enabled_dl_models:
                print("  ⚠ No hay modelos DL habilitados en la configuración.")
                return
            
            for model_key in enabled_dl_models:
                model_name = self.config['dl_models'][model_key]['name']
                print(f"\nEntrenando: {model_name}")
                
                if self.config['optuna']['enabled']:
                    # Optimizar hiperparámetros DL con Optuna
                    best_params, pipeline, study = self.optimize_dl_with_optuna(model_key)
                    self.pipelines[model_key] = pipeline
                else:
                    # Usar parámetros por defecto
                    pipeline = self.create_dl_pipeline(model_key)
                    
                    # Preparar datos para validación en skorch
                    # Convertir a numpy/float32 si es necesario
                    if hasattr(self.X_train, 'values'):
                        X_train_np = self.X_train.values.astype(np.float32)
                        y_train_np = self.y_train.values.astype(np.int64)
                    else:
                        X_train_np = self.X_train.astype(np.float32)
                        y_train_np = self.y_train.astype(np.int64)
                    
                    if hasattr(self.X_val, 'values'):
                        X_val_np = self.X_val.values.astype(np.float32)
                        y_val_np = self.y_val.values.astype(np.int64)
                    else:
                        X_val_np = self.X_val.astype(np.float32)
                        y_val_np = self.y_val.astype(np.int64)
                    
                    # Fit con validación
                    pipeline.fit(X_train_np, y_train_np,
                               **{f'{model_key}__X_valid': X_val_np,
                                  f'{model_key}__y_valid': y_val_np})
                    
                    self.pipelines[model_key] = pipeline
                    print(f"  ✓ Entrenado con parámetros por defecto")
    
    def optimize_dl_with_optuna(self, model_key):
        """
        Optimiza hiperparámetros de un modelo DL usando Optuna
        
        Args:
            model_key (str): Clave del modelo DL (transformer, cnn, lstm, gru)
            
        Returns:
            tuple: (best_params, best_pipeline, study)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch/skorch no está disponible")
        
        optuna_config = self.config['optuna']
        model_config = self.config['dl_models'][model_key]
        model_name = model_config['name']
        search_space = model_config['optuna_search_space']
        
        print(f"\n{'='*60}")
        print(f"  OPTIMIZACIÓN CON OPTUNA - {model_name}")
        print(f"  Trials: {optuna_config['n_trials']}")
        print(f"{'='*60}\n")
        
        # Crear función objetivo para Optuna
        def objective(trial):
            # Sugerir hiperparámetros
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config['type']
                
                if param_type == 'loguniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], log=True
                    )
                elif param_type == 'uniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Crear pipeline con hiperparámetros sugeridos
            pipeline = self.create_dl_pipeline(model_key, **params)
            
            # Preparar datos de validación
            if hasattr(self.X_val, 'values'):
                X_val_np = self.X_val.values.astype(np.float32)
                y_val_np = self.y_val.values.astype(np.int64)
            else:
                X_val_np = self.X_val.astype(np.float32)
                y_val_np = self.y_val.astype(np.int64)
            
            # Entrenar con validación
            pipeline.fit(self.X_train, self.y_train,
                        **{f'{model_key}__X_valid': X_val_np,
                           f'{model_key}__y_valid': y_val_np})
            
            # Evaluar en validación
            y_pred = pipeline.predict(self.X_val)
            
            # Usar F1 weighted como métrica
            f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)
            
            return f1
        
        # Crear y ejecutar estudio
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=optuna_config['n_startup_trials'],
                n_warmup_steps=optuna_config['n_warmup_steps']
            ),
            study_name=f'{model_key}_dl_optimization'
        )
        
        study.optimize(
            objective,
            n_trials=optuna_config['n_trials'],
            timeout=optuna_config.get('timeout'),
            show_progress_bar=optuna_config['show_progress_bar']
        )
        
        print(f"\n✓ Optimización completada!")
        print(f"  → Mejor F1 Score: {study.best_value:.4f}")
        print(f"  → Trial #: {study.best_trial.number}")
        print(f"  → Mejores hiperparámetros:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")
        
        # Guardar resultados
        trials_df = study.trials_dataframe()
        optuna_dir = Path(self.config['paths']['results']['optuna_dir'])
        trials_path = optuna_dir / f"optuna_dl_results_{model_key}.csv"
        trials_df.to_csv(trials_path, index=False)
        print(f"\n  → Resultados guardados: {trials_path}")
        
        # Visualizaciones
        try:
            import optuna.visualization as vis
            results_dir = Path(self.config['paths']['results']['metrics']).parent
            fig1 = vis.plot_optimization_history(study)
            fig2 = vis.plot_param_importances(study)
            fig3 = vis.plot_slice(study)
            fig1.write_html(results_dir / f"optuna_dl_history_{model_key}.html")
            fig2.write_html(results_dir / f"optuna_dl_importance_{model_key}.html")
            fig3.write_html(results_dir / f"optuna_dl_slice_{model_key}.html")
            print(f"  → Visualizaciones guardadas en {results_dir}/\n")
        except ImportError:
            print(f"  (Instala plotly para visualizaciones)\n")
        
        # Entrenar modelo final con mejores parámetros
        best_pipeline = self.create_dl_pipeline(model_key, **study.best_params)
        
        if hasattr(self.X_train, 'values'):
            X_train_np = self.X_train.values.astype(np.float32)
            y_train_np = self.y_train.values.astype(np.int64)
        else:
            X_train_np = self.X_train.astype(np.float32)
            y_train_np = self.y_train.astype(np.int64)
        
        if hasattr(self.X_val, 'values'):
            X_val_np = self.X_val.values.astype(np.float32)
            y_val_np = self.y_val.values.astype(np.int64)
        else:
            X_val_np = self.X_val.astype(np.float32)
            y_val_np = self.y_val.astype(np.int64)
        
        best_pipeline.fit(X_train_np, y_train_np,
                         **{f'{model_key}__X_valid': X_val_np,
                            f'{model_key}__y_valid': y_val_np})
        
        return study.best_params, best_pipeline, study
    
    def evaluate_on_validation(self):
        """Evalúa todos los modelos en el conjunto de validación"""
        print(f"\n{'='*60}")
        print(f"  EVALUACIÓN EN VALIDACIÓN")
        print(f"{'='*60}\n")
        
        val_results = {}
        
        for model_key, pipeline in self.pipelines.items():
            # Obtener nombre del modelo (ML o DL)
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
            else:
                model_name = model_key
            
            preds = pipeline.predict(self.X_val)
            
            metrics = {
                'accuracy': accuracy_score(self.y_val, preds),
                'precision': precision_score(self.y_val, preds, zero_division=0),
                'recall': recall_score(self.y_val, preds, zero_division=0),
                'f1': f1_score(self.y_val, preds, zero_division=0),
                'f1_macro': f1_score(self.y_val, preds, average='macro', zero_division=0),
            }
            
            if hasattr(pipeline, 'predict_proba'):
                try:
                    metrics['roc_auc'] = roc_auc_score(self.y_val, pipeline.predict_proba(self.X_val)[:, 1])
                except:
                    metrics['roc_auc'] = None
            
            val_results[model_key] = metrics
            
            print(f"{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  F1 Macro:  {metrics['f1_macro']:.4f}\n")
        
        # Encontrar mejor modelo
        best_model = max(val_results, key=lambda x: val_results[x]['f1_macro'])
        if best_model in self.config.get('models', {}):
            best_name = self.config['models'][best_model]['name']
        elif best_model in self.config.get('dl_models', {}):
            best_name = self.config['dl_models'][best_model]['name']
        else:
            best_name = best_model
        print(f"🏆 Mejor modelo en validación: {best_name}\n")
        
        return val_results, best_model
    
    def evaluate_on_test(self):
        """Evalúa todos los modelos en el conjunto de test"""
        print(f"\n{'='*60}")
        print(f"  EVALUACIÓN EN TEST")
        print(f"{'='*60}\n")
        
        test_results = {}
        
        for model_key, pipeline in self.pipelines.items():
            # Obtener nombre del modelo (ML o DL)
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
            else:
                model_name = model_key
            
            preds = pipeline.predict(self.X_test)
            
            metrics = {}
            for metric_name in self.config['metrics']:
                if metric_name == 'accuracy':
                    metrics['Accuracy'] = accuracy_score(self.y_test, preds)
                elif metric_name == 'precision':
                    metrics['Precision'] = precision_score(self.y_test, preds, zero_division=0)
                elif metric_name == 'recall':
                    metrics['Recall'] = recall_score(self.y_test, preds, zero_division=0)
                elif metric_name == 'f1_score':
                    metrics['F1 Score'] = f1_score(self.y_test, preds, zero_division=0)
                elif metric_name == 'f1_macro':
                    metrics['F1 Macro'] = f1_score(self.y_test, preds, average='macro', zero_division=0)
                elif metric_name == 'f1_micro':
                    metrics['F1 Micro'] = f1_score(self.y_test, preds, average='micro', zero_division=0)
                elif metric_name == 'roc_auc' and hasattr(pipeline, 'predict_proba'):
                    try:
                        metrics['ROC AUC'] = roc_auc_score(self.y_test, pipeline.predict_proba(self.X_test)[:, 1])
                    except:
                        metrics['ROC AUC'] = None
            
            test_results[model_name] = metrics
            
            print(f"{model_name}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
            print()
        
        self.results['test'] = test_results
        return test_results
    
    def plot_and_save_metrics(self, results_dict):
        """Guarda tabla de métricas como imagen"""
        save_path = self.config['paths']['results']['metrics']
        
        df = pd.DataFrame(results_dict).T
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1))
        ax.axis("off")
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"✓ Tabla de métricas guardada: {save_path}\n")
    
    def generate_xai_explanations(self):
        """Genera explicaciones XAI (SHAP y LIME) para todos los modelos"""
        if not self.config['xai']['enabled']:
            print("\n[XAI] Deshabilitado en config")
            return
        
        xai_config = self.config['xai']
        save_dir = Path(self.config['paths']['results']['xai_dir'])
        
        print(f"\n{'='*60}")
        print(f"  GENERANDO EXPLICACIONES XAI")
        print(f"{'='*60}\n")
        
        for model_key, pipeline in self.pipelines.items():
            model_name = self.config['models'][model_key]['name']
            print(f"Generando XAI para {model_name}...")
            
            # Obtener el modelo y scaler del pipeline
            final_model = None
            scaler = None
            for step_name, step_transformer in pipeline.steps:
                if isinstance(step_transformer, StandardScaler):
                    scaler = step_transformer
                elif hasattr(step_transformer, 'predict'):
                    final_model = step_transformer
            
            if final_model is None:
                print(f"  [ERROR] No se encontró clasificador en pipeline\n")
                continue
            
            # Preparar datos escalados
            X_train_scaled = scaler.transform(self.X_train) if scaler else self.X_train.values
            X_test_scaled = scaler.transform(self.X_test) if scaler else self.X_test.values
            feature_names = list(self.X_test.columns)
            
            # SHAP
            if xai_config['methods']['shap']['enabled']:
                try:
                    self._generate_shap_plot(
                        model_key, model_name, pipeline,
                        X_train_scaled, X_test_scaled, feature_names,
                        save_dir, xai_config['methods']['shap']
                    )
                except Exception as e:
                    print(f"  [ERROR] SHAP falló: {e}")
            
            # LIME
            if xai_config['methods']['lime']['enabled']:
                try:
                    self._generate_lime_plot(
                        model_key, model_name, pipeline,
                        X_train_scaled, X_test_scaled, feature_names,
                        save_dir, xai_config['methods']['lime']
                    )
                except Exception as e:
                    print(f"  [ERROR] LIME falló: {e}")
    
    def _generate_shap_plot(self, model_key, model_name, pipeline, 
                           X_train_scaled, X_test_scaled, feature_names,
                           save_dir, shap_config):
        """Genera gráfico de importancia SHAP"""
        background_samples = shap_config['background_samples']
        top_features = shap_config['top_features']
        
        background_data = X_train_scaled[:min(background_samples, len(X_train_scaled))]
        
        # Usar KernelExplainer
        explainer = shap.KernelExplainer(pipeline.predict_proba, background_data)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Media de valores absolutos SHAP para clase positiva
        mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
        importances = pd.Series(mean_abs_shap, index=feature_names)
        importances = importances.sort_values(ascending=False).head(top_features)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="Greys")
        plt.title(f"Importancia SHAP - {model_name}")
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Características")
        plt.tight_layout()
        
        plot_path = save_dir / f"{model_key}_shap.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"  ✓ SHAP guardado: {plot_path}")
    
    def _generate_lime_plot(self, model_key, model_name, pipeline,
                           X_train_scaled, X_test_scaled, feature_names,
                           save_dir, lime_config):
        """Genera gráfico de importancia LIME"""
        n_samples = lime_config['n_samples']
        top_features = lime_config['top_features']
        
        if not hasattr(pipeline, "predict_proba"):
            print(f"  [INFO] LIME requiere predict_proba")
            return
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=feature_names,
            class_names=["No Seizure", "Seizure"],
            mode="classification",
            discretize_continuous=lime_config['discretize_continuous']
        )
        
        # Calcular importancias promedio
        all_lime_importances = []
        num_samples = min(n_samples, len(X_test_scaled))
        
        for idx in range(num_samples):
            exp = explainer.explain_instance(
                X_test_scaled[idx],
                pipeline.predict_proba,
                num_features=len(feature_names),
                top_labels=1
            )
            
            lime_weights = dict(exp.as_list(label=1))
            current_series = pd.Series(lime_weights)
            current_series = current_series.reindex(feature_names, fill_value=0)
            all_lime_importances.append(current_series)
        
        if not all_lime_importances:
            print(f"  [ERROR] No se generaron explicaciones LIME")
            return
        
        # Promedio de importancias
        avg_abs_lime = pd.concat(all_lime_importances, axis=1).abs().mean(axis=1)
        lime_series = avg_abs_lime.sort_values(ascending=False).head(top_features)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=lime_series.values, y=lime_series.index, palette="Greys")
        plt.title(f"Importancia LIME - {model_name}")
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Características")
        plt.tight_layout()
        
        plot_path = save_dir / f"{model_key}_lime.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"  ✓ LIME guardado: {plot_path}")
    
    def run(self):
        """Ejecuta el pipeline completo del experimento (ML o DL)"""
        print("\n" + "="*60)
        print(f"  INICIANDO EXPERIMENTO: {self.config['experiment']['name']}")
        experiment_type = self.config['experiment'].get('type', 'ml')
        print(f"  Tipo: {experiment_type.upper()}")
        print("="*60 + "\n")
        
        # Determinar flujo según tipo de experimento
        if experiment_type == 'dl' and self.config.get('deep_learning', {}).get('enabled', False):
            # Flujo para Deep Learning
            dl_data_format = self.config['deep_learning'].get('data_format', 'features')
            
            if dl_data_format == 'raw':
                # Cargar datos raw (ventanas temporales)
                self.load_dl_data()
            else:
                # Usar features extraídas (para CNN sobre features)
                # 1. Cargar/extraer features
                self.load_or_extract_features()
                
                # 2. Selección de features
                self.select_features()
            
            # 3. Validación cruzada (opcional, para DL puede ser costosa)
            if self.config['cross_validation']['enabled']:
                print("\n[INFO] Validación cruzada para DL está deshabilitada (computacionalmente costosa)")
                print("       Se usará conjunto de validación para early stopping\n")
            
            # 4. Entrenar modelos DL
            self.train_models()
            
            # 5. Evaluar en validación
            val_results, best_val_model = self.evaluate_on_validation()
            
            # 6. Evaluar en test
            test_results = self.evaluate_on_test()
            
            # 7. Guardar tabla de métricas
            self.plot_and_save_metrics(test_results)
            
            # 8. XAI (limitado para DL, requiere adaptación)
            print("\n[INFO] XAI para DL requiere métodos específicos (ej: Attention weights, Grad-CAM)")
            print("       Saltando XAI tradicional (SHAP/LIME) que es muy lento para redes neuronales\n")
        
        else:
            # Flujo para Machine Learning tradicional
            # 1. Cargar/extraer features
            self.load_or_extract_features()
            
            # 2. Selección de features
            self.select_features()
            
            # 3. Validación cruzada (opcional)
            if self.config['cross_validation']['enabled']:
                cv_results, best_cv_model = self.cross_validate_models()
            
            # 4. Entrenar modelos
            self.train_models()
            
            # 5. Evaluar en validación
            val_results, best_val_model = self.evaluate_on_validation()
            
            # 6. Evaluar en test
            test_results = self.evaluate_on_test()
            
            # 7. Guardar tabla de métricas
            self.plot_and_save_metrics(test_results)
            
            # 8. Generar explicaciones XAI
            self.generate_xai_explanations()
        
        print("\n" + "="*60)
        print("  EXPERIMENTO COMPLETADO")
        print("="*60 + "\n")


def main():
    """Función principal"""
    # Crear y ejecutar experimento
    experiment = MLExperiment(config_path="experimentation/classic/config.yaml")
    experiment.run()


if __name__ == "__main__":
    main()
