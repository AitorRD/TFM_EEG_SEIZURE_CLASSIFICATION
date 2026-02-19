"""
Experimentos de Machine Learning y Deep Learning para Clasificación de Convulsiones en EEG
Configuración basada en YAML + Optimización con Optuna
Soporte para ML tradicional y DL con skorch
"""

import gc
import joblib
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
import time
from datetime import datetime, timedelta
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
                             f1_score, roc_auc_score, make_scorer, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, RocCurveDisplay)
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
    from dl_models import (CNN1DClassifier, LSTMClassifier, 
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
        
        # Fix n_jobs for Windows compatibility with multiprocessing
        n_jobs_config = self.config['experiment'].get('n_jobs', -1)
        if n_jobs_config == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = max(1, n_jobs_config)  # Ensure at least 1
        
        # Structures to store results
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        # Raw data for sequence models (LSTM/GRU) - separate from features
        self.X_train_raw = None
        self.y_train_raw = None
        self.X_val_raw = None
        self.y_val_raw = None
        self.X_test_raw = None
        self.y_test_raw = None
        self.selected_features = None
        self.selectors = {}  # Feature selectors per model (with optimized k)
        self.pipelines = {}
        self.results = {}
        
        # Create necessary directories
        self._create_directories()
    
    def _load_config(self, config_path):
        """Loads YAML configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuración cargada desde: {config_path}")
        print(f"  Experimento: {config['experiment']['name']}\n")
        return config
    
    def _create_directories(self):
        """Creates necessary directories for results"""
        dirs = [
            Path(self.config['paths']['results']['xai_dir']),
            # Path(self.config['paths']['results']['optuna_dir']),  # Disabled: Optuna results not saved
            Path(self.config['paths']['results']['metrics']).parent,
            Path('images/graphs'),  # For ROC curves and confusion matrices
            Path(self.config['paths']['results']['predictions_dir']),  # CSVs with predictions per model
            Path(self.config['paths']['models_dir']),  # Trained models (pipelines + selectors)
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def extract_tsfresh_features(self, file_path, mode="train"):
        """
        Extracts features using TSFRESH
        
        Args:
            file_path (str): Path to CSV file with window data
            mode (str): 'train', 'val' or 'test'
            
        Returns:
            tuple: (features_df, labels_series)
        """
        print(f"\n[{mode.upper()}] Extrayendo características con TSFRESH...")
        
        df = pd.read_csv(file_path)
        df['Time (s)'] = df.groupby('window_id').cumcount()
        df_windowed = df.rename(columns={'window_id': 'id'})
        
        signal_cols = [col for col in df_windowed.columns 
                      if col not in ['id', 'Time (s)', 'Seizure', 'idSession', 'idPatient']]
        
        # Get parameters from config
        custom_fc_parameters = self.config['feature_extraction']['custom_fc_parameters']
        
        features = extract_features(
            df_windowed[['id', 'Time (s)'] + signal_cols],
            column_id="id",
            column_sort="Time (s)",
            disable_progressbar=False,
            n_jobs=self.n_jobs,
            default_fc_parameters=custom_fc_parameters
        )
        
        # Imputation of missing values
        features = features.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        imputer.fit(features)
        
        features_imputed = pd.DataFrame(
            imputer.transform(features),
            columns=imputer.get_feature_names_out(features.columns),
            index=features.index
        )
        
        labels = df_windowed.groupby('id')['Seizure'].max()
        
        # Save features and labels
        features_path = self.config['paths']['features'][mode]
        labels_path = self.config['paths']['labels'][mode]
        features_imputed.to_csv(features_path)
        labels.to_csv(labels_path, header=True)
        print(f"  ✓ Features guardadas: {features_path}")
        print(f"  ✓ Labels guardadas: {labels_path}")
        
        return features_imputed, labels
    
    def load_or_extract_features(self):
        """Loads features from CSV or extracts them if they don't exist"""
        modes = ['train', 'val', 'test']
        datasets = {}
        
        for mode in modes:
            features_path = self.config['paths']['features'][mode]
            labels_path = self.config['paths']['labels'][mode]
            data_path = self.config['paths']['data'][mode]
            
            # Check if feature files exist
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
        
        # Ensure all datasets have the same columns in the same order
        common_cols = self.X_train.columns.intersection(self.X_val.columns).intersection(self.X_test.columns)
        self.X_train = self.X_train[common_cols]
        self.X_val = self.X_val[common_cols]
        self.X_test = self.X_test[common_cols]
        print(f"\n[INFO] Columnas comunes alineadas: {len(common_cols)} features")
    
    def select_features(self):
        """Selects the best k features (only if Optuna is disabled)"""
        if not self.config['feature_selection']['enabled']:
            print("\n[FEATURE SELECTION] Deshabilitada en config")
            self.selected_features = self.X_train.columns
            return
        
        # Si Optuna está habilitado, cada modelo optimizará su propio k
        if self.config['optuna']['enabled']:
            print("\n[FEATURE SELECTION] Con Optuna habilitado, cada modelo optimizará su propio k")
            print("  → Manteniendo todas las features para optimización individual\n")
            self.selected_features = self.X_train.columns
            return
        
        k = self.config['feature_selection']['k']
        k = min(k, self.X_train.shape[1])
        
        print(f"\n[FEATURE SELECTION] Seleccionando {k} mejores features...")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(self.X_train, self.y_train)  

        # Get selected feature names
        self.selected_features = self.X_train.columns[selector.get_support()]

        # Apply transformation
        self.X_train = pd.DataFrame(
            selector.transform(self.X_train),  
            columns=self.selected_features
        )
        self.X_val = pd.DataFrame(
            selector.transform(self.X_val),    
            columns=self.selected_features
        )
        self.X_test = pd.DataFrame(
            selector.transform(self.X_test),   
            columns=self.selected_features
        )
        
        # Save selected features
        selected_path = self.config['paths']['selected_features']
        pd.Series(self.selected_features, name="feature").to_csv(
            selected_path, index=False, header=True
        )
        print(f"  ✓ {len(self.selected_features)} features seleccionadas")
        print(f"  ✓ Guardadas en: {selected_path}")
    
    def get_enabled_models(self):
        """Returns list of enabled models in configuration"""
        enabled = []
        for model_key, model_config in self.config['models'].items():
            if model_config.get('enabled', False):
                enabled.append(model_key)
        return enabled
    
    def create_default_pipeline(self, model_key):
        """
        Creates a pipeline with default parameters for a model
        
        Args:
            model_key (str): Model key (lr, rf, svc, knn, xgb)
            
        Returns:
            Pipeline: sklearn Pipeline
        """
        model_config = self.config['models'][model_key]
        default_params = model_config.get('default_params', {}).copy()
        
        # KNN doesn't accept random_state, so we handle it separately
        if model_key != "knn":
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
                ('xgb', xgb.XGBClassifier(**default_params))
            ])
        else:
            raise ValueError(f"Modelo no soportado: {model_key}")
    
    def get_enabled_dl_models(self):
        """Returns list of enabled DL models in configuration"""
        if not PYTORCH_AVAILABLE or not self.config.get('deep_learning', {}).get('enabled', False):
            return []
        
        enabled = []
        for model_key, model_config in self.config.get('dl_models', {}).items():
            if model_config.get('enabled', False):
                enabled.append(model_key)
        return enabled
    
    # Models that require raw windowed data (3D: batch, time, channels)
    RAW_DL_MODELS = {'lstm', 'gru'}
    
    def _is_raw_dl_model(self, model_key):
        """Returns True if this DL model needs raw windowed data instead of features"""
        return model_key in self.RAW_DL_MODELS
    
    def _get_model_data(self, model_key, split='train'):
        """Returns the appropriate (X, y) for a model based on its data format
        
        Args:
            model_key (str): Model key
            split (str): 'train', 'val', or 'test'
            
        Returns:
            tuple: (X, y) appropriate for the model
        """
        if self._is_raw_dl_model(model_key) and self.X_train_raw is not None:
            data_map = {
                'train': (self.X_train_raw, self.y_train_raw),
                'val': (self.X_val_raw, self.y_val_raw),
                'test': (self.X_test_raw, self.y_test_raw),
            }
        else:
            data_map = {
                'train': (self.X_train, self.y_train),
                'val': (self.X_val, self.y_val),
                'test': (self.X_test, self.y_test),
            }
        return data_map[split]
    
    def load_raw_data(self):
        """Loads raw windowed data for sequence models (LSTM/GRU).
        Stores in separate attributes (X_*_raw, y_*_raw) to avoid overwriting features."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch no está disponible")
        
        dl_config = self.config['deep_learning']
        
        print("\n[DL] Cargando datos raw (ventanas EEG) para modelos de secuencia...")
        
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
        
        self.X_train_raw = train_dataset.data.numpy()
        self.y_train_raw = train_dataset.labels.numpy()
        self.X_val_raw = val_dataset.data.numpy()
        self.y_val_raw = val_dataset.labels.numpy()
        self.X_test_raw = test_dataset.data.numpy()
        self.y_test_raw = test_dataset.labels.numpy()
        
        print(f"  Train shape: {self.X_train_raw.shape}")
        print(f"  Val shape: {self.X_val_raw.shape}")
        print(f"  Test shape: {self.X_test_raw.shape}")
    
    def create_dl_model(self, model_key, **params):
        """
        Creates a DL model wrapped in skorch NeuralNetClassifier
        
        Args:
            model_key (str): DL model key (cnn, lstm, gru)
            **params: Model parameters
            
        Returns:
            NeuralNetClassifier: Model wrapped in skorch
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch/skorch no está disponible")
        
        model_config = self.config['dl_models'][model_key]
        default_params = model_config.get('default_params', {})
        dl_config = self.config['deep_learning']
        
        # Combine parameters
        model_params = {**default_params, **params}
        
        # Select architecture
        if model_key == "cnn":
            module = CNN1DClassifier
            # For CNN, we need the number of features
            # Use passed input_features if provided, otherwise use full training set
            if 'input_features' not in model_params:
                model_params['input_features'] = self.X_train.shape[1] if self.X_train is not None else 50
        elif model_key == "lstm":
            module = LSTMClassifier
        elif model_key == "gru":
            module = GRUClassifier
        else:
            raise ValueError(f"DL Model not supported: {model_key}")
        
        # Calculate class weights (use raw labels for raw models if available)
        y_for_weights = None
        if self._is_raw_dl_model(model_key) and self.y_train_raw is not None:
            y_for_weights = self.y_train_raw
        elif self.y_train is not None:
            y_for_weights = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
        
        if y_for_weights is not None:
            y_np = y_for_weights.values if hasattr(y_for_weights, 'values') else y_for_weights
            y_np = np.asarray(y_np).astype(np.int64)  # Ensure integer type for compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y_np), y=y_np)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights_tensor = None
        
        # Configure callbacks
        callbacks = []
        
        # Note: With train_split=None, we don't have internal validation during training
        # so we skip callbacks that monitor validation metrics during Optuna optimization
        # They can be added later for final training if needed
        
        # Early Stopping - skip for Optuna since we validate externally
        # if dl_config['early_stopping']['enabled']:
        #     callbacks.append(
        #         EarlyStopping(
        #             monitor='valid_loss',
        #             patience=dl_config['early_stopping']['patience'],
        #             threshold=dl_config['early_stopping']['min_delta']
        #         )
        #     )
        
        # LR Scheduler - skip for Optuna since we validate externally  
        # if dl_config['lr_scheduler']['enabled']:
        #     if dl_config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
        #         callbacks.append(
        #             LRScheduler(
        #                 policy='ReduceLROnPlateau',
        #                 monitor='valid_loss',
        #                 mode=dl_config['lr_scheduler']['mode'],
        #                 factor=dl_config['lr_scheduler']['factor'],
        #                 patience=dl_config['lr_scheduler']['patience']
        #             )
        #         )
        
        # Determine device
        device = self.config['experiment'].get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("  ⚠ CUDA not available, using CPU")
        
        # Prepare module parameters (filter lr and weight_decay that go to optimizer)
        module_params = {k: v for k, v in model_params.items() if k not in ['lr', 'weight_decay']}
        
        # Create kwargs for NeuralNetClassifier
        net_kwargs = {
            'module': module,
            'max_epochs': dl_config['epochs'],
            'batch_size': dl_config['batch_size'],
            'lr': params.get('lr', 1e-4),
            'optimizer': torch.optim.Adam,
            'optimizer__weight_decay': params.get('weight_decay', 0.0),
            'criterion': nn.CrossEntropyLoss,
            'iterator_train__shuffle': True,
            'train_split': None,  # We'll use separate validation set
            'callbacks': callbacks,
            'device': device,
            'verbose': 1
        }
        
        # Add module parameters with module__ prefix
        for key, value in module_params.items():
            net_kwargs[f'module__{key}'] = value
        
        # Add class weights if they exist
        if class_weights_tensor is not None:
            net_kwargs['criterion__weight'] = class_weights_tensor
        
        # Create NeuralNetClassifier
        net = NeuralNetClassifier(**net_kwargs)
        
        return net
    
    def create_dl_pipeline(self, model_key, **params):
        """
        Creates a pipeline that includes preprocessing and DL model
        
        Args:
            model_key (str): DL model key
            **params: Model parameters
            
        Returns:
            Pipeline: Pipeline with preprocessing and DL model
        """
        # Para CNN que trabaja con features, agregar scaler
        if model_key == "cnn":
            return Pipeline([
                ('scaler', StandardScaler()),
                (model_key, self.create_dl_model(model_key, **params))
            ])
        else:
            # LSTM, GRU work with raw data (no scaler needed)
            return Pipeline([
                (model_key, self.create_dl_model(model_key, **params))
            ])
    
    def load_dl_data(self):
        """
        Loads data in appropriate format for DL
        Returns PyTorch datasets according to configured format
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch no está disponible")
        
        dl_config = self.config['deep_learning']
        data_format = dl_config.get('data_format', 'features')
        
        if data_format == 'raw':
            # Load raw windows for LSTM/GRU
            print("\n[DL] Loading raw data (EEG windows)...")
            
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
            
            # For skorch, we need X, y in numpy format
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
            # Use already extracted features (for CNN on features)
            print("\n[DL] Using extracted features...")
            # X_train, X_val, X_test already loaded by load_or_extract_features
            pass
        
        else:
            raise ValueError(f"Data format not supported: {data_format}")
    
    def cross_validate_models(self):
        """Performs cross-validation on all enabled models"""
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
        Creates objective function for Optuna optimization
        Includes optimization of k (number of features) with SelectKBest
        
        Args:
            model_key (str): Model key to optimize
            
        Returns:
            callable: Objective function for Optuna
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
        
        # Get original features (before selection)
        X_train_full = self.X_train
        y_train_full = self.y_train
        
        def objective(trial):
            # 1. Suggest k (number of features to select)
            # Use config k as reference, but allow optimization
            k_reference = self.config['feature_selection'].get('k', 50)
            k_max = min(200, X_train_full.shape[1])  # Max 200 or total features
            k_min = max(10, int(k_max * 0.1))  # Min 10 or 10% of features
            
            k_features = trial.suggest_int('k_features', k_min, k_max)
            
            # 2. Apply SelectKBest with this k
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_train_selected = selector.fit_transform(X_train_full, y_train_full)
            
            # 3. Construir parámetros del modelo desde el espacio de búsqueda
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
            
            # 4. Crear pipeline (sin SelectKBest, ya aplicado arriba)
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
            
            # 5. Evaluar con CV en features seleccionadas
            scores = cross_val_score(
                pipeline, X_train_selected, y_train_full,
                cv=cv_strategy,
                scoring=cv_config['scoring'],
                n_jobs=self.n_jobs
            )
            
            return scores.mean()
        
        return objective
    
    def optimize_with_optuna(self, model_key):
        """
        Optimizes hyperparameters using Optuna
        
        Args:
            model_key (str): Model key to optimize
            
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
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=optuna_config['n_startup_trials'],
                n_warmup_steps=optuna_config['n_warmup_steps']
            ),
            study_name=f'{model_key}_optimization'
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=optuna_config['n_trials'],
            timeout=optuna_config.get('timeout'),
            show_progress_bar=optuna_config['show_progress_bar']
        )
        
        print(f"\n✓ Optimization completed!")
        print(f"  → Best Score (CV): {study.best_value:.4f}")
        print(f"  → Trial #: {study.best_trial.number}")
        print(f"  → Best hyperparameters:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")
        
        # Optuna results saving disabled (CSVs and HTMLs not needed)
        # trials_df = study.trials_dataframe()
        # optuna_dir = Path(self.config['paths']['results']['optuna_dir'])
        # trials_path = optuna_dir / f"optuna_results_{model_key}.csv"
        # trials_df.to_csv(trials_path, index=False)
        # print(f"\n  → Results saved: {trials_path}")
        
        print(f"\n  (Optuna results not saved - disabled)\n")
        
        # Extract k_features from best_params and create selector
        best_params = study.best_params.copy()
        k_features = best_params.pop('k_features')  # Remove k from model params
        
        # Create and fit selector with optimized k
        print(f"\n  → Creating selector with k={k_features} features")
        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(self.X_train, self.y_train)
        
        # Store selector for this model
        self.selectors[model_key] = selector
        
        # Transform data with selected features
        X_train_selected = selector.transform(self.X_train)
        
        # Create pipeline with best parameters (without k_features)
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
        
        # Train on selected features
        best_pipeline.fit(X_train_selected, self.y_train)
        print(f"  ✓ Model trained with {k_features} selected features")
        
        return best_params, best_pipeline, study
    
    def _get_model_path(self, model_key):
        """Returns the file path for a saved model pipeline"""
        models_dir = Path(self.config['paths']['models_dir'])
        return models_dir / f"pipeline_{model_key}.joblib"
    
    def _get_selector_path(self, model_key):
        """Returns the file path for a saved feature selector"""
        models_dir = Path(self.config['paths']['models_dir'])
        return models_dir / f"selector_{model_key}.joblib"
    
    def save_model(self, model_key, pipeline):
        """Saves a trained pipeline (and its selector if exists) to disk"""
        model_path = self._get_model_path(model_key)
        joblib.dump(pipeline, model_path)
        print(f"  ✓ Modelo guardado: {model_path}")
        
        # Save selector if it exists for this model
        if model_key in self.selectors:
            selector_path = self._get_selector_path(model_key)
            joblib.dump(self.selectors[model_key], selector_path)
            print(f"  ✓ Selector guardado: {selector_path}")
    
    def load_model(self, model_key):
        """Loads a previously saved pipeline (and its selector) from disk.
        Returns True if loaded successfully, False otherwise."""
        model_path = self._get_model_path(model_key)
        selector_path = self._get_selector_path(model_key)
        
        if not model_path.exists():
            return False
        
        try:
            pipeline = joblib.load(model_path)
            self.pipelines[model_key] = pipeline
            
            # Load selector if it exists
            if selector_path.exists():
                self.selectors[model_key] = joblib.load(selector_path)
            
            # Get model name for display
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
            else:
                model_name = model_key
            
            print(f"  ✓ {model_name} cargado desde caché: {model_path}")
            if selector_path.exists():
                print(f"  ✓ Selector cargado desde caché: {selector_path}")
            return True
        except Exception as e:
            print(f"  [WARNING] Error cargando modelo {model_key}: {e}. Se re-entrenará.")
            return False
    
    def train_models(self, force_type=None):
        """Trains all enabled models (ML and DL, with or without optimization)
        
        Args:
            force_type (str): Force training of specific type ('ml', 'dl', or None for config)
        """
        # Determinar tipo de experimento
        experiment_type = force_type if force_type else self.config['experiment'].get('type', 'ml')
        
        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO DE MODELOS ({experiment_type.upper()})")
        print(f"{'='*60}\n")
        
        # Define execution order: DL first (lstm, gru, cnn), then ML (xgb, svc, rf, knn, lr)
        dl_execution_order = ['lstm', 'gru', 'cnn']
        ml_execution_order = ['xgb', 'svc', 'rf', 'knn', 'lr']
        
        # --- DL models first ---
        if experiment_type == 'dl' or (experiment_type == 'all' and PYTORCH_AVAILABLE and self.config.get('deep_learning', {}).get('enabled', False)):
            # Train DL models
            if not PYTORCH_AVAILABLE:
                print("  ⚠ PyTorch/skorch not available. Cannot train DL models.")
            else:
                enabled_dl_models = self.get_enabled_dl_models()
                # Sort by defined execution order
                enabled_dl_models = [m for m in dl_execution_order if m in enabled_dl_models]
                
                if not enabled_dl_models:
                    print("  ⚠ No DL models enabled in configuration.")
                else:
                    print("\n>>> Entrenando modelos de Deep Learning...\n")
                    for model_key in enabled_dl_models:
                        model_name = self.config['dl_models'][model_key]['name']
                        print(f"\nEntrenando: {model_name}")
                        
                        # Try to load cached model first
                        if self.load_model(model_key):
                            continue
                        
                        try:
                            if self.config['optuna']['enabled']:
                                # Optimize DL hyperparameters with Optuna
                                best_params, pipeline, study = self.optimize_dl_with_optuna(model_key)
                                self.pipelines[model_key] = pipeline
                            else:
                                # Use default parameters
                                pipeline = self.create_dl_pipeline(model_key)
                                
                                # Get appropriate data for this model (raw for sequence models, features for CNN)
                                X_train_model, y_train_model = self._get_model_data(model_key, 'train')
                                
                                # Convert to numpy/float32 if necessary
                                if hasattr(X_train_model, 'values'):
                                    X_train_np = X_train_model.values.astype(np.float32)
                                    y_train_np = y_train_model.values.astype(np.int64)
                                else:
                                    X_train_np = X_train_model.astype(np.float32)
                                    y_train_np = y_train_model.astype(np.int64)
                                
                                # Fit without validation split (train_split=None in model)
                                pipeline.fit(X_train_np, y_train_np)
                                
                                self.pipelines[model_key] = pipeline
                                print(f"  ✓ Trained with default parameters")
                            
                            # Save trained model to disk
                            self.save_model(model_key, self.pipelines[model_key])
                        finally:
                            # Free GPU memory between DL models
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
        
        # --- ML models after DL ---
        if experiment_type == 'ml' or (experiment_type == 'all' and self.get_enabled_models()):
            # Train traditional ML models
            enabled_models = self.get_enabled_models()
            # Sort by defined execution order
            enabled_models = [m for m in ml_execution_order if m in enabled_models]
            
            if enabled_models:
                print("\n>>> Entrenando modelos de Machine Learning tradicional...\n")
                for model_key in enabled_models:
                    model_name = self.config['models'][model_key]['name']
                    print(f"\nEntrenando: {model_name}")
                    
                    # Try to load cached model first
                    if self.load_model(model_key):
                        continue
                    
                    if self.config['optuna']['enabled']:
                        # Optimize with Optuna
                        best_params, pipeline, study = self.optimize_with_optuna(model_key)
                        self.pipelines[model_key] = pipeline
                    else:
                        # Use default parameters
                        pipeline = self.create_default_pipeline(model_key)
                        pipeline.fit(self.X_train, self.y_train)
                        self.pipelines[model_key] = pipeline
                        print(f"  ✓ Trained with default parameters")
                    
                    # Save trained model to disk
                    self.save_model(model_key, self.pipelines[model_key])
    
    def optimize_dl_with_optuna(self, model_key):
        """
        Optimizes hyperparameters of a DL model using Optuna
        
        Args:
            model_key (str): DL model key (cnn, lstm, gru)
            
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
        X_val_data, y_val_data = self._get_model_data(model_key, 'val')
        
        # Crear función objetivo para Optuna
        def objective(trial):
            # 1. Si usa features, optimizar k_features
            if use_feature_selection:
                k_max = min(200, X_train_data.shape[1])
                k_min = max(10, int(k_max * 0.1))
                k_features = trial.suggest_int('k_features', k_min, k_max)
                
                # Apply SelectKBest
                selector = SelectKBest(score_func=f_classif, k=k_features)
                X_train_selected = selector.fit_transform(X_train_data, y_train_data)
                X_val_selected = selector.transform(X_val_data)
            else:
                X_train_selected = X_train_data
                X_val_selected = X_val_data
            
            # 2. Sugerir hiperparámetros del modelo
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
            
            # Pass input_features based on selected features for CNN
            if use_feature_selection and model_key == 'cnn':
                params['input_features'] = k_features
            elif model_key == 'cnn':
                params['input_features'] = X_train_selected.shape[1]
            
            # Crear pipeline con hiperparámetros sugeridos
            pipeline = self.create_dl_pipeline(model_key, **params)
            
            # Preparar datos de validación
            if hasattr(X_val_selected, 'values'):
                X_val_np = X_val_selected.values.astype(np.float32)
            elif isinstance(X_val_selected, np.ndarray):
                X_val_np = X_val_selected.astype(np.float32)
            else:
                X_val_np = X_val_selected
            
            if hasattr(self.y_val, 'values'):
                y_val_np = self.y_val.values.astype(np.int64)
            elif hasattr(y_val_data, 'values'):
                y_val_np = y_val_data.values.astype(np.int64)
            else:
                y_val_np = y_val_data.astype(np.int64)
            
            # Preparar datos de entrenamiento
            if hasattr(X_train_selected, 'values'):
                X_train_np = X_train_selected.values.astype(np.float32)
            elif isinstance(X_train_selected, np.ndarray):
                X_train_np = X_train_selected.astype(np.float32)
            else:
                X_train_np = X_train_selected
            
            if hasattr(self.y_train, 'values'):
                y_train_np = self.y_train.values.astype(np.int64)
            elif hasattr(y_train_data, 'values'):
                y_train_np = y_train_data.values.astype(np.int64)
            else:
                y_train_np = y_train_data.astype(np.int64)
            
            # Train on training data only (no validation split in skorch)
            pipeline.fit(X_train_np, y_train_np)
            
            # Evaluate on separate validation set
            y_pred = pipeline.predict(X_val_np)
            
            # Usar F1 weighted como métrica
            f1 = f1_score(y_val_data, y_pred, average='weighted', zero_division=0)
            
            # Free GPU memory after each trial
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
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
        
        # Optuna results saving disabled (CSVs and HTMLs not needed)
        # trials_df = study.trials_dataframe()
        # optuna_dir = Path(self.config['paths']['results']['optuna_dir'])
        # trials_path = optuna_dir / f"optuna_dl_results_{model_key}.csv"
        # trials_df.to_csv(trials_path, index=False)
        # print(f"\n  → Resultados guardados: {trials_path}")
        
        print(f"\n  (Resultados de Optuna no guardados - desactivado)\n")
        
        # Extract k_features and create selector if using features
        best_params = study.best_params.copy()
        if use_feature_selection and 'k_features' in best_params:
            k_features = best_params.pop('k_features')
            
            print(f"  → Creating selector with k={k_features} features")
            selector = SelectKBest(score_func=f_classif, k=k_features)
            selector.fit(X_train_data, y_train_data)
            
            # Store selector for this model
            self.selectors[model_key] = selector
            
            # Transform data
            X_train_selected = selector.transform(X_train_data)
            X_val_selected = selector.transform(X_val_data)
        else:
            X_train_selected = X_train_data
            X_val_selected = X_val_data
        
        # For CNN, add input_features parameter with the correct feature count
        if model_key == 'cnn':
            if use_feature_selection and 'k_features' in locals():
                best_params['input_features'] = k_features
            else:
                # Use actual feature count from X_train_selected
                if hasattr(X_train_selected, 'shape'):
                    best_params['input_features'] = X_train_selected.shape[1]
                else:
                    best_params['input_features'] = len(X_train_selected[0])
        
        # Entrenar modelo final con mejores parámetros
        best_pipeline = self.create_dl_pipeline(model_key, **best_params)
        
        if hasattr(X_train_selected, 'values'):
            X_train_np = X_train_selected.values.astype(np.float32)
        elif isinstance(X_train_selected, np.ndarray):
            X_train_np = X_train_selected.astype(np.float32)
        else:
            X_train_np = X_train_selected
        
        if hasattr(y_train_data, 'values'):
            y_train_np = y_train_data.values.astype(np.int64)
        else:
            y_train_np = y_train_data.astype(np.int64)
        
        if hasattr(X_val_selected, 'values'):
            X_val_np = X_val_selected.values.astype(np.float32)
        elif isinstance(X_val_selected, np.ndarray):
            X_val_np = X_val_selected.astype(np.float32)
        else:
            X_val_np = X_val_selected
        
        if hasattr(y_val_data, 'values'):
            y_val_np = y_val_data.values.astype(np.int64)
        else:
            y_val_np = y_val_data.astype(np.int64)
        
        # Train best pipeline (no validation split in skorch)
        best_pipeline.fit(X_train_np, y_train_np)
        
        if use_feature_selection and model_key in self.selectors:
            print(f"  ✓ Model trained with {k_features} selected features")
        
        return best_params, best_pipeline, study
    
    def evaluate_on_validation(self):
        """Evaluates all models on validation set"""
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
            
            # Get appropriate data for this model
            if self._is_raw_dl_model(model_key) and self.X_val_raw is not None:
                X_val_model = self.X_val_raw.astype(np.float32)
                y_val_current = self.y_val_raw
            else:
                # Apply model-specific feature selection if exists
                if model_key in self.selectors:
                    X_val_model = self.selectors[model_key].transform(self.X_val)
                else:
                    X_val_model = self.X_val
                y_val_current = self.y_val
                
                # Convert to float32 for DL models (PyTorch requires float32)
                if model_key in self.config.get('dl_models', {}):
                    if hasattr(X_val_model, 'values'):
                        X_val_model = X_val_model.values.astype(np.float32)
                    elif isinstance(X_val_model, np.ndarray):
                        X_val_model = X_val_model.astype(np.float32)
            
            preds = pipeline.predict(X_val_model)
            
            metrics = {
                'accuracy': accuracy_score(y_val_current, preds),
                'precision': precision_score(y_val_current, preds, zero_division=0),
                'recall': recall_score(y_val_current, preds, zero_division=0),
                'f1': f1_score(y_val_current, preds, zero_division=0),
                'f1_macro': f1_score(y_val_current, preds, average='macro', zero_division=0),
            }
            
            if hasattr(pipeline, 'predict_proba'):
                try:
                    metrics['roc_auc'] = roc_auc_score(y_val_current, pipeline.predict_proba(X_val_model)[:, 1])
                except:
                    metrics['roc_auc'] = None
            
            val_results[model_key] = metrics
            
            print(f"{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  F1 Macro:  {metrics['f1_macro']:.4f}\n")
        
        # Find best model
        best_model = max(val_results, key=lambda x: val_results[x]['f1_macro'])
        if best_model in self.config.get('models', {}):
            best_name = self.config['models'][best_model]['name']
        elif best_model in self.config.get('dl_models', {}):
            best_name = self.config['dl_models'][best_model]['name']
        else:
            best_name = best_model
        print(f"🏆 Best model on validation: {best_name}\n")
        
        return val_results, best_model
    
    def evaluate_on_test(self):
        """Evaluates all models on test set"""
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
            
            # Get appropriate data for this model
            if self._is_raw_dl_model(model_key) and self.X_test_raw is not None:
                X_test_model = self.X_test_raw.astype(np.float32)
                y_test_current = self.y_test_raw
            else:
                # Apply model-specific feature selection if exists
                if model_key in self.selectors:
                    X_test_model = self.selectors[model_key].transform(self.X_test)
                else:
                    X_test_model = self.X_test
                y_test_current = self.y_test
                
                # Convert to float32 for DL models (PyTorch requires float32)
                if model_key in self.config.get('dl_models', {}):
                    if hasattr(X_test_model, 'values'):
                        X_test_model = X_test_model.values.astype(np.float32)
                    elif isinstance(X_test_model, np.ndarray):
                        X_test_model = X_test_model.astype(np.float32)
            
            preds = pipeline.predict(X_test_model)
            
            metrics = {}
            for metric_name in self.config['metrics']:
                if metric_name == 'accuracy':
                    metrics['Accuracy'] = accuracy_score(y_test_current, preds)
                elif metric_name == 'precision':
                    metrics['Precision'] = precision_score(y_test_current, preds, zero_division=0)
                elif metric_name == 'recall':
                    metrics['Recall'] = recall_score(y_test_current, preds, zero_division=0)
                elif metric_name == 'f1_score':
                    metrics['F1 Score'] = f1_score(y_test_current, preds, zero_division=0)
                elif metric_name == 'f1_macro':
                    metrics['F1 Macro'] = f1_score(y_test_current, preds, average='macro', zero_division=0)
                elif metric_name == 'f1_micro':
                    metrics['F1 Micro'] = f1_score(y_test_current, preds, average='micro', zero_division=0)
                elif metric_name == 'roc_auc' and hasattr(pipeline, 'predict_proba'):
                    try:
                        metrics['ROC AUC'] = roc_auc_score(y_test_current, pipeline.predict_proba(X_test_model)[:, 1])
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
    
    def save_predictions_csv(self):
        """Saves predictions of each model as individual CSV files for later reuse.
        Each CSV contains: y_true, y_pred, and y_proba (if available)."""
        predictions_dir = Path(self.config['paths']['results']['predictions_dir'])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  GUARDANDO PREDICCIONES EN CSV")
        print(f"{'='*60}\n")
        
        for model_key, pipeline in self.pipelines.items():
            # Get model name
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
            else:
                model_name = model_key
            
            # Get appropriate test data for this model
            if self._is_raw_dl_model(model_key) and self.X_test_raw is not None:
                X_test_model = self.X_test_raw.astype(np.float32)
                y_test_current = self.y_test_raw
            else:
                if model_key in self.selectors:
                    X_test_model = self.selectors[model_key].transform(self.X_test)
                else:
                    X_test_model = self.X_test
                y_test_current = self.y_test
                
                if model_key in self.config.get('dl_models', {}):
                    if hasattr(X_test_model, 'values'):
                        X_test_model = X_test_model.values.astype(np.float32)
                    elif isinstance(X_test_model, np.ndarray):
                        X_test_model = X_test_model.astype(np.float32)
            
            try:
                y_pred = pipeline.predict(X_test_model)
                
                # Build DataFrame with true labels and predictions
                y_true_arr = y_test_current.values if hasattr(y_test_current, 'values') else np.asarray(y_test_current)
                df_pred = pd.DataFrame({
                    'y_true': y_true_arr.ravel(),
                    'y_pred': np.asarray(y_pred).ravel()
                })
                
                # Add probability column if available
                if hasattr(pipeline, 'predict_proba'):
                    try:
                        y_proba = pipeline.predict_proba(X_test_model)
                        df_pred['y_proba_0'] = y_proba[:, 0]
                        df_pred['y_proba_1'] = y_proba[:, 1]
                    except Exception:
                        pass
                
                # Save CSV with model_key as filename
                csv_path = predictions_dir / f"predictions_{model_key}.csv"
                df_pred.to_csv(csv_path, index=False)
                print(f"  ✓ {model_name}: {csv_path}")
                
            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
        
        print(f"\n✓ Todas las predicciones guardadas en: {predictions_dir}\n")
    
    def plot_and_save_metrics(self, results_dict):
        """Saves metrics table as image"""
        save_path = self.config['paths']['results']['metrics']
        
        df = pd.DataFrame(results_dict).T
        
        # Format values to 4 decimals (map replaces deprecated applymap)
        df_formatted = df.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else x)
        
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1))
        ax.axis("off")
        
        table = ax.table(
            cellText=df_formatted.values,
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
        
        print(f"✓ Metrics table saved: {save_path}\n")
    
    def plot_roc_curves(self):
        """Generates ROC curves for all models on test set"""
        print(f"\n{'='*60}")
        print(f"  GENERANDO CURVAS ROC")
        print(f"{'='*60}\n")
        
        plt.figure(figsize=(10, 8))
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#c0392b']
        color_idx = 0
        
        for model_key, pipeline in self.pipelines.items():
            # Get model name
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
            else:
                model_name = model_key
            
            # Check if model has predict_proba
            if not hasattr(pipeline, 'predict_proba'):
                print(f"  [INFO] {model_name} no tiene predict_proba, saltando...")
                continue
            
            # Get appropriate data for this model
            if self._is_raw_dl_model(model_key) and self.X_test_raw is not None:
                X_test_model = self.X_test_raw.astype(np.float32)
                y_test_current = self.y_test_raw
            else:
                # Apply model-specific feature selection if exists
                if model_key in self.selectors:
                    X_test_model = self.selectors[model_key].transform(self.X_test)
                else:
                    X_test_model = self.X_test
                y_test_current = self.y_test
                
                # Convert to float32 for DL models (PyTorch requires float32)
                if model_key in self.config.get('dl_models', {}):
                    if hasattr(X_test_model, 'values'):
                        X_test_model = X_test_model.values.astype(np.float32)
                    elif isinstance(X_test_model, np.ndarray):
                        X_test_model = X_test_model.astype(np.float32)
            
            try:
                # Get probability predictions
                y_proba = pipeline.predict_proba(X_test_model)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_test_current, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                plt.plot(fpr, tpr, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})',
                        linewidth=2.5,
                        color=colors[color_idx % len(colors)])
                color_idx += 1
                
                print(f"  ✓ {model_name}: AUC = {roc_auc:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
                continue
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = Path('images/graphs/roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ ROC curves saved: {save_path}\n")
    
    def plot_confusion_matrices(self):
        """Generates confusion matrices for all models on test set"""
        print(f"\n{'='*60}")
        print(f"  GENERANDO MATRICES DE CONFUSIÓN")
        print(f"{'='*60}\n")
        
        n_models = len(self.pipelines)
        
        # Calculate grid dimensions
        n_cols = min(3, n_models)  # Max 3 columns
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Handle axes array consistently
        if n_models == 1:
            axes = [axes]  # Convert to list for consistent indexing
        else:
            axes = axes.flatten()
        
        model_idx = 0
        for model_key, pipeline in self.pipelines.items():
            # Get model name
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
            else:
                model_name = model_key
            
            # Get appropriate data for this model
            if self._is_raw_dl_model(model_key) and self.X_test_raw is not None:
                X_test_model = self.X_test_raw.astype(np.float32)
                y_test_current = self.y_test_raw
            else:
                # Apply model-specific feature selection if exists
                if model_key in self.selectors:
                    X_test_model = self.selectors[model_key].transform(self.X_test)
                else:
                    X_test_model = self.X_test
                y_test_current = self.y_test
                
                # Convert to float32 for DL models (PyTorch requires float32)
                if model_key in self.config.get('dl_models', {}):
                    if hasattr(X_test_model, 'values'):
                        X_test_model = X_test_model.values.astype(np.float32)
                    elif isinstance(X_test_model, np.ndarray):
                        X_test_model = X_test_model.astype(np.float32)
            
            try:
                # Get predictions
                y_pred = pipeline.predict(X_test_model)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test_current, y_pred)
                
                # Extract TP, TN, FP, FN
                tn, fp, fn, tp = cm.ravel()
                
                # Create display
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=['No Seizure (0)', 'Seizure (1)']
                )
                
                # Plot on specific axis
                ax = axes[model_idx]
                disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
                
                # Add title with TP, TN, FP, FN info
                title_text = f'{model_name}\nTN={tn}, FP={fp}, FN={fn}, TP={tp}'
                ax.set_title(title_text, fontsize=11, fontweight='bold')
                ax.grid(False)
                
                # Add text annotations for clarity
                # Top-left: TN (True Negative)
                ax.text(0, 0, f'\nTN', ha='center', va='top', fontsize=8, color='darkblue', weight='bold')
                # Top-right: FP (False Positive)
                ax.text(1, 0, f'\nFP', ha='center', va='top', fontsize=8, color='darkred', weight='bold')
                # Bottom-left: FN (False Negative)
                ax.text(0, 1, f'\nFN', ha='center', va='top', fontsize=8, color='darkred', weight='bold')
                # Bottom-right: TP (True Positive)
                ax.text(1, 1, f'\nTP', ha='center', va='top', fontsize=8, color='darkblue', weight='bold')
                
                # Print metrics to console
                print(f"{model_name}:")
                print(f"  TN={tn} (True Negative - Predijo No Seizure correctamente)")
                print(f"  FP={fp} (False Positive - Predijo Seizure incorrectamente)")
                print(f"  FN={fn} (False Negative - No predijo Seizure cuando había)")
                print(f"  TP={tp} (True Positive - Predijo Seizure correctamente)")
                print()
                
                model_idx += 1
                
            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
                continue
        
        # Hide unused subplots
        if n_models > 1:
            for idx in range(model_idx, len(axes)):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        save_path = Path('images/graphs/confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Confusion matrices saved: {save_path}\n")
    
    def generate_xai_explanations(self):
        """Generates XAI explanations (SHAP and LIME) for all models"""
        if not self.config['xai']['enabled']:
            print("\n[XAI] Deshabilitado en config")
            return
        
        xai_config = self.config['xai']
        save_dir = Path(self.config['paths']['results']['xai_dir'])
        
        print(f"\n{'='*60}")
        print(f"  GENERANDO EXPLICACIONES XAI")
        print(f"{'='*60}\n")
        
        for model_key, pipeline in self.pipelines.items():
            # Obtener nombre del modelo (ML o DL)
            if model_key in self.config.get('models', {}):
                model_name = self.config['models'][model_key]['name']
            elif model_key in self.config.get('dl_models', {}):
                model_name = self.config['dl_models'][model_key]['name']
                print(f"[INFO] Saltando XAI para {model_name} (modelo DL, XAI requiere métodos específicos)\n")
                continue
            else:
                model_name = model_key
            
            print(f"Generando XAI para {model_name}...")
            
            # Verificar que el modelo tenga predict_proba para XAI
            if not hasattr(pipeline, 'predict_proba'):
                print(f"  [INFO] Modelo {model_name} no tiene predict_proba, saltando XAI\n")
                continue
            
            # Apply model-specific feature selection if exists
            # IMPORTANT: Don't scale here - the pipeline has its own scaler!
            if model_key in self.selectors:
                X_train_xai = self.selectors[model_key].transform(self.X_train)
                X_test_xai = self.selectors[model_key].transform(self.X_test)
                # Get selected feature names
                selected_mask = self.selectors[model_key].get_support()
                feature_names = list(self.X_train.columns[selected_mask])
            else:
                X_train_xai = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
                X_test_xai = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
                feature_names = list(self.X_test.columns)
            
            # SHAP
            if xai_config['methods']['shap']['enabled']:
                try:
                    self._generate_shap_plot(
                        model_key, model_name, pipeline,
                        X_train_xai, X_test_xai, feature_names,
                        save_dir, xai_config['methods']['shap']
                    )
                except Exception as e:
                    print(f"  [ERROR] SHAP falló: {e}")
            
            # LIME
            if xai_config['methods']['lime']['enabled']:
                try:
                    self._generate_lime_plot(
                        model_key, model_name, pipeline,
                        X_train_xai, X_test_xai, feature_names,
                        save_dir, xai_config['methods']['lime']
                    )
                except Exception as e:
                    print(f"  [ERROR] LIME falló: {e}")
    
    def _generate_shap_plot(self, model_key, model_name, pipeline, 
                           X_train_xai, X_test_xai, feature_names,
                           save_dir, shap_config):
        """Generates SHAP importance plot using the optimal explainer for each model type"""
        background_samples = shap_config['background_samples']
        top_features = shap_config['top_features']
        
        background_data = X_train_xai[:min(background_samples, len(X_train_xai))]
        
        # Get the actual estimator from the pipeline (last step)
        estimator = pipeline.steps[-1][1]
        
        # If pipeline has a scaler, pre-scale data for Tree/Linear explainers
        has_scaler = 'scaler' in dict(pipeline.steps)
        if has_scaler:
            scaler = dict(pipeline.steps)['scaler']
            X_test_scaled = scaler.transform(X_test_xai)
            background_scaled = scaler.transform(background_data)
        else:
            X_test_scaled = X_test_xai
            background_scaled = background_data
        
        # Select the best SHAP explainer for the model type
        if model_key in ('rf', 'xgb'):
            # TreeExplainer: fast and exact for tree-based models (no scaler in their pipelines)
            print(f"  Usando TreeExplainer (óptimo para {model_name})")
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test_scaled)
        elif model_key == 'lr':
            # LinearExplainer: fast and exact for linear models (needs scaled data)
            print(f"  Usando LinearExplainer (óptimo para {model_name})")
            explainer = shap.LinearExplainer(estimator, background_scaled)
            shap_values = explainer.shap_values(X_test_scaled)
        else:
            # KernelExplainer: generic but slow (SVC, KNN, etc.)
            # Uses full pipeline so data doesn't need pre-scaling
            print(f"  Usando KernelExplainer (genérico para {model_name})")
            explainer = shap.KernelExplainer(pipeline.predict_proba, background_data)
            shap_values = explainer.shap_values(X_test_xai)
        
        # Handle different SHAP value formats
        # For binary classification, shap_values can be:
        # - A list [class_0_values, class_1_values]
        # - A single array for the positive class
        # - A 3D array (n_samples, n_features, n_classes)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Use positive class (index 1)
            shap_for_positive_class = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            # Check dimensions
            if shap_values.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                # Extract class 1 (seizure)
                shap_for_positive_class = shap_values[:, :, 1]
            else:
                # Already the values for the positive class
                shap_for_positive_class = shap_values
        else:
            raise ValueError(f"Unexpected shap_values format: {type(shap_values)}")
        
        # Mean of absolute SHAP values
        mean_abs_shap = np.abs(shap_for_positive_class).mean(axis=0)
        
        # Ensure we have the right number of features
        if len(mean_abs_shap) != len(feature_names):
            print(f"  [WARNING] SHAP values length ({len(mean_abs_shap)}) != features ({len(feature_names)}), skipping...")
            return
        
        importances = pd.Series(mean_abs_shap, index=feature_names)
        importances = importances.sort_values(ascending=False).head(top_features)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="Greys")
        plt.title(f"Importancia SHAP - {model_name}")
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Características")
        plt.tight_layout()
        
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_path = save_dir / f"{model_key}_shap.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"  ✓ SHAP guardado: {plot_path}")
    
    def _generate_lime_plot(self, model_key, model_name, pipeline,
                           X_train_xai, X_test_xai, feature_names,
                           save_dir, lime_config):
        """Generates LIME importance plot"""
        n_samples = lime_config['n_samples']
        top_features = lime_config['top_features']
        
        if not hasattr(pipeline, "predict_proba"):
            print(f"  [INFO] LIME requiere predict_proba")
            return
        
        # Pipeline will handle scaling internally
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_xai,
            feature_names=feature_names,
            class_names=["No Seizure", "Seizure"],
            mode="classification",
            discretize_continuous=lime_config['discretize_continuous']
        )
        
        # Calculate average importances
        all_lime_importances = []
        num_samples = min(n_samples, len(X_test_xai))
        
        for idx in range(num_samples):
            exp = explainer.explain_instance(
                X_test_xai[idx],
                pipeline.predict_proba,
                num_features=len(feature_names),
                top_labels=1
            )
            
            # Get the label that was explained (usually the predicted class)
            # exp.available_labels() returns list of labels that were explained
            available_labels = exp.available_labels()
            if not available_labels:
                continue
            
            # Use the first available label (usually the predicted class)
            target_label = available_labels[0]
            
            # exp.as_list() returns list of tuples: [(feature_description, weight), ...]
            # feature_description can be like "feature_name <= 0.5" or just "feature_name"
            # We need to extract the actual feature name
            lime_explanation = exp.as_list(label=target_label)
            
            # Parse feature names from LIME's descriptions
            lime_weights = {}
            for feature_desc, weight in lime_explanation:
                # Extract feature name from descriptions like "feature <= value" or "feature > value"
                # We try to match with our feature_names
                matched = False
                for fname in feature_names:
                    if fname in feature_desc:
                        lime_weights[fname] = weight
                        matched = True
                        break
                
                # If no match, try to use the description as-is (in case it's just the feature name)
                if not matched:
                    lime_weights[feature_desc] = weight
            
            current_series = pd.Series(lime_weights)
            # Reindex to match all feature names, filling missing with 0
            current_series = current_series.reindex(feature_names, fill_value=0)
            all_lime_importances.append(current_series)
        
        if not all_lime_importances:
            print(f"  [ERROR] No se generaron explicaciones LIME")
            return
        
        # Average importances (signed: positive = contribuye a Seizure, negative = contribuye a No Seizure)
        avg_lime = pd.concat(all_lime_importances, axis=1).mean(axis=1)
        # Select top features by absolute value but keep the sign
        top_idx = avg_lime.abs().sort_values(ascending=False).head(top_features).index
        lime_series = avg_lime[top_idx].sort_values()  # Sort ascending for horizontal barplot
        
        # Plot with green (negative = No Seizure) and red (positive = Seizure)
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in lime_series.values]
        
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(lime_series)), lime_series.values, color=colors)
        plt.yticks(range(len(lime_series)), lime_series.index)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.title(f"Importancia LIME - {model_name}")
        plt.xlabel("Contribución media (← No Seizure | Seizure →)")
        plt.ylabel("Características")
        plt.tight_layout()
        
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_path = save_dir / f"{model_key}_lime.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"  ✓ LIME guardado: {plot_path}")
    
    def run(self):
        """Executes the complete experiment pipeline (ML, DL, or BOTH)"""
        start_time = time.time()
        start_datetime = datetime.now()
        
        print("\n" + "="*60)
        print(f"  INICIANDO EXPERIMENTO: {self.config['experiment']['name']}")
        experiment_type = self.config['experiment'].get('type', 'ml')
        print(f"  Tipo: {experiment_type.upper()}")
        print(f"  Fecha y hora: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        # Determinar flujo según tipo de experimento
        if experiment_type == 'all':
            print("\n" + "#"*60)
            print("  MODO: ENTRENAMIENTO COMPLETO")
            print("#"*60 + "\n")
            
            # 1. Cargar/extraer features (necesario para ML y para DL con features como CNN)
            self.load_or_extract_features()
            
            # 2. Selección de features
            self.select_features()
            
            # 3. Cargar datos raw para modelos de secuencia (LSTM/GRU) si hay alguno habilitado
            raw_dl_enabled = [m for m in self.get_enabled_dl_models() if self._is_raw_dl_model(m)]
            if raw_dl_enabled and PYTORCH_AVAILABLE:
                self.load_raw_data()
            
            # 4. Validación cruzada (solo para ML)
            if self.config['cross_validation']['enabled']:
                print("\n[INFO] Validación cruzada solo para modelos ML\n")
                cv_results, best_cv_model = self.cross_validate_models()
            
            # 5. Entrenar TODOS los modelos (ML + DL)
            self.train_models(force_type='all')
            
            # 5. Evaluar en validación
            val_results, best_val_model = self.evaluate_on_validation()
            
            # 6. Evaluar en test
            test_results = self.evaluate_on_test()
            
            # 7. Guardar predicciones en CSV
            self.save_predictions_csv()
            
            # 8. Guardar tabla de métricas
            self.plot_and_save_metrics(test_results)
            
            # 9. Generar curvas ROC y matrices de confusión
            self.plot_roc_curves()
            self.plot_confusion_matrices()
            
            # 10. Generar explicaciones XAI (solo para modelos ML)
            print("\n[INFO] Generando XAI solo para modelos ML (DL requiere métodos específicos)\n")
            self.generate_xai_explanations()
        
        elif experiment_type == 'dl' and self.config.get('deep_learning', {}).get('enabled', False):
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
            
            # 7. Guardar predicciones en CSV
            self.save_predictions_csv()
            
            # 8. Guardar tabla de métricas
            self.plot_and_save_metrics(test_results)
            
            # 9. Generar curvas ROC y matrices de confusión
            self.plot_roc_curves()
            self.plot_confusion_matrices()
            
            # 10. XAI (limitado para DL, requiere adaptación)
            print("\n[INFO] XAI para DL requiere métodos específicos (ej: Attention weights, Grad-CAM)")
            print("       Saltando XAI tradicional (SHAP/LIME) que es muy lento para redes neuronales\n")
        
        else:
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
            
            # 7. Guardar predicciones en CSV
            self.save_predictions_csv()
            
            # 8. Guardar tabla de métricas
            self.plot_and_save_metrics(test_results)
            
            # 9. Generar curvas ROC y matrices de confusión
            self.plot_roc_curves()
            self.plot_confusion_matrices()
            
            # 10. Generar explicaciones XAI
            self.generate_xai_explanations()
        
        # Calcular tiempo total
        end_time = time.time()
        end_datetime = datetime.now()
        duration_seconds = end_time - start_time
        duration_timedelta = timedelta(seconds=duration_seconds)
        
        # Formatear tiempo
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*60)
        print("  EXPERIMENTO COMPLETADO")
        print("="*60)
        print(f"  Inicio: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Fin:    {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duración total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"  Duración total: {duration_timedelta}")
        print("="*60 + "\n")


def main():
    """Función principal"""
    # Determine config path based on execution directory
    if os.path.exists("config.yaml"):
        config_path = "config.yaml"
    elif os.path.exists("experimentation/classic/config.yaml"):
        config_path = "experimentation/classic/config.yaml"
    else:
        raise FileNotFoundError("config.yaml not found. Run from project root or experimentation/classic/")
    
    # Crear y ejecutar experimento
    experiment = MLExperiment(config_path=config_path)
    experiment.run()


if __name__ == "__main__":
    main()
