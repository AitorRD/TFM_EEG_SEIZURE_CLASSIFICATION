"""
ML and DL model factory: pipeline creation (LR, RF, SVC, KNN, XGB, CNN, LSTM, GRU),
model persistence (save/load), and GPU memory management.
"""

import gc
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from .utils import get_model_path, get_selector_path, get_model_name

try:
    import torch
    import torch.nn as nn
    from skorch import NeuralNetClassifier
    from .dl_models import CNN1DClassifier, LSTMClassifier, GRUClassifier
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠ PyTorch/skorch not available. Only traditional ML models can be used.")

RAW_DL_MODELS = {'lstm', 'gru'}


def is_raw_dl_model(model_key):
    return model_key in RAW_DL_MODELS


def get_enabled_ml_models(config):
    return [k for k, v in config['models'].items() if v.get('enabled', False)]


def get_enabled_dl_models(config):
    if not PYTORCH_AVAILABLE or not config.get('deep_learning', {}).get('enabled', False):
        return []
    return [k for k, v in config.get('dl_models', {}).items() if v.get('enabled', False)]


def create_ml_pipeline(config, model_key):
    model_config = config['models'][model_key]
    default_params = model_config.get('default_params', {}).copy()
    random_state = config['experiment']['random_state']

    if model_key != "knn":
        default_params['random_state'] = random_state

    builders = {
        "lr": lambda p: Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(**p))]),
        "rf": lambda p: Pipeline([('rf', RandomForestClassifier(**p))]),
        "svc": lambda p: Pipeline([('scaler', StandardScaler()), ('svc', SVC(**p))]),
        "knn": lambda p: Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(**p))]),
        "xgb": lambda p: Pipeline([('xgb', xgb.XGBClassifier(**p))]),
    }

    if model_key not in builders:
        raise ValueError(f"Unsupported ML model: {model_key}")
    return builders[model_key](default_params)


def create_ml_pipeline_with_params(model_key, params, config):
    model_config = config['models'][model_key]
    default_params = model_config.get('default_params', {})
    random_state = config['experiment']['random_state']

    if model_key == "knn":
        all_params = {**params}
    else:
        all_params = {**default_params, **params, 'random_state': random_state}

    builders = {
        "lr": lambda p: Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(**p))]),
        "rf": lambda p: Pipeline([('rf', RandomForestClassifier(**p))]),
        "svc": lambda p: Pipeline([('scaler', StandardScaler()), ('svc', SVC(**p))]),
        "knn": lambda p: Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(**p))]),
        "xgb": lambda p: Pipeline([('xgb', xgb.XGBClassifier(**p))]),
    }

    if model_key not in builders:
        raise ValueError(f"Unsupported ML model: {model_key}")
    return builders[model_key](all_params)


def create_dl_model(config, model_key, y_train=None, **params):
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch/skorch is not available")

    model_config = config['dl_models'][model_key]
    default_params = model_config.get('default_params', {})
    dl_config = config['deep_learning']

    model_params = {**default_params, **params}

    module_map = {
        "cnn": CNN1DClassifier,
        "lstm": LSTMClassifier,
        "gru": GRUClassifier,
    }
    if model_key not in module_map:
        raise ValueError(f"Unsupported DL model: {model_key}")

    module = module_map[model_key]

    class_weights_tensor = None
    if y_train is not None:
        y_np = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        y_np = y_np.astype(np.int64)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_np), y=y_np)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    device = config['experiment'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  ⚠ CUDA not available, using CPU")

    module_params = {k: v for k, v in model_params.items() if k not in ['lr', 'weight_decay']}

    net_kwargs = {
        'module': module,
        'max_epochs': dl_config['epochs'],
        'batch_size': dl_config['batch_size'],
        'lr': params.get('lr', 1e-4),
        'optimizer': torch.optim.Adam,
        'optimizer__weight_decay': params.get('weight_decay', 0.0),
        'criterion': nn.CrossEntropyLoss,
        'iterator_train__shuffle': True,
        'train_split': None,
        'callbacks': [],
        'device': device,
        'verbose': 1,
    }

    for key, value in module_params.items():
        net_kwargs[f'module__{key}'] = value

    if class_weights_tensor is not None:
        net_kwargs['criterion__weight'] = class_weights_tensor

    return NeuralNetClassifier(**net_kwargs)


def create_dl_pipeline(config, model_key, y_train=None, **params):
    model = create_dl_model(config, model_key, y_train=y_train, **params)

    if model_key == "cnn":
        return Pipeline([('scaler', StandardScaler()), (model_key, model)])
    else:
        return Pipeline([(model_key, model)])


def save_model(config, model_key, pipeline, selectors=None, suffix=""):
    model_path = get_model_path(config, model_key, suffix)
    joblib.dump(pipeline, model_path)
    print(f"  ✓ Model saved: {model_path}")

    if selectors and model_key in selectors:
        selector_path = get_selector_path(config, model_key, suffix)
        joblib.dump(selectors[model_key], selector_path)
        print(f"  ✓ Selector saved: {selector_path}")


def load_model(config, model_key, pipelines, selectors, suffix=""):
    model_path = get_model_path(config, model_key, suffix)
    selector_path = get_selector_path(config, model_key, suffix)

    if not model_path.exists():
        return False

    try:
        pipeline = joblib.load(model_path)
        pipelines[model_key] = pipeline

        if selector_path.exists():
            selectors[model_key] = joblib.load(selector_path)

        model_name = get_model_name(config, model_key)
        print(f"  ✓ {model_name} loaded from cache: {model_path}")
        if selector_path.exists():
            print(f"  ✓ Selector loaded from cache: {selector_path}")
        return True
    except Exception as e:
        print(f"  [WARNING] Error loading model {model_key}: {e}. Will retrain.")
        return False


def free_gpu_memory():
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
