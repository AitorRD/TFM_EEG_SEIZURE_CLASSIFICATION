"""
ML model factory: pipeline creation (LR, RF, SVC, KNN, XGB, TabPFN, TabICL),
model persistence (save/load).
"""

import gc
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False

from .utils import get_model_path, get_selector_path, get_model_name


def get_enabled_ml_models(config):
    return [k for k, v in config['models'].items() if v.get('enabled', False)]


def _make_pipeline(model_key, params):
    if model_key == "tabpfn" and not TABPFN_AVAILABLE:
        raise ImportError("tabpfn is not installed. Run: pip install tabpfn")
    if model_key == "tabicl" and not TABICL_AVAILABLE:
        raise ImportError("tabicl is not installed. Run: pip install tabicl")

    builders = {
        "lr":     lambda p: Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(**p))]),
        "rf":     lambda p: Pipeline([('rf', RandomForestClassifier(**p))]),
        "svc":    lambda p: Pipeline([('scaler', StandardScaler()), ('svc', SVC(**p))]),
        "knn":    lambda p: Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(**p))]),
        "xgb":    lambda p: Pipeline([('xgb', xgb.XGBClassifier(**p))]),
        "tabpfn": lambda p: Pipeline([('scaler', StandardScaler()), ('tabpfn', TabPFNClassifier(**p))]),
        "tabicl": lambda p: Pipeline([('scaler', StandardScaler()), ('tabicl', TabICLClassifier(**p))]),
    }

    if model_key not in builders:
        raise ValueError(f"Unsupported ML model: {model_key}")
    return builders[model_key](params)


def create_ml_pipeline(config, model_key):
    model_config = config['models'][model_key]
    params = model_config.get('default_params', {}).copy()
    random_state = config['experiment']['random_state']

    if model_key in {"lr", "rf", "svc", "xgb", "tabicl"}:
        params['random_state'] = random_state

    return _make_pipeline(model_key, params)


def create_ml_pipeline_with_params(model_key, params, config):
    model_config = config['models'][model_key]
    default_params = model_config.get('default_params', {})
    random_state = config['experiment']['random_state']

    if model_key in {"knn", "tabicl"}:
        all_params = {**params}
    elif model_key == "tabpfn":
        all_params = {**default_params, **params}
    else:
        all_params = {**default_params, **params, 'random_state': random_state}

    return _make_pipeline(model_key, all_params)


def save_model(config, model_key, pipeline, selectors=None, suffix=""):
    model_path = get_model_path(config, model_key, suffix)
    joblib.dump(pipeline, model_path)
    print(f"  ✓ Model saved: {model_path}")

    if selectors and model_key in selectors:
        selector_path = get_selector_path(config, model_key, suffix)
        joblib.dump(selectors[model_key], selector_path)
        print(f"  ✓ Selector saved: {selector_path}")


class _ColumnSelector:
    """Selects a fixed list of columns by name from a DataFrame."""

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].values
        return X

    def get_support(self):
        return self.feature_names


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
            print(f"  ✓ Selector loaded from cache: {selector_path}")
        else:
            first_estimator = pipeline.steps[0][1] if hasattr(pipeline, 'steps') else None
            if first_estimator is not None and hasattr(first_estimator, 'feature_names_in_'):
                selector = _ColumnSelector(first_estimator.feature_names_in_)
                selectors[model_key] = selector
                joblib.dump(selector, selector_path)
                print(f"  ✓ Selector reconstructed and saved: {selector_path}")

        model_name = get_model_name(config, model_key)
        print(f"  ✓ {model_name} loaded from cache: {model_path}")
        return True
    except Exception as e:
        print(f"  [WARNING] Error loading model {model_key}: {e}. Will retrain.")
        return False
