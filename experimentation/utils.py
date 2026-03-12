"""
Shared utilities for the experimentation framework: config loading,
experiment ID generation, data preparation helpers, and model name resolution.
"""

import os
import yaml
import multiprocessing
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_experiment_id():
    return datetime.now().strftime("exp_%Y%m%d_%H%M%S")


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from: {config_path}")
    print(f"  Experiment: {config['experiment']['name']}\n")
    return config


def resolve_n_jobs(config):
    n_jobs_config = config['experiment'].get('n_jobs', -1)
    if n_jobs_config == -1:
        return multiprocessing.cpu_count()
    return max(1, n_jobs_config)


def create_experiment_directories(config, experiment_id):
    base_dirs = [
        Path(config['paths']['results']['xai_dir']),
        Path(config['paths']['results']['metrics']).parent,
        Path('images/graphs'),
        Path(config['paths']['results']['predictions_dir']),
        Path(config['paths']['models_dir']),
    ]
    for directory in base_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_name(config, model_key):
    if model_key in config.get('models', {}):
        return config['models'][model_key]['name']
    elif model_key in config.get('dl_models', {}):
        return config['dl_models'][model_key]['name']
    return model_key


def prepare_data_for_model(X, y, as_float32=False):
    if hasattr(X, 'values'):
        X_np = X.values
    else:
        X_np = np.asarray(X)

    if hasattr(y, 'values'):
        y_np = y.values
    else:
        y_np = np.asarray(y)

    if as_float32:
        X_np = X_np.astype(np.float32)

    y_np = y_np.astype(np.int64)
    return X_np, y_np


def get_model_path(config, model_key, suffix=""):
    models_dir = Path(config['paths']['models_dir'])
    return models_dir / f"pipeline_{model_key}{suffix}.joblib"


def get_selector_path(config, model_key, suffix=""):
    models_dir = Path(config['paths']['models_dir'])
    return models_dir / f"selector_{model_key}{suffix}.joblib"
