"""
Hyperparameter optimization with Optuna for ML and DL models,
including k-feature optimization via SelectKBest.
"""

import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

from .models import (
    create_ml_pipeline_with_params, create_dl_pipeline,
    is_raw_dl_model, free_gpu_memory, PYTORCH_AVAILABLE,
)
from .utils import prepare_data_for_model

if PYTORCH_AVAILABLE:
    import torch


def _suggest_params(trial, search_space):
    params = {}
    for param_name, param_config in search_space.items():
        param_type = param_config['type']
        if param_type == 'loguniform':
            params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
        elif param_type == 'uniform':
            params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
        elif param_type == 'int':
            params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
    return params


def _create_study(config, study_name):
    optuna_config = config['optuna']
    return optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config['experiment']['random_state']),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=optuna_config['n_startup_trials'],
            n_warmup_steps=optuna_config['n_warmup_steps'],
        ),
        study_name=study_name,
    )


def _run_study(config, study, objective):
    optuna_config = config['optuna']
    study.optimize(
        objective,
        n_trials=optuna_config['n_trials'],
        timeout=optuna_config.get('timeout'),
        show_progress_bar=optuna_config['show_progress_bar'],
    )

    print("\n✓ Optimization completed!")
    print(f"  → Best Score: {study.best_value:.4f}")
    print(f"  → Trial #: {study.best_trial.number}")
    print("  → Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"      {param}: {value}")
    print()

    return study


def optimize_ml(config, model_key, X_train, y_train, n_jobs=1):
    model_config = config['models'][model_key]
    model_name = model_config['name']
    search_space = model_config['optuna_search_space']

    print(f"\n{'=' * 60}")
    print(f"  OPTUNA OPTIMIZATION - {model_name}")
    print(f"  Trials: {config['optuna']['n_trials']}")
    print(f"{'=' * 60}\n")

    cv_config = config['cross_validation']
    cv_strategy = StratifiedKFold(
        n_splits=cv_config['n_folds'],
        shuffle=cv_config['shuffle'],
        random_state=config['experiment']['random_state'],
    )

    def objective(trial):
        k_max = min(200, X_train.shape[1])
        k_min = max(10, int(k_max * 0.1))
        k_features = trial.suggest_int('k_features', k_min, k_max)

        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_selected = selector.fit_transform(X_train, y_train)

        params = _suggest_params(trial, search_space)
        pipeline = create_ml_pipeline_with_params(model_key, params, config)

        scores = cross_val_score(
            pipeline, X_selected, y_train,
            cv=cv_strategy, scoring=cv_config['scoring'], n_jobs=n_jobs,
        )
        return scores.mean()

    study = _create_study(config, f'{model_key}_optimization')
    _run_study(config, study, objective)

    best_params = study.best_params.copy()
    k_features = best_params.pop('k_features')

    print(f"  → Creating selector with k={k_features} features")
    selector = SelectKBest(score_func=f_classif, k=k_features)
    selector.fit(X_train, y_train)
    X_selected = selector.transform(X_train)

    best_pipeline = create_ml_pipeline_with_params(model_key, best_params, config)
    best_pipeline.fit(X_selected, y_train)
    print(f"  ✓ Model trained with {k_features} selected features")

    return best_params, best_pipeline, selector, study


def optimize_dl(config, model_key, X_train, y_train, X_val, y_val):
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch/skorch not available")

    model_config = config['dl_models'][model_key]
    model_name = model_config['name']
    search_space = model_config['optuna_search_space']

    print(f"\n{'=' * 60}")
    print(f"  OPTUNA OPTIMIZATION - {model_name}")
    print(f"  Trials: {config['optuna']['n_trials']}")
    print(f"{'=' * 60}\n")

    use_feature_selection = not is_raw_dl_model(model_key) and config['feature_selection']['enabled']

    def objective(trial):
        if use_feature_selection:
            k_max = min(200, X_train.shape[1])
            k_min = max(10, int(k_max * 0.1))
            k_features = trial.suggest_int('k_features', k_min, k_max)

            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_tr = selector.fit_transform(X_train, y_train)
            X_v = selector.transform(X_val)
        else:
            X_tr = X_train
            X_v = X_val

        params = _suggest_params(trial, search_space)

        if model_key == 'cnn':
            params['input_features'] = k_features if use_feature_selection else (
                X_tr.shape[1] if hasattr(X_tr, 'shape') else len(X_tr[0])
            )

        pipeline = create_dl_pipeline(config, model_key, y_train=y_train, **params)

        X_tr_np, y_tr_np = prepare_data_for_model(X_tr, y_train, as_float32=True)
        X_v_np, y_v_np = prepare_data_for_model(X_v, y_val, as_float32=True)

        pipeline.fit(X_tr_np, y_tr_np)
        y_pred = pipeline.predict(X_v_np)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        del pipeline
        free_gpu_memory()
        return f1

    study = _create_study(config, f'{model_key}_dl_optimization')
    _run_study(config, study, objective)

    best_params = study.best_params.copy()
    selector = None

    if use_feature_selection and 'k_features' in best_params:
        k_features = best_params.pop('k_features')
        print(f"  → Creating selector with k={k_features} features")
        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(X_train, y_train)
        X_tr_final = selector.transform(X_train)
    else:
        X_tr_final = X_train

    if model_key == 'cnn':
        if selector is not None:
            best_params['input_features'] = k_features
        else:
            best_params['input_features'] = X_tr_final.shape[1] if hasattr(X_tr_final, 'shape') else len(X_tr_final[0])

    best_pipeline = create_dl_pipeline(config, model_key, y_train=y_train, **best_params)
    X_tr_np, y_tr_np = prepare_data_for_model(X_tr_final, y_train, as_float32=True)
    best_pipeline.fit(X_tr_np, y_tr_np)

    if selector is not None:
        print(f"  ✓ Model trained with {k_features} selected features")

    return best_params, best_pipeline, selector, study
