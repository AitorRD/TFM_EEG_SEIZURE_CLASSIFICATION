"""
Hyperparameter optimization with Optuna for ML models,
including k-feature optimization via SelectKBest.
"""

import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

from .models import create_ml_pipeline_with_params


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
