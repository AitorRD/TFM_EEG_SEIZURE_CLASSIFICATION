import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb

def cross_validate_models(X, y, models_dict, cv_folds=5, random_state=42):
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average='weighted')
    
    results = {}
    print(f"\n{'='*60}")
    print(f"  VALIDACIÓN CRUZADA ({cv_folds}-Fold)")
    print(f"{'='*60}\n")
    
    for model_name, pipeline in models_dict.items():
        print(f"Evaluando {model_name}...")
        scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring=scorer, n_jobs=-1)
        results[model_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"  → F1 Score: {scores.mean():.4f} (±{scores.std():.4f})")
        print(f"  → Scores por fold: {scores}\n")
    
    best_model = max(results, key=lambda x: results[x]['mean'])
    print(f"🏆 Mejor modelo (CV): {best_model} - F1: {results[best_model]['mean']:.4f}\n")
    
    return results, best_model


def create_model_objective(model_name, X_train, y_train, cv_folds=5, random_state=42):
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average='weighted')
    
    def objective(trial):
        if model_name == 'lr':
            C = trial.suggest_loguniform('C', 1e-3, 1e2)
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
            max_iter = trial.suggest_int('max_iter', 1000, 2000)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(C=C, solver=solver, max_iter=max_iter, 
                                         class_weight='balanced', random_state=random_state))
            ])
        
        elif model_name == 'rf':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 10, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            
            pipeline = Pipeline([
                ('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                             min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf,
                                             class_weight='balanced', random_state=random_state))
            ])
        
        elif model_name == 'svc':
            C = trial.suggest_loguniform('C', 1e-1, 1e2)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(C=C, kernel=kernel, gamma=gamma, probability=True,
                           class_weight='balanced', random_state=random_state))
            ])
        
        elif model_name == 'knn':
            n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric))
            ])
        
        elif model_name == 'xgb':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
            subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            
            pipeline = Pipeline([
                ('xgb', xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         learning_rate=learning_rate, subsample=subsample,
                                         colsample_bytree=colsample_bytree,
                                         use_label_encoder=False, eval_metric='logloss',
                                         random_state=random_state))
            ])
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, 
                                scoring=scorer, n_jobs=-1)
        return scores.mean()
    
    return objective


def optimize_hyperparameters_optuna(X_train, y_train, model_name, cv_folds=5, 
                                    n_trials=100, random_state=42):
    print(f"\n{'='*60}")
    print(f"  OPTIMIZACIÓN CON OPTUNA - {model_name.upper()}")
    print(f"  Trials: {n_trials} | CV Folds: {cv_folds}")
    print(f"{'='*60}\n")
    
    objective = create_model_objective(model_name, X_train, y_train, cv_folds, random_state)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'{model_name}_optimization'
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✓ Optimización completada!")
    print(f"  → Mejor F1 Score (CV): {study.best_value:.4f}")
    print(f"  → Trial #: {study.best_trial.number}")
    print(f"  → Mejores hiperparámetros:")
    for param, value in study.best_params.items():
        print(f"      {param}: {value}")
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"data/processed/optuna_results_{model_name}.csv", index=False)
    print(f"\n  → Resultados guardados en: data/processed/optuna_results_{model_name}.csv")
    
    best_params = study.best_params
    
    if model_name == 'lr':
        best_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(**best_params, class_weight='balanced', random_state=random_state))
        ])
    elif model_name == 'rf':
        best_pipeline = Pipeline([
            ('rf', RandomForestClassifier(**best_params, class_weight='balanced', random_state=random_state))
        ])
    elif model_name == 'svc':
        best_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(**best_params, probability=True, class_weight='balanced', random_state=random_state))
        ])
    elif model_name == 'knn':
        best_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**best_params))
        ])
    elif model_name == 'xgb':
        best_pipeline = Pipeline([
            ('xgb', xgb.XGBClassifier(**best_params, use_label_encoder=False, 
                                     eval_metric='logloss', random_state=random_state))
        ])
    
    best_pipeline.fit(X_train, y_train)
    
    try:
        import optuna.visualization as vis
        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)
        fig3 = vis.plot_slice(study)
        
        fig1.write_html(f"images/results/optuna_history_{model_name}.html")
        fig2.write_html(f"images/results/optuna_importance_{model_name}.html")
        fig3.write_html(f"images/results/optuna_slice_{model_name}.html")
        print(f"  → Visualizaciones guardadas en images/results/\n")
    except ImportError:
        print(f"  (Instala plotly para visualizaciones: pip install plotly)\n")
    
    return best_params, best_pipeline, study


def train_with_cv_and_tuning(X_train, y_train, X_val, y_val, selected_models=['lr', 'rf', 'svc'], 
                              cv_folds=5, tune_best=True, n_trials=100, random_state=42):
    
    pipelines = {}
    for model in selected_models:
        if model == "lr":
            pipelines[model] = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))
            ])
        elif model == "rf":
            pipelines[model] = Pipeline([
                ('rf', RandomForestClassifier(class_weight='balanced', random_state=random_state))
            ])
        elif model == "svc":
            pipelines[model] = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(probability=True, class_weight='balanced', random_state=random_state))
            ])
        elif model == "knn":
            pipelines[model] = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ])
        elif model == "xgb":
            pipelines[model] = Pipeline([
                ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))
            ])
    
    cv_results, best_model_name = cross_validate_models(X_train, y_train, pipelines, cv_folds, random_state)
    
    if tune_best:
        best_params, best_pipeline, study = optimize_hyperparameters_optuna(
            X_train, y_train,
            best_model_name,
            cv_folds=cv_folds,
            n_trials=n_trials,
            random_state=random_state
        )
        tuning_results = study.trials_dataframe()
    else:
        best_pipeline = pipelines[best_model_name]
        best_pipeline.fit(X_train, y_train)
        best_params = None
        tuning_results = None
    
    val_preds = best_pipeline.predict(X_val)
    val_metrics = {
        'accuracy': accuracy_score(y_val, val_preds),
        'precision': precision_score(y_val, val_preds, average='weighted'),
        'recall': recall_score(y_val, val_preds, average='weighted'),
        'f1': f1_score(y_val, val_preds, average='weighted')
    }
    
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN EN CONJUNTO DE VALIDACIÓN")
    print(f"{'='*60}")
    print(f"  Modelo: {best_model_name}")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}\n")
    
    results = {
        'cv_results': cv_results,
        'best_model_name': best_model_name,
        'best_params': best_params,
        'tuning_results': tuning_results,
        'val_metrics': val_metrics
    }
    
    return best_pipeline, results


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    best_pipeline, results = train_with_cv_and_tuning(
        X_train, y_train, X_val, y_val,
        selected_models=['lr', 'rf', 'svc', 'xgb'],
        cv_folds=5,
        tune_best=True,
        n_trials=100
    )
    
    print("✓ Archivo creado: ml_with_cv_and_tuning.py")