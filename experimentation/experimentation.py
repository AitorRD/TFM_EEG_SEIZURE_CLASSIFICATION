"""
Main experiment orchestrator: data loading, feature extraction/selection, cross-validation,
training (ML and DL), evaluation, prediction export, plot generation, and XAI.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from tsfresh import extract_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from .utils import (
    load_config, generate_experiment_id, resolve_n_jobs,
    create_experiment_directories, get_model_name, prepare_data_for_model,
)
from .models import (
    create_ml_pipeline, create_dl_pipeline,
    get_enabled_ml_models, get_enabled_dl_models,
    is_raw_dl_model, save_model, load_model, free_gpu_memory,
    PYTORCH_AVAILABLE,
)
from .tuning import optimize_ml, optimize_dl
from .graphs import plot_roc_curves, plot_confusion_matrices, plot_metrics_table
from .xai import generate_xai

if PYTORCH_AVAILABLE:
    import torch
    from .dl_models import EEGWindowDataset


class Experiment:

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.experiment_id = generate_experiment_id()
        self.random_state = self.config['experiment']['random_state']
        self.n_jobs = resolve_n_jobs(self.config)
        self.output_suffix = self.config['experiment'].get('output_suffix', '')

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.X_train_raw = None
        self.y_train_raw = None
        self.X_val_raw = None
        self.y_val_raw = None
        self.X_test_raw = None
        self.y_test_raw = None

        self.selected_features = None
        self.selectors = {}
        self.pipelines = {}
        self.results = {}

        create_experiment_directories(self.config, self.experiment_id)

        print(f"  Experiment ID: {self.experiment_id}")

    def extract_tsfresh_features(self, file_path, mode="train"):
        print(f"\n[{mode.upper()}] Extracting features with TSFRESH...")

        df = pd.read_csv(file_path)
        df['Time (s)'] = df.groupby('window_id').cumcount()
        df_windowed = df.rename(columns={'window_id': 'id'})

        signal_cols = [c for c in df_windowed.columns
                       if c not in ['id', 'Time (s)', 'Seizure', 'idSession', 'idPatient']]

        custom_fc = self.config['feature_extraction']['custom_fc_parameters']

        features = extract_features(
            df_windowed[['id', 'Time (s)'] + signal_cols],
            column_id="id", column_sort="Time (s)",
            disable_progressbar=False, n_jobs=self.n_jobs,
            default_fc_parameters=custom_fc,
        )

        features = features.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        imputer.fit(features)
        features_imputed = pd.DataFrame(
            imputer.transform(features),
            columns=imputer.get_feature_names_out(features.columns),
            index=features.index,
        )

        labels = df_windowed.groupby('id')['Seizure'].max()

        features_path = self.config['paths']['features'][mode]
        labels_path = self.config['paths']['labels'][mode]
        features_imputed.to_csv(features_path)
        labels.to_csv(labels_path, header=True)
        print(f"  ✓ Features saved: {features_path}")
        print(f"  ✓ Labels saved: {labels_path}")

        return features_imputed, labels

    def load_or_extract_features(self):
        datasets = {}
        for mode in ['train', 'val', 'test']:
            feat_path = self.config['paths']['features'][mode]
            lab_path = self.config['paths']['labels'][mode]
            data_path = self.config['paths']['data'][mode]

            if (os.path.exists(feat_path) and os.path.exists(lab_path)
                    and not self.config['feature_extraction']['enabled']):
                print(f"\n[{mode.upper()}] Loading features from CSV...")
                X = pd.read_csv(feat_path, index_col=0)
                y = pd.read_csv(lab_path, index_col=0).squeeze()
            else:
                X, y = self.extract_tsfresh_features(data_path, mode)

            datasets[mode] = (X, y)
            print(f"  Shape: {X.shape}")

        self.X_train, self.y_train = datasets['train']
        self.X_val, self.y_val = datasets['val']
        self.X_test, self.y_test = datasets['test']

        common = self.X_train.columns.intersection(self.X_val.columns).intersection(self.X_test.columns)
        self.X_train = self.X_train[common]
        self.X_val = self.X_val[common]
        self.X_test = self.X_test[common]
        print(f"\n[INFO] Common columns aligned: {len(common)} features")

    def load_raw_data(self):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        dl_config = self.config['deep_learning']
        print("\n[DL] Loading raw data for sequence models...")

        for mode, attr_x, attr_y in [
            ('train', 'X_train_raw', 'y_train_raw'),
            ('val', 'X_val_raw', 'y_val_raw'),
            ('test', 'X_test_raw', 'y_test_raw'),
        ]:
            df = pd.read_csv(self.config['paths']['data'][mode])
            ds = EEGWindowDataset(df, n_channels=dl_config['input_channels'],
                                  seq_len=dl_config['sequence_length'])
            setattr(self, attr_x, ds.data.numpy())
            setattr(self, attr_y, ds.labels.numpy())
            print(f"  {mode.capitalize()} shape: {getattr(self, attr_x).shape}")

    def select_features(self):
        if not self.config['feature_selection']['enabled']:
            print("\n[FEATURE SELECTION] Disabled in config")
            self.selected_features = self.X_train.columns
            return

        if self.config['optuna']['enabled']:
            print("\n[FEATURE SELECTION] With Optuna, each model will optimize its own k")
            self.selected_features = self.X_train.columns
            return

        k = min(self.config['feature_selection']['k'], self.X_train.shape[1])
        print(f"\n[FEATURE SELECTION] Selecting {k} best features...")

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(self.X_train, self.y_train)
        self.selected_features = self.X_train.columns[selector.get_support()]

        self.X_train = pd.DataFrame(selector.transform(self.X_train), columns=self.selected_features)
        self.X_val = pd.DataFrame(selector.transform(self.X_val), columns=self.selected_features)
        self.X_test = pd.DataFrame(selector.transform(self.X_test), columns=self.selected_features)

        print(f"  ✓ {len(self.selected_features)} features selected")

    def cross_validate(self):
        if not self.config['cross_validation']['enabled']:
            print("\n[CROSS VALIDATION] Disabled in config")
            return None

        cv_cfg = self.config['cross_validation']
        print(f"\n{'=' * 60}")
        print(f"  CROSS VALIDATION ({cv_cfg['n_folds']}-Fold)")
        print(f"{'=' * 60}\n")

        cv_strategy = StratifiedKFold(
            n_splits=cv_cfg['n_folds'], shuffle=cv_cfg['shuffle'],
            random_state=self.random_state,
        )

        cv_results = {}
        for model_key in get_enabled_ml_models(self.config):
            model_name = get_model_name(self.config, model_key)
            print(f"Evaluating {model_name}...")
            pipeline = create_ml_pipeline(self.config, model_key)
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv_strategy, scoring=cv_cfg['scoring'], n_jobs=self.n_jobs,
            )
            cv_results[model_key] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
            print(f"  → Score: {scores.mean():.4f} (±{scores.std():.4f})\n")

        if cv_results:
            best = max(cv_results, key=lambda x: cv_results[x]['mean'])
            print(f"🏆 Best model (CV): {get_model_name(self.config, best)} "
                  f"- Score: {cv_results[best]['mean']:.4f}\n")

        return cv_results

    def _get_model_data(self, model_key, split='train'):
        if is_raw_dl_model(model_key) and self.X_train_raw is not None:
            data = {
                'train': (self.X_train_raw, self.y_train_raw),
                'val': (self.X_val_raw, self.y_val_raw),
                'test': (self.X_test_raw, self.y_test_raw),
            }
        else:
            data = {
                'train': (self.X_train, self.y_train),
                'val': (self.X_val, self.y_val),
                'test': (self.X_test, self.y_test),
            }
        return data[split]

    def train(self):
        experiment_type = self.config['experiment'].get('type', 'ml')

        print(f"\n{'=' * 60}")
        print(f"  TRAINING ({experiment_type.upper()})")
        print(f"{'=' * 60}\n")

        dl_order = ['lstm', 'gru', 'cnn']
        ml_order = ['xgb', 'svc', 'rf', 'knn', 'lr']

        if experiment_type in ('dl', 'all'):
            if not PYTORCH_AVAILABLE:
                print("  ⚠ PyTorch/skorch not available.")
            else:
                enabled_dl = [m for m in dl_order if m in get_enabled_dl_models(self.config)]
                if enabled_dl:
                    print("\n>>> Training Deep Learning models...\n")
                    for mk in enabled_dl:
                        self._train_dl_model(mk)

        if experiment_type in ('ml', 'all'):
            enabled_ml = [m for m in ml_order if m in get_enabled_ml_models(self.config)]
            if enabled_ml:
                print("\n>>> Training Machine Learning models...\n")
                for mk in enabled_ml:
                    self._train_ml_model(mk)

    def _train_ml_model(self, model_key):
        model_name = get_model_name(self.config, model_key)
        print(f"\nTraining: {model_name}")

        if load_model(self.config, model_key, self.pipelines, self.selectors, self.output_suffix):
            return

        if self.config['optuna']['enabled']:
            _, pipeline, selector, _ = optimize_ml(
                self.config, model_key, self.X_train, self.y_train, self.n_jobs,
            )
            self.pipelines[model_key] = pipeline
            self.selectors[model_key] = selector
        else:
            pipeline = create_ml_pipeline(self.config, model_key)
            pipeline.fit(self.X_train, self.y_train)
            self.pipelines[model_key] = pipeline
            print("  ✓ Trained with default parameters")

        save_model(self.config, model_key, self.pipelines[model_key],
                   self.selectors, self.output_suffix)

    def _train_dl_model(self, model_key):
        model_name = get_model_name(self.config, model_key)
        print(f"\nTraining: {model_name}")

        if load_model(self.config, model_key, self.pipelines, self.selectors, self.output_suffix):
            return

        try:
            X_train_m, y_train_m = self._get_model_data(model_key, 'train')

            if self.config['optuna']['enabled']:
                X_val_m, y_val_m = self._get_model_data(model_key, 'val')
                _, pipeline, selector, _ = optimize_dl(
                    self.config, model_key, X_train_m, y_train_m, X_val_m, y_val_m,
                )
                self.pipelines[model_key] = pipeline
                if selector is not None:
                    self.selectors[model_key] = selector
            else:
                pipeline = create_dl_pipeline(self.config, model_key, y_train=y_train_m)
                X_np, y_np = prepare_data_for_model(X_train_m, y_train_m, as_float32=True)
                pipeline.fit(X_np, y_np)
                self.pipelines[model_key] = pipeline
                print("  ✓ Trained with default parameters")

            save_model(self.config, model_key, self.pipelines[model_key],
                       self.selectors, self.output_suffix)
        finally:
            free_gpu_memory()

    def _prepare_eval_data(self, model_key, split='test'):
        if is_raw_dl_model(model_key):
            raw_map = {'test': (self.X_test_raw, self.y_test_raw),
                       'val': (self.X_val_raw, self.y_val_raw)}
            if raw_map.get(split, (None,))[0] is not None:
                X_raw, y_raw = raw_map[split]
                return X_raw.astype(np.float32), y_raw

        data_map = {'test': (self.X_test, self.y_test), 'val': (self.X_val, self.y_val)}
        X, y = data_map[split]

        if model_key in self.selectors:
            X = self.selectors[model_key].transform(X)

        if model_key in self.config.get('dl_models', {}):
            if hasattr(X, 'values'):
                X = X.values.astype(np.float32)
            elif isinstance(X, np.ndarray):
                X = X.astype(np.float32)

        return X, y

    def evaluate_validation(self):
        print(f"\n{'=' * 60}")
        print("  VALIDATION EVALUATION")
        print(f"{'=' * 60}\n")

        val_results = {}
        for model_key, pipeline in self.pipelines.items():
            model_name = get_model_name(self.config, model_key)
            X_val, y_val = self._prepare_eval_data(model_key, 'val')
            preds = pipeline.predict(X_val)

            metrics = {
                'accuracy': accuracy_score(y_val, preds),
                'precision': precision_score(y_val, preds, zero_division=0),
                'recall': recall_score(y_val, preds, zero_division=0),
                'f1': f1_score(y_val, preds, zero_division=0),
                'f1_macro': f1_score(y_val, preds, average='macro', zero_division=0),
            }

            if hasattr(pipeline, 'predict_proba'):
                try:
                    metrics['roc_auc'] = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
                except Exception:
                    metrics['roc_auc'] = None

            val_results[model_key] = metrics

            print(f"{model_name}:")
            for m, v in metrics.items():
                if v is not None:
                    print(f"  {m}: {v:.4f}")
            print()

        if val_results:
            best = max(val_results, key=lambda x: val_results[x]['f1_macro'])
            print(f"🏆 Best model (validation): {get_model_name(self.config, best)}\n")

        return val_results

    def evaluate_test(self):
        print(f"\n{'=' * 60}")
        print("  TEST EVALUATION")
        print(f"{'=' * 60}\n")

        test_results = {}
        metric_map = {
            'accuracy': ('Accuracy', lambda y, p: accuracy_score(y, p)),
            'precision': ('Precision', lambda y, p: precision_score(y, p, zero_division=0)),
            'recall': ('Recall', lambda y, p: recall_score(y, p, zero_division=0)),
            'f1_score': ('F1 Score', lambda y, p: f1_score(y, p, zero_division=0)),
            'f1_macro': ('F1 Macro', lambda y, p: f1_score(y, p, average='macro', zero_division=0)),
            'f1_micro': ('F1 Micro', lambda y, p: f1_score(y, p, average='micro', zero_division=0)),
        }

        for model_key, pipeline in self.pipelines.items():
            model_name = get_model_name(self.config, model_key)
            X_test, y_test = self._prepare_eval_data(model_key, 'test')
            preds = pipeline.predict(X_test)

            metrics = {}
            for metric_name in self.config['metrics']:
                if metric_name in metric_map:
                    label, fn = metric_map[metric_name]
                    metrics[label] = fn(y_test, preds)
                elif metric_name == 'roc_auc' and hasattr(pipeline, 'predict_proba'):
                    try:
                        metrics['ROC AUC'] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
                    except Exception:
                        metrics['ROC AUC'] = None

            test_results[model_name] = metrics

            print(f"{model_name}:")
            for m, v in metrics.items():
                if v is not None:
                    print(f"  {m}: {v:.4f}")
            print()

        self.results['test'] = test_results
        return test_results

    def save_predictions(self):
        predictions_dir = Path(self.config['paths']['results']['predictions_dir'])
        predictions_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("  SAVING PREDICTIONS")
        print(f"{'=' * 60}\n")

        for model_key, pipeline in self.pipelines.items():
            model_name = get_model_name(self.config, model_key)
            X_test, y_test = self._prepare_eval_data(model_key, 'test')

            try:
                y_pred = pipeline.predict(X_test)
                y_true = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

                df_pred = pd.DataFrame({
                    'y_true': y_true.ravel(),
                    'y_pred': np.asarray(y_pred).ravel(),
                })

                if hasattr(pipeline, 'predict_proba'):
                    try:
                        y_proba = pipeline.predict_proba(X_test)
                        df_pred['y_proba_0'] = y_proba[:, 0]
                        df_pred['y_proba_1'] = y_proba[:, 1]
                    except Exception:
                        pass

                csv_path = predictions_dir / f"predictions_{model_key}{self.output_suffix}.csv"
                df_pred.to_csv(csv_path, index=False)
                print(f"  ✓ {model_name}: {csv_path}")

            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")

        print(f"\n✓ Predictions saved to: {predictions_dir}\n")

    def _build_graph_context(self):
        return {
            'config': self.config,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'X_test_raw': self.X_test_raw,
            'y_test_raw': self.y_test_raw,
            'selectors': self.selectors,
        }

    def generate_plots(self):
        ctx = self._build_graph_context()
        plot_roc_curves(self.pipelines, ctx, self.output_suffix)
        plot_confusion_matrices(self.pipelines, ctx, self.output_suffix)

        if 'test' in self.results:
            metrics_path = self.config['paths']['results']['metrics']
            plot_metrics_table(self.results['test'], metrics_path)

    def generate_xai(self):
        generate_xai(
            self.config, self.pipelines, self.selectors,
            self.X_train, self.X_test, self.y_test, self.output_suffix,
        )

    def run(self):
        start_time = time.time()
        start_dt = datetime.now()
        experiment_type = self.config['experiment'].get('type', 'ml')

        print("\n" + "=" * 60)
        print(f"  EXPERIMENT: {self.config['experiment']['name']}")
        print(f"  ID: {self.experiment_id}")
        print(f"  Type: {experiment_type.upper()}")
        print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        needs_features = experiment_type in ('ml', 'all') or (
            experiment_type == 'dl'
            and self.config['deep_learning'].get('data_format') != 'raw'
        )
        if needs_features:
            self.load_or_extract_features()
            self.select_features()

        needs_raw = experiment_type in ('dl', 'all') and PYTORCH_AVAILABLE
        if needs_raw:
            raw_models = [m for m in get_enabled_dl_models(self.config) if is_raw_dl_model(m)]
            if raw_models:
                self.load_raw_data()

        if experiment_type in ('ml', 'all') and self.config['cross_validation']['enabled']:
            self.cross_validate()

        self.train()

        self.evaluate_validation()
        test_results = self.evaluate_test()

        self.save_predictions()

        self.generate_plots()

        if experiment_type in ('ml', 'all'):
            self.generate_xai()

        elapsed = time.time() - start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        end_dt = datetime.now()

        print("\n" + "=" * 60)
        print("  EXPERIMENT COMPLETED")
        print("=" * 60)
        print(f"  ID: {self.experiment_id}")
        print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {int(h)}h {int(m)}m {int(s)}s")
        print("=" * 60 + "\n")

        return test_results
