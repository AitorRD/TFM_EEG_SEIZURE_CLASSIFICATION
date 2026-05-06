"""
Chronos2 / Moirai2 classification over windowed EEG CSV files.

Workflow:
1) Read windowed train/val/test CSV files.
2) Build one univariate time series per EEG window (selected channel).
3) Run zero-shot forecasting with Chronos2 and/or Moirai2 on each window.
4) Convert forecast error (MAE over prediction horizon) into seizure scores.
5) Tune threshold on validation split and evaluate classification on test split.
6) Save predictions CSV and a metrics table image with model names.
7) (Optional) Run post-hoc XAI (SHAP/LIME over the trained foundation model + embedded TSMixer saliency).

Examples:
    python foundation_models.py --models chronos2 moirai2 --channel "EEG F3"
    python foundation_models.py --model chronos2 --channel "EEG C3" --max-windows 256
    python foundation_models.py --model tsmixer --xai-enabled --xai-methods shap lime embedded
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

from experimentation.graphs import plot_metrics_table


MODEL_DISPLAY_NAMES = {
    "chronos2": "Chronos2",
    "moirai2": "Moirai2",
    "tsmixer": "TSMixer (Google-style)",
}


@dataclass
class SplitData:
    split: str
    window_ids: List[str]
    series: Dict[str, np.ndarray]
    labels: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronos2 / Moirai2 seizure classification over EEG windowed CSV",
    )

    parser.add_argument(
        "--model",
        choices=["chronos2", "moirai2", "tsmixer"],
        default=None,
        help="Run a single model",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["chronos2", "moirai2", "tsmixer"],
        default=None,
        help="Run multiple models (overrides --model)",
    )

    parser.add_argument("--channel", type=str, default="EEG F3")
    parser.add_argument("--context-length", type=int, default=800)
    parser.add_argument("--prediction-length", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument(
        "--window-order",
        choices=["first", "random"],
        default="first",
        help="How to choose windows when --max-windows truncates each split",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)

    parser.add_argument(
        "--metrics-path",
        type=str,
        default="images/results/chronos_moirai_metrics.png",
        help="Output path for metrics image/table",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="images/results/csv",
        help="Directory to save per-model predictions CSV",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=str,
        default="images/graphs/confusion_matrices_chronos_moirai.png",
        help="Output path for confusion matrices image",
    )
    parser.add_argument(
        "--roc-curves-path",
        type=str,
        default="images/graphs/roc_curves_chronos_moirai.png",
        help="Output path for ROC curves image",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device map for Chronos2 (cuda/cpu)",
    )
    parser.add_argument(
        "--chronos-model-id",
        type=str,
        default="amazon/chronos-2",
        help="Hugging Face model id for Chronos2",
    )
    parser.add_argument(
        "--moirai2-model-id",
        type=str,
        default="Salesforce/moirai-2.0-R-small",
        help="Hugging Face model id for Moirai2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--tsmixer-epochs",
        type=int,
        default=20,
        help="Training epochs for TSMixer",
    )
    parser.add_argument(
        "--tsmixer-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for TSMixer optimizer",
    )
    parser.add_argument(
        "--tsmixer-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for TSMixer blocks",
    )
    parser.add_argument(
        "--tsmixer-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for TSMixer MLP blocks",
    )
    parser.add_argument(
        "--tsmixer-blocks",
        type=int,
        default=3,
        help="Number of mixer blocks in TSMixer",
    )
    parser.add_argument(
        "--tsmixer-val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train windows used as internal validation for TSMixer",
    )
    parser.add_argument(
        "--tsmixer-device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Execution device for TSMixer (cpu disables CUDA before TensorFlow import)",
    )
    parser.add_argument(
        "--xai-enabled",
        action="store_true",
        help="Enable post-hoc XAI plots for foundation models",
    )
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        default=["shap", "lime", "embedded"],
        choices=["shap", "lime", "embedded"],
        help="XAI methods to run (subset of: shap, lime, embedded)",
    )
    parser.add_argument(
        "--xai-dir",
        type=str,
        default="images/xai/foundation",
        help="Output directory for XAI plots",
    )
    parser.add_argument(
        "--xai-segments",
        type=int,
        default=20,
        help="Number of context segments used by SHAP/LIME perturbations",
    )
    parser.add_argument(
        "--xai-max-samples",
        type=int,
        default=256,
        help="Max windows used for direct SHAP/LIME explainability",
    )
    parser.add_argument(
        "--xai-top-features",
        type=int,
        default=10,
        help="Top features displayed in SHAP/LIME plots",
    )
    parser.add_argument(
        "--xai-lime-instances",
        type=int,
        default=24,
        help="Number of test windows explained by SHAP/LIME",
    )
    parser.add_argument(
        "--xai-lime-perturbations",
        type=int,
        default=1000,
        help="Synthetic samples used by each LIME local explanation",
    )
    parser.add_argument(
        "--xai-shap-background",
        type=int,
        default=24,
        help="Background windows used by SHAP KernelExplainer",
    )
    parser.add_argument(
        "--xai-shap-nsamples",
        type=int,
        default=128,
        help="Kernel SHAP evaluation budget per explained window",
    )

    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return list(dict.fromkeys(args.models))
    if args.model:
        return [args.model]
    return ["chronos2", "moirai2", "tsmixer"]


def get_default_split_path(split: str) -> str:
    return f"data/processed/windowed/dataset_windowed_{split}.csv"


def resolve_split_path(args: argparse.Namespace, split: str) -> str:
    custom = getattr(args, f"{split}_csv")
    return custom if custom else get_default_split_path(split)


def load_split_data(
    split: str,
    csv_path: str,
    channel: str,
    context_length: int,
    prediction_length: int,
    max_windows: int,
    window_order: str,
    seed: int,
) -> SplitData:
    df = pd.read_csv(csv_path)

    required_cols = {"window_id", "Seizure", channel}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"[{split}] Missing required columns: {sorted(missing)}")

    grouped = df.groupby("window_id", sort=False)
    window_ids = list(grouped.groups.keys())

    if window_order == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(window_ids)

    if max_windows > 0:
        window_ids = window_ids[:max_windows]

    min_len = context_length + prediction_length
    series: Dict[str, np.ndarray] = {}
    labels: Dict[str, int] = {}

    for window_id in window_ids:
        g = grouped.get_group(window_id)
        values = g[channel].to_numpy(dtype=np.float32)

        if len(values) < min_len:
            continue

        series[str(window_id)] = values[:min_len]
        labels[str(window_id)] = int(g["Seizure"].max())

    final_ids = [wid for wid in window_ids if wid in series]
    if not final_ids:
        raise RuntimeError(
            f"[{split}] No valid windows found. "
            f"Try reducing context/prediction length or increasing max windows."
        )

    print(
        f"[{split.upper()}] Loaded {len(final_ids)} windows from {csv_path} "
        f"(channel={channel}, ctx={context_length}, pred={prediction_length})"
    )

    return SplitData(split=split, window_ids=final_ids, series=series, labels=labels)


def _get_pred_column(df_pred: pd.DataFrame) -> str:
    for candidate in ["predictions", "prediction", "mean", "0.5"]:
        if candidate in df_pred.columns:
            return candidate

    numeric_cols = [
        c for c in df_pred.columns if np.issubdtype(df_pred[c].dtype, np.number)
    ]
    if not numeric_cols:
        raise RuntimeError("Could not find numeric prediction column in model output")
    return numeric_cols[0]


def _build_context_df(series_dict: Dict[str, np.ndarray], context_length: int) -> pd.DataFrame:
    rows: List[dict] = []
    for sid, values in series_dict.items():
        context = values[:context_length]
        rows.extend(
            {
                "id": sid,
                "timestamp": t,
                "target": float(v),
            }
            for t, v in enumerate(context)
        )
    return pd.DataFrame(rows)


class Chronos2Forecaster:
    def __init__(
        self,
        model_id: str,
        context_length: int,
        prediction_length: int,
        device: str,
    ):
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is not installed. Run: pip install chronos-forecasting"
            ) from exc

        resolved_device = str(device).lower().strip()
        if resolved_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("  [INFO] CUDA requested for Chronos2 but not available. Falling back to CPU.")
                    resolved_device = "cpu"
            except Exception:
                print("  [INFO] Could not validate CUDA availability. Falling back to CPU for Chronos2.")
                resolved_device = "cpu"

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=resolved_device,
        )

    def forecast(self, series_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        context_df = _build_context_df(series_dict, self.context_length)

        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=self.prediction_length,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            quantile_levels=[0.5],
        )

        pred_col = _get_pred_column(pred_df)
        preds_by_id: Dict[str, np.ndarray] = {}

        for sid, g in pred_df.groupby("id", sort=False):
            pred_values = g.sort_values("timestamp")[pred_col].to_numpy(dtype=np.float32)
            if len(pred_values) >= self.prediction_length:
                preds_by_id[str(sid)] = pred_values[: self.prediction_length]

        return preds_by_id


class Moirai2Forecaster:
    def __init__(
        self,
        model_id: str,
        context_length: int,
        prediction_length: int,
        batch_size: int,
    ):
        try:
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.dataset.split import split
        except ImportError as exc:
            raise ImportError(
                "gluonts is not installed. Run: pip install gluonts"
            ) from exc

        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError:
            try:
                from uni2ts.model.moirai import Moirai2Forecast, Moirai2Module
            except ImportError as exc:
                raise ImportError(
                    "uni2ts with Moirai2 is not installed. Run: pip install uni2ts"
                ) from exc

        self.PandasDataset = PandasDataset
        self.split = split
        self.Moirai2Forecast = Moirai2Forecast

        self.module = Moirai2Module.from_pretrained(model_id)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size

    def forecast(self, series_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ordered_ids = list(series_dict.keys())
        wide_df = pd.DataFrame({sid: series_dict[sid] for sid in ordered_ids})
        wide_df.index = pd.date_range(
            start="2000-01-01",
            periods=len(wide_df),
            freq="10ms",
        )

        ds = self.PandasDataset(dict(wide_df))

        _, test_template = self.split(ds, offset=-self.prediction_length)
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=1,
            distance=self.prediction_length,
        )

        model = self.Moirai2Forecast(
            module=self.module,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=getattr(ds, "num_feat_dynamic_real", 0),
            past_feat_dynamic_real_dim=getattr(ds, "num_past_feat_dynamic_real", 0),
        )
        predictor = model.create_predictor(batch_size=self.batch_size)

        inputs = list(test_data.input)
        forecasts = list(predictor.predict(inputs))

        preds_by_id: Dict[str, np.ndarray] = {}
        for idx, forecast in enumerate(forecasts):
            if idx >= len(ordered_ids):
                break

            sid = ordered_ids[idx]
            if hasattr(forecast, "quantile"):
                pred_values = np.asarray(forecast.quantile(0.5), dtype=np.float32)
            elif hasattr(forecast, "mean"):
                pred_values = np.asarray(forecast.mean, dtype=np.float32)
            else:
                raise RuntimeError("Unsupported forecast object returned by Moirai2")

            if len(pred_values) >= self.prediction_length:
                preds_by_id[sid] = pred_values[: self.prediction_length]

        return preds_by_id


class TSMixerForecaster:
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        dropout: float,
        hidden_dim: int,
        num_blocks: int,
        val_ratio: float,
        seed: int,
        device: str = "cpu",
    ):
        self.device = device
        if self.device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from exc

        self.tf = tf
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.val_ratio = max(0.0, min(0.5, val_ratio))
        self.seed = seed
        self.model = None
        self._is_fitted = False

    def _build_model(self):
        tf = self.tf

        def mixer_block(x):
            # Time-mixing MLP over sequence dimension.
            y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            y = tf.keras.layers.Permute((2, 1))(y)
            y = tf.keras.layers.Dense(self.hidden_dim, activation="gelu")(y)
            y = tf.keras.layers.Dropout(self.dropout)(y)
            y = tf.keras.layers.Dense(self.context_length)(y)
            y = tf.keras.layers.Permute((2, 1))(y)
            x = tf.keras.layers.Add()([x, y])

            # Feature-mixing MLP over channel dimension.
            y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            y = tf.keras.layers.Dense(self.hidden_dim, activation="gelu")(y)
            y = tf.keras.layers.Dropout(self.dropout)(y)
            y = tf.keras.layers.Dense(1)(y)
            return tf.keras.layers.Add()([x, y])

        inputs = tf.keras.Input(shape=(self.context_length, 1))
        x = inputs
        for _ in range(self.num_blocks):
            x = mixer_block(x)

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation="gelu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(self.prediction_length)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mae",
        )
        return model

    def _to_xy(self, series_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        ids: List[str] = []
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        min_len = self.context_length + self.prediction_length
        for sid, values in series_dict.items():
            arr = np.asarray(values, dtype=np.float32)
            if len(arr) < min_len:
                continue
            x = arr[: self.context_length]
            y = arr[self.context_length : self.context_length + self.prediction_length]
            xs.append(x[:, None])
            ys.append(y)
            ids.append(sid)

        if not xs:
            raise RuntimeError("TSMixer found no valid windows to train/predict")

        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), ids

    def fit(self, series_dict: Dict[str, np.ndarray]) -> None:
        tf = self.tf
        tf.keras.utils.set_random_seed(self.seed)

        X, y, _ = self._to_xy(series_dict)
        self.model = self._build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ]

        if len(X) > 20 and self.val_ratio > 0.0:
            self.model.fit(
                X,
                y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.val_ratio,
                shuffle=True,
                verbose=0,
                callbacks=callbacks,
            )
        else:
            self.model.fit(
                X,
                y,
                batch_size=self.batch_size,
                epochs=max(1, min(self.epochs, 8)),
                shuffle=True,
                verbose=0,
            )

        self._is_fitted = True

    def forecast(self, series_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("TSMixer model is not fitted. Call fit() first.")

        X, _, ids = self._to_xy(series_dict)
        preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        preds_by_id: Dict[str, np.ndarray] = {}
        for sid, pred in zip(ids, preds):
            pred_values = np.asarray(pred, dtype=np.float32)
            if len(pred_values) >= self.prediction_length:
                preds_by_id[sid] = pred_values[: self.prediction_length]

        return preds_by_id

    def explain_saliency(
        self,
        series_dict: Dict[str, np.ndarray],
        max_samples: int,
    ) -> Tuple[np.ndarray, List[str]]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("TSMixer model is not fitted. Call fit() before XAI.")

        X, y, ids = self._to_xy(series_dict)
        if max_samples > 0 and len(ids) > max_samples:
            rng = np.random.default_rng(self.seed + 19)
            selected = np.sort(rng.choice(len(ids), size=max_samples, replace=False))
            X = X[selected]
            y = y[selected]
            ids = [ids[i] for i in selected]

        tf = self.tf
        inputs = tf.convert_to_tensor(X, dtype=tf.float32)
        targets = tf.convert_to_tensor(y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            preds = self.model(inputs, training=False)
            loss_per_window = tf.reduce_mean(tf.abs(preds - targets), axis=1)
            mean_loss = tf.reduce_mean(loss_per_window)

        grads = tape.gradient(mean_loss, inputs)
        if grads is None:
            raise RuntimeError("TSMixer saliency gradient is None.")

        gradients = grads.numpy()[:, :, 0]
        mean_abs_saliency = np.mean(np.abs(gradients), axis=0).astype(np.float32)
        return mean_abs_saliency, ids


def build_scores(
    split_data: SplitData,
    preds_by_id: Dict[str, np.ndarray],
    context_length: int,
    prediction_length: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    kept_ids: List[str] = []
    y_true: List[int] = []
    scores: List[float] = []

    for sid in split_data.window_ids:
        pred = preds_by_id.get(sid)
        if pred is None:
            continue

        values = split_data.series[sid]
        y_future = values[context_length : context_length + prediction_length]
        y_pred = np.asarray(pred, dtype=np.float32)[:prediction_length]

        if len(y_future) != prediction_length or len(y_pred) != prediction_length:
            continue

        score = float(np.mean(np.abs(y_future - y_pred)))

        kept_ids.append(sid)
        y_true.append(split_data.labels[sid])
        scores.append(score)

    if not scores:
        raise RuntimeError(
            f"[{split_data.split}] No aligned predictions found after merge with labels"
        )

    return kept_ids, np.asarray(y_true, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def select_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        thr = float(np.median(scores))
        print(
            "  [WARNING] Validation has a single class. "
            f"Using median score threshold={thr:.6f}"
        )
        return thr

    quantiles = np.linspace(0.0, 1.0, 201)
    candidates = np.unique(np.quantile(scores, quantiles))

    best_thr = float(candidates[0])
    best_f1_macro = -1.0

    for thr in candidates:
        y_pred = (scores >= thr).astype(np.int64)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_thr = float(thr)

    print(f"  Threshold tuned on VAL: {best_thr:.6f} (best F1-macro={best_f1_macro:.4f})")
    return best_thr


def evaluate_classification(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float | None], np.ndarray]:
    y_pred = (scores >= threshold).astype(np.int64)

    metrics: Dict[str, float | None] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "F1 Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 Micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }

    if np.unique(y_true).size > 1:
        try:
            metrics["ROC AUC"] = float(roc_auc_score(y_true, scores))
        except Exception:
            metrics["ROC AUC"] = None
    else:
        metrics["ROC AUC"] = None

    return metrics, y_pred


def save_predictions_csv(
    model_key: str,
    window_ids: List[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    predictions_dir: Path,
) -> Path:
    predictions_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        {
            "window_id": window_ids,
            "y_true": y_true,
            "y_score": scores,
            "y_pred": y_pred,
            "threshold": threshold,
        }
    )

    csv_path = predictions_dir / f"predictions_{model_key}_foundation.csv"
    df_out.to_csv(csv_path, index=False)
    return csv_path


def _sample_indices(total: int, max_samples: int, seed: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False))


def _build_segment_features(
    split_data: SplitData,
    window_ids: List[str],
    context_length: int,
    n_segments: int,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if context_length <= 0:
        raise ValueError("context_length must be > 0 for XAI features")

    n_segments = max(1, min(n_segments, context_length))
    segment_edges = np.linspace(0, context_length, n_segments + 1, dtype=np.int64)
    feature_names = [f"s{seg_idx + 1:02d}_mean" for seg_idx in range(n_segments)]

    X = np.zeros((len(window_ids), n_segments), dtype=np.float32)
    for row_idx, window_id in enumerate(window_ids):
        context = np.asarray(
            split_data.series[window_id][:context_length],
            dtype=np.float32,
        )

        for seg_idx in range(n_segments):
            start = int(segment_edges[seg_idx])
            end = int(segment_edges[seg_idx + 1])
            if end <= start:
                end = min(context_length, start + 1)
            segment = context[start:end] if end > start else context[start : start + 1]
            if segment.size == 0:
                segment = context

            X[row_idx, seg_idx] = float(np.mean(segment))

    return X, feature_names, segment_edges


def _plot_importance_bars(
    feature_names: List[str],
    values: np.ndarray,
    title: str,
    xlabel: str,
    save_path: Path,
    top_k: int,
    signed: bool = False,
) -> None:
    if len(feature_names) == 0 or values.size == 0:
        return

    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(np.abs(values) if signed else values)[::-1]
    order = order[: min(top_k, len(order))]

    selected_features = [feature_names[i] for i in order]
    selected_values = values[order]

    if signed:
        bar_colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in selected_values]
    else:
        bar_colors = ["#34495e"] * len(selected_values)

    y_pos = np.arange(len(selected_features))
    plt.figure(figsize=(9, 6))
    plt.barh(y_pos, selected_values, color=bar_colors)
    plt.yticks(y_pos, selected_features)
    plt.gca().invert_yaxis()
    if signed:
        plt.axvline(0.0, color="black", linewidth=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  XAI plot saved: {save_path}")


def _segment_means_to_context(
    segment_means: np.ndarray,
    segment_edges: np.ndarray,
    context_length: int,
) -> np.ndarray:
    context = np.zeros(context_length, dtype=np.float32)
    n_segments = min(len(segment_means), len(segment_edges) - 1)
    for seg_idx in range(n_segments):
        start = int(segment_edges[seg_idx])
        end = int(segment_edges[seg_idx + 1])
        if end <= start:
            end = min(context_length, start + 1)
        context[start:end] = float(segment_means[seg_idx])
    return context


def _extract_future_targets(
    split_data: SplitData,
    window_ids: List[str],
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    futures = np.zeros((len(window_ids), prediction_length), dtype=np.float32)
    for row_idx, window_id in enumerate(window_ids):
        values = np.asarray(split_data.series[window_id], dtype=np.float32)
        target = values[context_length : context_length + prediction_length]
        if len(target) != prediction_length:
            raise RuntimeError(
                f"Window {window_id} has invalid future length for XAI "
                f"(expected {prediction_length}, got {len(target)})."
            )
        futures[row_idx] = target
    return futures


def _predict_scores_with_foundation_model(
    X_segment: np.ndarray,
    future_target: np.ndarray,
    forecaster,
    segment_edges: np.ndarray,
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    X_arr = np.asarray(X_segment, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    n_rows = X_arr.shape[0]
    baseline = float(np.mean(np.abs(future_target - np.mean(future_target))))
    scores = np.full(n_rows, baseline, dtype=np.float32)

    series_dict: Dict[str, np.ndarray] = {}
    for row_idx in range(n_rows):
        context = _segment_means_to_context(
            X_arr[row_idx],
            segment_edges=segment_edges,
            context_length=context_length,
        )
        series_dict[str(row_idx)] = np.concatenate(
            [context, future_target.astype(np.float32)],
            axis=0,
        ).astype(np.float32)

    preds_by_id = forecaster.forecast(series_dict)
    for row_idx in range(n_rows):
        pred = preds_by_id.get(str(row_idx))
        if pred is None:
            continue
        y_pred = np.asarray(pred, dtype=np.float32)[:prediction_length]
        if len(y_pred) != prediction_length:
            continue
        scores[row_idx] = float(np.mean(np.abs(future_target - y_pred)))

    return scores


def _choose_lime_indices(scores: np.ndarray, n_instances: int, seed: int) -> np.ndarray:
    total = len(scores)
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if n_instances <= 0 or n_instances >= total:
        return np.arange(total, dtype=np.int64)

    selected: List[int] = []
    sorted_idx = np.argsort(scores)
    anchors = [
        int(sorted_idx[0]),
        int(sorted_idx[-1]),
        int(sorted_idx[len(sorted_idx) // 2]),
    ]
    for idx in anchors:
        if idx not in selected:
            selected.append(idx)
    if len(selected) >= n_instances:
        return np.asarray(sorted(selected[:n_instances]), dtype=np.int64)

    rng = np.random.default_rng(seed + 103)
    remaining = [i for i in range(total) if i not in selected]
    needed = max(0, n_instances - len(selected))
    if needed > 0 and remaining:
        sampled = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
        selected.extend(int(i) for i in sampled)

    return np.asarray(sorted(selected), dtype=np.int64)


def _run_shap_direct(
    X: np.ndarray,
    y_targets: np.ndarray,
    scores: np.ndarray,
    forecaster,
    segment_edges: np.ndarray,
    context_length: int,
    prediction_length: int,
    feature_names: List[str],
    model_key: str,
    model_name: str,
    xai_dir: Path,
    top_features: int,
    shap_instances: int,
    shap_background: int,
    shap_nsamples: int,
    seed: int,
) -> bool:
    try:
        import shap
    except ImportError:
        print("  [INFO] SHAP not installed, skipping SHAP.")
        return False

    if len(X) < 3:
        print("  [INFO] Not enough windows for SHAP.")
        return False

    explain_idx = _choose_lime_indices(scores, shap_instances, seed=seed + 7)
    if len(explain_idx) == 0:
        print("  [INFO] SHAP had no windows to explain.")
        return False

    collected_rows: List[np.ndarray] = []
    explained_features: List[np.ndarray] = []

    for idx in explain_idx:
        pool = np.array([i for i in range(len(X)) if i != idx], dtype=np.int64)
        if len(pool) == 0:
            pool = np.array([idx], dtype=np.int64)

        bg_local_idx = _sample_indices(
            total=len(pool),
            max_samples=max(2, shap_background),
            seed=seed + int(idx) * 17,
        )
        background = X[pool[bg_local_idx]]
        future_target = y_targets[idx]

        def score_fn(data: np.ndarray) -> np.ndarray:
            return _predict_scores_with_foundation_model(
                X_segment=data,
                future_target=future_target,
                forecaster=forecaster,
                segment_edges=segment_edges,
                context_length=context_length,
                prediction_length=prediction_length,
            )

        try:
            explainer = shap.KernelExplainer(score_fn, background)
            shap_values = explainer.shap_values(
                X[idx : idx + 1],
                nsamples=max(16, shap_nsamples),
            )
            row_values = np.asarray(shap_values, dtype=np.float32).reshape(-1)
            if row_values.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Unexpected SHAP row shape {row_values.shape}, expected {X.shape[1]}."
                )
            collected_rows.append(row_values)
            explained_features.append(X[idx])
        except Exception as exc:
            print(f"  [INFO] SHAP failed for window index {idx}: {exc}")

    if not collected_rows:
        print("  [INFO] SHAP produced no usable explanations.")
        return False

    shap_matrix = np.vstack(collected_rows)
    X_explained = np.vstack(explained_features)
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)

    top_k = min(top_features, len(feature_names), shap_matrix.shape[1])
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_matrix[:, top_idx],
        features=X_explained[:, top_idx],
        feature_names=top_names,
        plot_type="dot",
        max_display=top_k,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Beeswarm - {model_name} (direct foundation model)")
    plt.tight_layout()
    beeswarm_path = xai_dir / f"{model_key}_direct_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  XAI plot saved: {beeswarm_path}")

    _plot_importance_bars(
        feature_names=feature_names,
        values=mean_abs,
        title=f"SHAP Importance - {model_name} (direct foundation model)",
        xlabel="Mean |SHAP value|",
        save_path=xai_dir / f"{model_key}_direct_shap_importance.png",
        top_k=top_features,
        signed=False,
    )
    return True


def _run_lime_direct(
    X: np.ndarray,
    y_targets: np.ndarray,
    scores: np.ndarray,
    forecaster,
    segment_edges: np.ndarray,
    context_length: int,
    prediction_length: int,
    feature_names: List[str],
    model_key: str,
    model_name: str,
    xai_dir: Path,
    top_features: int,
    lime_instances: int,
    lime_perturbations: int,
    seed: int,
) -> bool:
    try:
        import lime.lime_tabular
    except ImportError:
        print("  [INFO] LIME not installed, skipping LIME.")
        return False

    if len(X) < 2:
        print("  [INFO] Not enough windows for LIME.")
        return False

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        training_labels=scores,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=seed,
    )

    explain_idx = _choose_lime_indices(scores, lime_instances, seed=seed)
    collected: List[pd.Series] = []

    for idx in explain_idx:
        future_target = y_targets[idx]

        def score_fn(data: np.ndarray) -> np.ndarray:
            return _predict_scores_with_foundation_model(
                X_segment=data,
                future_target=future_target,
                forecaster=forecaster,
                segment_edges=segment_edges,
                context_length=context_length,
                prediction_length=prediction_length,
            )

        exp = explainer.explain_instance(
            X[idx],
            score_fn,
            num_features=min(top_features, len(feature_names)),
            num_samples=max(200, lime_perturbations),
        )

        local_weights: Dict[str, float] = {}
        for feature_desc, weight in exp.as_list():
            mapped_name = None
            for fname in feature_names:
                if fname in feature_desc:
                    mapped_name = fname
                    break
            if mapped_name is None:
                mapped_name = feature_desc
            local_weights[mapped_name] = float(weight)

        row = pd.Series(local_weights).reindex(feature_names, fill_value=0.0)
        collected.append(row)

    if not collected:
        print("  [INFO] LIME produced no explanations.")
        return False

    avg_weights = pd.concat(collected, axis=1).mean(axis=1)
    _plot_importance_bars(
        feature_names=feature_names,
        values=avg_weights.to_numpy(dtype=np.float32),
        title=f"LIME Importance - {model_name} (direct foundation model)",
        xlabel="Mean local contribution (negative to positive)",
        save_path=xai_dir / f"{model_key}_direct_lime.png",
        top_k=top_features,
        signed=True,
    )
    return True


def _run_tsmixer_embedded_xai(
    model_key: str,
    model_name: str,
    forecaster: TSMixerForecaster,
    split_data: SplitData,
    window_ids: List[str],
    context_length: int,
    xai_dir: Path,
    max_samples: int,
) -> None:
    subset_series = {window_id: split_data.series[window_id] for window_id in window_ids}
    saliency, _ = forecaster.explain_saliency(subset_series, max_samples=max_samples)

    x_axis = np.arange(1, min(context_length, len(saliency)) + 1)
    y_values = saliency[: len(x_axis)]

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, y_values, color="#c0392b", linewidth=1.7)
    plt.fill_between(x_axis, y_values, alpha=0.25, color="#e74c3c")
    plt.title(f"Embedded Saliency - {model_name}")
    plt.xlabel("Context timestep")
    plt.ylabel("Mean absolute gradient")
    plt.tight_layout()
    plot_path = xai_dir / f"{model_key}_embedded_saliency.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  XAI plot saved: {plot_path}")


def generate_foundation_xai(
    args: argparse.Namespace,
    model_key: str,
    model_name: str,
    split_data: SplitData,
    window_ids: List[str],
    scores: np.ndarray,
    forecaster,
) -> None:
    if not args.xai_enabled:
        return

    if len(window_ids) < 10:
        print(f"  [INFO] {model_name}: not enough windows for reliable XAI (need >= 10).")
        return

    xai_dir = Path(args.xai_dir)
    xai_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running XAI for {model_name}...")
    X_full, feature_names, segment_edges = _build_segment_features(
        split_data=split_data,
        window_ids=window_ids,
        context_length=args.context_length,
        n_segments=args.xai_segments,
    )
    y_full = np.asarray(scores, dtype=np.float32)
    y_targets_full = _extract_future_targets(
        split_data=split_data,
        window_ids=window_ids,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    sample_idx = _sample_indices(
        total=len(window_ids),
        max_samples=args.xai_max_samples,
        seed=args.seed + len(model_key) * 31,
    )
    X = X_full[sample_idx]
    y = y_full[sample_idx]
    y_targets = y_targets_full[sample_idx]

    if len(y) < 5:
        print(f"  [INFO] {model_name}: not enough sampled windows for XAI.")
        return

    method_set = set(args.xai_methods)
    if "shap" in method_set:
        _run_shap_direct(
            X=X,
            y_targets=y_targets,
            scores=y,
            forecaster=forecaster,
            segment_edges=segment_edges,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            feature_names=feature_names,
            model_key=model_key,
            model_name=model_name,
            xai_dir=xai_dir,
            top_features=args.xai_top_features,
            shap_instances=args.xai_lime_instances,
            shap_background=args.xai_shap_background,
            shap_nsamples=args.xai_shap_nsamples,
            seed=args.seed + 11,
        )

    if "lime" in method_set:
        _run_lime_direct(
            X=X,
            y_targets=y_targets,
            scores=y,
            forecaster=forecaster,
            segment_edges=segment_edges,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            feature_names=feature_names,
            model_key=model_key,
            model_name=model_name,
            xai_dir=xai_dir,
            top_features=args.xai_top_features,
            lime_instances=args.xai_lime_instances,
            lime_perturbations=args.xai_lime_perturbations,
            seed=args.seed + 13,
        )

    if (
        "embedded" in method_set
        and model_key == "tsmixer"
        and isinstance(forecaster, TSMixerForecaster)
    ):
        _run_tsmixer_embedded_xai(
            model_key=model_key,
            model_name=model_name,
            forecaster=forecaster,
            split_data=split_data,
            window_ids=window_ids,
            context_length=args.context_length,
            xai_dir=xai_dir,
            max_samples=args.xai_max_samples,
        )
    elif "embedded" in method_set:
        print(f"  [INFO] {model_name}: embedded explainability is only available for TSMixer.")


def plot_foundation_confusion_matrices(
    eval_outputs: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    n_models = len(eval_outputs)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (model_name, out) in enumerate(eval_outputs.items()):
        y_true = out["y_true"]
        y_pred = out["y_pred"]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model_name}\nTN={tn} FP={fp} FN={fn} TP={tp}", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Seizure", "Seizure"])
        ax.set_yticklabels(["No Seizure", "Seizure"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=12)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.colorbar(im, ax=axes[:n_models], fraction=0.046, pad=0.04)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrices image saved: {save_path}")


def plot_foundation_roc_curves(
    eval_outputs: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1.5)

    plotted = 0
    for model_name, out in eval_outputs.items():
        y_true = out["y_true"]
        y_score = out["y_score"]

        if np.unique(y_true).size < 2:
            print(f"  [INFO] {model_name}: ROC skipped (single class in test labels)")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{model_name} (AUC={roc_auc:.3f})")
        plotted += 1

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Foundation Models")
    ax.grid(True, alpha=0.3)

    if plotted > 0:
        ax.legend(loc="lower right")
    else:
        ax.text(
            0.5,
            0.5,
            "ROC curves unavailable (single class in test labels)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves image saved: {save_path}")


def create_forecaster(model_key: str, args: argparse.Namespace):
    if model_key == "chronos2":
        return Chronos2Forecaster(
            model_id=args.chronos_model_id,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            device=args.device,
        )

    if model_key == "moirai2":
        return Moirai2Forecaster(
            model_id=args.moirai2_model_id,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            batch_size=args.batch_size,
        )

    if model_key == "tsmixer":
        return TSMixerForecaster(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            batch_size=args.batch_size,
            epochs=args.tsmixer_epochs,
            learning_rate=args.tsmixer_learning_rate,
            dropout=args.tsmixer_dropout,
            hidden_dim=args.tsmixer_hidden_dim,
            num_blocks=args.tsmixer_blocks,
            val_ratio=args.tsmixer_val_ratio,
            seed=args.seed,
            device=args.tsmixer_device,
        )

    raise ValueError(f"Unsupported model key: {model_key}")


def print_metrics(model_name: str, metrics: Dict[str, float | None]) -> None:
    print(f"\n{model_name} - TEST")
    for k, v in metrics.items():
        if v is None:
            print(f"  {k}: N/A")
        else:
            print(f"  {k}: {v:.4f}")


def main() -> None:
    args = parse_args()
    models = resolve_models(args)

    print("Models to run:", ", ".join(MODEL_DISPLAY_NAMES[m] for m in models))

    train_path = resolve_split_path(args, "train")
    val_path = resolve_split_path(args, "val")
    test_path = resolve_split_path(args, "test")

    train_data = None
    if "tsmixer" in models:
        train_data = load_split_data(
            split="train",
            csv_path=train_path,
            channel=args.channel,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            max_windows=args.max_windows,
            window_order=args.window_order,
            seed=args.seed - 1,
        )

    val_data = load_split_data(
        split="val",
        csv_path=val_path,
        channel=args.channel,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        max_windows=args.max_windows,
        window_order=args.window_order,
        seed=args.seed,
    )
    test_data = load_split_data(
        split="test",
        csv_path=test_path,
        channel=args.channel,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        max_windows=args.max_windows,
        window_order=args.window_order,
        seed=args.seed + 1,
    )

    results: Dict[str, Dict[str, float | None]] = {}
    eval_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    predictions_dir = Path(args.predictions_dir)

    for model_key in models:
        model_name = MODEL_DISPLAY_NAMES[model_key]
        print("\n" + "=" * 60)
        print(f"Running {model_name}")
        print("=" * 60)

        forecaster = create_forecaster(model_key, args)

        if model_key == "tsmixer":
            print("  Fitting TSMixer on TRAIN split...")
            forecaster.fit(train_data.series)

        val_preds = forecaster.forecast(val_data.series)
        val_ids, y_val, val_scores = build_scores(
            split_data=val_data,
            preds_by_id=val_preds,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
        print(f"  VAL aligned windows: {len(val_ids)}")

        threshold = select_threshold(val_scores, y_val)

        test_preds = forecaster.forecast(test_data.series)
        test_ids, y_test, test_scores = build_scores(
            split_data=test_data,
            preds_by_id=test_preds,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
        print(f"  TEST aligned windows: {len(test_ids)}")

        metrics, y_pred = evaluate_classification(y_test, test_scores, threshold)
        results[model_name] = metrics
        eval_outputs[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_score": test_scores,
        }

        pred_path = save_predictions_csv(
            model_key=model_key,
            window_ids=test_ids,
            y_true=y_test,
            scores=test_scores,
            y_pred=y_pred,
            threshold=threshold,
            predictions_dir=predictions_dir,
        )
        generate_foundation_xai(
            args=args,
            model_key=model_key,
            model_name=model_name,
            split_data=test_data,
            window_ids=test_ids,
            scores=test_scores,
            forecaster=forecaster,
        )

        print_metrics(model_name, metrics)
        print(f"  Predictions CSV: {pred_path}")

    metrics_path = Path(args.metrics_path)
    plot_metrics_table(results, metrics_path)
    print(f"Metrics image saved: {metrics_path}")

    cm_path = Path(args.confusion_matrix_path)
    plot_foundation_confusion_matrices(eval_outputs, cm_path)

    roc_path = Path(args.roc_curves_path)
    plot_foundation_roc_curves(eval_outputs, roc_path)


if __name__ == "__main__":
    main()
