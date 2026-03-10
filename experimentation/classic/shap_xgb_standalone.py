"""
SHAP Visualizations for XGBoost (standalone)
Loads saved pipeline + selector from joblib and generates:
  1. Beeswarm plot (top 20 features)
  2. Violin plot (top 20 features)
  3. Bar plot (mean |SHAP|)

No need to re-run ml_experiments.py
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import matplotlib.colors as mcolors


# ============================================================
# CONFIGURATION — change suffix to match your experiment
# ============================================================
SUFFIX = ""
TOP_K = 20       # Number of features for beeswarm/violin

# Paths
PIPELINE_PATH = f"data/processed/models/pipeline_xgb{SUFFIX}.joblib"
SELECTOR_PATH = f"data/processed/models/selector_xgb{SUFFIX}.joblib"
FEATURES_TEST_PATH = f"data/processed/features_test{SUFFIX}.csv"
LABELS_TEST_PATH = f"data/processed/labels_test{SUFFIX}.csv"
FEATURES_TRAIN_PATH = f"data/processed/features_train{SUFFIX}.csv"
OUTPUT_DIR = Path(f"images/xai")


def main():
    print("=" * 60)
    print("  SHAP Visualizations — XGBoost")
    print(f"  Suffix: '{SUFFIX}'")
    print("=" * 60)
    
    # Load pipeline and selector
    print(f"\nLoading pipeline: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    
    print(f"Loading selector: {SELECTOR_PATH}")
    selector = joblib.load(SELECTOR_PATH)
    
    # Load features and labels
    print(f"Loading test features: {FEATURES_TEST_PATH}")
    X_test_full = pd.read_csv(FEATURES_TEST_PATH, index_col=0)
    y_test = pd.read_csv(LABELS_TEST_PATH, index_col=0).values.ravel()
    
    print(f"Loading train features: {FEATURES_TRAIN_PATH}")
    X_train_full = pd.read_csv(FEATURES_TRAIN_PATH, index_col=0)
    
    # Apply feature selection (reorder columns to match selector's expected order)
    expected_cols = list(selector.feature_names_in_)
    X_test_full = X_test_full[expected_cols]
    X_train_full = X_train_full[expected_cols]
    
    selected_mask = selector.get_support()
    feature_names = list(X_test_full.columns[selected_mask])
    X_test = selector.transform(X_test_full)
    X_train = selector.transform(X_train_full)
    
    print(f"\nFeatures selected: {len(feature_names)}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Get XGBoost estimator from pipeline
    estimator = pipeline.steps[-1][1]
    print(f"Model: {type(estimator).__name__}")
    
    # Compute SHAP values using TreeExplainer (exact for XGBoost)
    print("\nComputing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test)
    
    # Handle format: list [class_0, class_1] or single array
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_pos = shap_values[1]  # Class 1 = Seizure
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_pos = shap_values[:, :, 1]
    else:
        shap_pos = shap_values
    
    print(f"SHAP values shape: {shap_pos.shape}")
    
    # Mean absolute SHAP
    mean_abs_shap = np.abs(shap_pos).mean(axis=0)
    
    # Top K features
    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_K]
    top_feature_names = [feature_names[i] for i in top_indices]
    shap_top = shap_pos[:, top_indices]
    X_display_top = X_test[:, top_indices]
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating beeswarm plot (top {TOP_K})...")
    plt.figure(figsize=(10, 8))
    
    cmap_inverted = matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_inverted", ["#1f77d0", "#ffffff", "#d62728"]
    )
    shap.plots.colors.red_blue = cmap_inverted 

    shap.summary_plot(
        shap_top,
        features=X_display_top,
        feature_names=top_feature_names,
        plot_type="dot",
        max_display=TOP_K,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Beeswarm — XGBoost (Top {TOP_K})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUTPUT_DIR / f"xgb_shap_beeswarm{SUFFIX}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")
    
    # Print top features summary
    print(f"\n{'='*60}")
    print(f"  TOP {TOP_K} FEATURES BY MEAN |SHAP|")
    print(f"{'='*60}")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1:2d}. {feature_names[idx]:<45s} {mean_abs_shap[idx]:.6f}")
    
    print(f"\n✓ All SHAP plots saved to: {OUTPUT_DIR}/")
#shap force plot

if __name__ == "__main__":
    main()
