"""
Model explainability (XAI): SHAP (TreeExplainer, LinearExplainer, KernelExplainer),
LIME, and brain topomap visualization based on SHAP importance.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import shap
import lime
import lime.lime_tabular

from .utils import get_model_name


def generate_xai(config, pipelines, selectors, X_train, X_test, y_test, suffix=""):
    if not config['xai']['enabled']:
        print("\n[XAI] Disabled in config")
        return

    xai_config = config['xai']
    save_dir = Path(config['paths']['results']['xai_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  GENERATING XAI EXPLANATIONS")
    print(f"{'=' * 60}\n")

    for model_key, pipeline in pipelines.items():
        if model_key in config.get('dl_models', {}):
            model_name = get_model_name(config, model_key)
            print(f"[INFO] Skipping XAI for {model_name} (DL model)\n")
            continue

        if model_key not in config.get('models', {}):
            continue

        model_name = get_model_name(config, model_key)
        print(f"Generating XAI for {model_name}...")

        if not hasattr(pipeline, 'predict_proba'):
            print(f"  [INFO] {model_name} has no predict_proba, skipping XAI\n")
            continue

        if model_key in selectors:
            X_train_xai = selectors[model_key].transform(X_train)
            X_test_xai = selectors[model_key].transform(X_test)
            selected_mask = selectors[model_key].get_support()
            feature_names = list(X_train.columns[selected_mask])
        else:
            X_train_xai = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_xai = X_test.values if hasattr(X_test, 'values') else X_test
            feature_names = list(X_test.columns)

        if xai_config['methods']['shap']['enabled']:
            try:
                _generate_shap(model_key, model_name, pipeline,
                               X_train_xai, X_test_xai, feature_names,
                               save_dir, xai_config['methods']['shap'], suffix)
            except Exception as e:
                print(f"  [ERROR] SHAP failed: {e}")

        if xai_config['methods']['lime']['enabled']:
            try:
                _generate_lime(model_key, model_name, pipeline,
                               X_train_xai, X_test_xai, feature_names,
                               save_dir, xai_config['methods']['lime'], suffix)
            except Exception as e:
                print(f"  [ERROR] LIME failed: {e}")


def _generate_shap(model_key, model_name, pipeline,
                   X_train_xai, X_test_xai, feature_names,
                   save_dir, shap_config, suffix=""):
    background_samples = shap_config['background_samples']
    top_features = shap_config['top_features']

    background_data = X_train_xai[:min(background_samples, len(X_train_xai))]
    estimator = pipeline.steps[-1][1]

    has_scaler = 'scaler' in dict(pipeline.steps)
    if has_scaler:
        scaler = dict(pipeline.steps)['scaler']
        X_test_scaled = scaler.transform(X_test_xai)
        background_scaled = scaler.transform(background_data)
    else:
        X_test_scaled = X_test_xai
        background_scaled = background_data

    if model_key in ('rf', 'xgb'):
        print(f"  Using TreeExplainer (optimal for {model_name})")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test_scaled)
    elif model_key == 'lr':
        print(f"  Using LinearExplainer (optimal for {model_name})")
        explainer = shap.LinearExplainer(estimator, background_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
    else:
        print(f"  Using KernelExplainer (generic for {model_name})")
        explainer = shap.KernelExplainer(pipeline.predict_proba, background_data)
        shap_values = explainer.shap_values(X_test_xai)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_positive = shap_values[1]
    elif isinstance(shap_values, np.ndarray):
        shap_positive = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
    else:
        raise ValueError(f"Unexpected SHAP format: {type(shap_values)}")

    mean_abs_shap = np.abs(shap_positive).mean(axis=0)

    if len(mean_abs_shap) != len(feature_names):
        print(f"  [WARNING] SHAP length ({len(mean_abs_shap)}) != features ({len(feature_names)})")
        return

    X_display = X_test_xai if has_scaler else X_test_scaled
    shap_df = pd.DataFrame(X_display, columns=feature_names)

    top_k = 20
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_feature_names = [feature_names[i] for i in top_indices]
    shap_top = shap_positive[:, top_indices]
    X_display_top = shap_df[top_feature_names].values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_top, features=X_display_top, feature_names=top_feature_names,
                      plot_type="dot", max_display=top_k, show=False, plot_size=None)
    plt.title(f"SHAP Beeswarm - {model_name} (Top {top_k})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_path = save_dir / f"{model_key}_shap_beeswarm{suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP beeswarm saved: {plot_path}")

    _generate_shap_topomap(model_key, model_name, mean_abs_shap, feature_names, save_dir, suffix)

    importances = pd.Series(mean_abs_shap, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_features)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="Greys")
    plt.title(f"SHAP Importance - {model_name}")
    plt.xlabel("Mean absolute importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plot_path = save_dir / f"{model_key}_shap{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  ✓ SHAP barplot saved: {plot_path}")


def _generate_shap_topomap(model_key, model_name, mean_abs_shap, feature_names, save_dir, suffix=""):
    try:
        import mne
    except ImportError:
        print("  [INFO] MNE not available, skipping topomap")
        return

    channel_map = {
        'Fp1': 'Fp1', 'Fp2': 'Fp2',
        'F7': 'F7', 'F3': 'F3', 'Fz': 'Fz', 'F4': 'F4', 'F8': 'F8',
        'T3': 'T7', 'C3': 'C3', 'Cz': 'Cz', 'C4': 'C4', 'T4': 'T8',
        'T5': 'P7', 'P3': 'P3', 'Pz': 'Pz', 'P4': 'P4', 'T6': 'P8',
        'O1': 'O1', 'O2': 'O2',
    }

    electrode_importance = {ch: 0.0 for ch in channel_map}
    electrode_count = {ch: 0 for ch in channel_map}

    for feat_name, shap_val in zip(feature_names, mean_abs_shap):
        match = re.search(r'EEG\s+(\w+)__', feat_name)
        if match:
            ch = match.group(1)
            if ch in electrode_importance:
                electrode_importance[ch] += shap_val
                electrode_count[ch] += 1

    for ch in electrode_importance:
        if electrode_count[ch] > 0:
            electrode_importance[ch] /= electrode_count[ch]

    mne_ch_names = list(channel_map.values())
    shap_per_electrode = np.array([electrode_importance[ch] for ch in channel_map])

    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=mne_ch_names, sfreq=100, ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(7, 6))
    im, _ = mne.viz.plot_topomap(
        shap_per_electrode, info, axes=ax,
        cmap='YlOrRd', show=False, contours=6,
        sensors=True, names=mne_ch_names
    )
    for text in ax.texts:
        text.set_fontsize(7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label('Mean |SHAP| Importance', fontsize=10)
    ax.set_title(f'Brain SHAP Importance Map\n{model_name}', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    plot_path = save_dir / f"{model_key}_shap_topomap{suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP topomap saved: {plot_path}")


def _generate_lime(model_key, model_name, pipeline,
                   X_train_xai, X_test_xai, feature_names,
                   save_dir, lime_config, suffix=""):
    n_samples = lime_config['n_samples']
    top_features = lime_config['top_features']

    if not hasattr(pipeline, "predict_proba"):
        print("  [INFO] LIME requires predict_proba")
        return

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_xai, feature_names=feature_names,
        class_names=["No Seizure", "Seizure"], mode="classification",
        discretize_continuous=lime_config['discretize_continuous'],
    )

    all_lime_importances = []
    num_samples = min(n_samples, len(X_test_xai))

    for idx in range(num_samples):
        exp = explainer.explain_instance(
            X_test_xai[idx], pipeline.predict_proba,
            num_features=len(feature_names), top_labels=1,
        )

        lime_explanation = exp.as_list(label=1)
        lime_weights = {}
        for feature_desc, weight in lime_explanation:
            matched = False
            for fname in feature_names:
                if fname in feature_desc:
                    lime_weights[fname] = weight
                    matched = True
                    break
            if not matched:
                lime_weights[feature_desc] = weight

        current_series = pd.Series(lime_weights).reindex(feature_names, fill_value=0)
        all_lime_importances.append(current_series)

    if not all_lime_importances:
        print("  [ERROR] No LIME explanations generated")
        return

    avg_lime = pd.concat(all_lime_importances, axis=1).mean(axis=1)
    top_idx = avg_lime.abs().sort_values(ascending=False).head(top_features).index
    lime_series = avg_lime[top_idx].sort_values()

    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in lime_series.values]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(lime_series)), lime_series.values, color=colors)
    plt.yticks(range(len(lime_series)), lime_series.index)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.title(f"LIME Importance - {model_name}")
    plt.xlabel("Mean contribution (← No Seizure | Seizure →)")
    plt.ylabel("Features")
    plt.tight_layout()

    plot_path = save_dir / f"{model_key}_lime{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  ✓ LIME saved: {plot_path}")
