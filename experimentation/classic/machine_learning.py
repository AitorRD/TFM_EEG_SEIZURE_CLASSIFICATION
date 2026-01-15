import pandas as pd
import numpy as np
import os
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb_model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Paths
file_path_train = os.path.join("data", "processed", "windowed", "dataset_windowed_train.csv")
file_path_val = os.path.join("data", "processed", "windowed", "dataset_windowed_val.csv")
file_path_test = os.path.join("data", "processed", "windowed", "dataset_windowed_test.csv")
selected_features_path = os.path.join("data", "processed", "selected_features.csv")
# image_output_path is used for metrics table, it's defined in plot_and_save_metrics function


def extract_tsfresh_features(file_path, mode="train"):
    df = pd.read_csv(file_path)
    df['Time (s)'] = df.groupby('window_id').cumcount()
    df_windowed = df.rename(columns={'window_id': 'id'})

    signal_cols = [col for col in df_windowed.columns if col not in ['id', 'Time (s)', 'Seizure', 'idSession', 'idPatient']]
    custom_fc_parameters = {
        "absolute_sum_of_changes": None,
        "mean_abs_change": None,
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "number_peaks": [{"n": 3}, {"n": 5}],
        "root_mean_square": None,
        "autocorrelation": [{"lag": 1}]
    }

    features = extract_features(
        df_windowed[['id', 'Time (s)'] + signal_cols],
        column_id="id",
        column_sort="Time (s)",
        disable_progressbar=False,
        n_jobs=multiprocessing.cpu_count(),
        default_fc_parameters=custom_fc_parameters
    )
    
    features = features.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    
    # Fit on features to learn missing value patterns
    imputer.fit(features)
    
    # Transform features and use get_feature_names_out for column names
    features_imputed = pd.DataFrame(imputer.transform(features), 
                                    columns=imputer.get_feature_names_out(features.columns), 
                                    index=features.index)

    labels = df_windowed.groupby('id')['Seizure'].max()
    # Guardar a CSV si se especifica
    if mode == "train":
        features_imputed.to_csv("data/processed/features_train.csv")
        labels.to_csv("data/processed/labels_train.csv", header=True)
    elif mode == "val":
        features_imputed.to_csv("data/processed/features_val.csv")
        labels.to_csv("data/processed/labels_val.csv", header=True)
    elif mode == "test":
        features_imputed.to_csv("data/processed/features_test.csv")
        labels.to_csv("data/processed/labels_test.csv", header=True)

    return features_imputed, labels

def select_features(X, y, k=50):
    # X is assumed to be already imputed by extract_tsfresh_features
    # Ensure k is not greater than the number of features
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    pd.Series(selected_columns, name="feature").to_csv(selected_features_path, index=False, header=True)
    return pd.DataFrame(X_selected, columns=selected_columns), selected_columns

def apply_feature_selection(X, selected_columns):
    missing = [col for col in selected_columns if col not in X.columns]
    if missing:
        print(f"\n[ERROR] Las siguientes columnas no están en X y causarán errores: {missing}")
    return X[selected_columns]

def train_and_validate_models(X_train, y_train, X_val, y_val, selected_models, random_state=42):
    results = {}
    pipelines = {}
    
    for model in selected_models:
        if model == "lr":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))
            ])
        elif model == "knn":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ])
        elif model == "svc":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(probability=True, class_weight='balanced', random_state=random_state))
            ])
        elif model == "rf":
            pipeline = Pipeline([
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state))
            ])
        elif model == "xgb":
            pipeline = Pipeline([
                ('xgb', xgb_model.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))
            ])
        else:
            print(f"Modelo no reconocido: {model}")
            continue

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        f1 = f1_score(y_val, preds)
        f1_macro = f1_score(y_val, preds, average='macro')
        results[model] = f1
        pipelines[model] = pipeline
        
    print("\nF1 en validación para cada modelo:")
    for model, f1_macro in results.items():
        print(f" - {model}: {f1_macro:.4f}")

    best_model = max(results, key=results.get)
    print(f"\nMejor modelo en validación: {best_model} (F1 Macro: {results[best_model]:.4f})")
        
    return pipelines, results, best_model

def evaluate_on_test(best_pipeline, X_test, y_test):
    preds = best_pipeline.predict(X_test)
    results = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1 Score': f1_score(y_test, preds),
        'F1 Macro': f1_score(y_test, preds, average='macro'),
        'F1 Micro': f1_score(y_test, preds, average='micro'),
        'AUC': roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1]) if hasattr(best_pipeline, 'predict_proba') else None
    }
    return results


def plot_and_save_metrics(results_dict, save_path="images/results/ml_test_metrics.png"):
    df = pd.DataFrame(results_dict).T 
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1)) 
    ax.axis("off")
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n Imagen de métricas guardada en: {save_path}")

def generate_xai_barplots(pipelines, X_test, X_train, save_dir="images/xai", top_n=10):
    os.makedirs(save_dir, exist_ok=True)

    for model_name, pipeline in pipelines.items():
        # Get the actual model from the pipeline
        final_model = None
        scaler = None
        for step_name, step_transformer in pipeline.steps:
            if isinstance(step_transformer, StandardScaler):
                scaler = step_transformer
            elif hasattr(step_transformer, 'predict_proba') or hasattr(step_transformer, 'predict'):
                final_model = step_transformer
        
        if final_model is None:
            print(f"[ERROR] No se pudo encontrar el clasificador final en el pipeline para {model_name}.")
            continue

        # Prepare scaled data for explainers (both train for background, test for explanation)
        X_train_scaled_for_explainer = scaler.transform(X_train) if scaler else X_train.values
        X_test_scaled_for_explainer = scaler.transform(X_test) if scaler else X_test.values

        # SHAP
        try:
            # Use a smaller sample of X_train for background to speed up KernelExplainer
            background_data_for_shap = X_train_scaled_for_explainer[:min(100, len(X_train_scaled_for_explainer))]

            # Use KernelExplainer for general models
            explainer = shap.KernelExplainer(pipeline.predict_proba, background_data_for_shap)
            
            # Calculate SHAP values for the test set
            shap_values = explainer.shap_values(X_test_scaled_for_explainer)
            
            # shap_values[1] is for the positive class (seizure)
            mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
            
            feature_names = X_test.columns if isinstance(X_test, pd.DataFrame) else [f"f{i}" for i in range(X_test_scaled_for_explainer.shape[1])]
            importances = pd.Series(mean_abs_shap, index=feature_names)
            
        except Exception as e:
            print(f"[INFO] Fallback para SHAP en {model_name}: {e}. Intentando feature_importances_ o coef_.")
            if hasattr(final_model, "feature_importances_"):
                importances = pd.Series(final_model.feature_importances_, index=X_test.columns)
            elif hasattr(final_model, "coef_"):
                importances = pd.Series(np.abs(final_model.coef_).flatten(), index=X_test.columns)
            else:
                print(f"No se puede generar explicabilidad para {model_name}")
                continue

        importances = importances.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="Greys")
        plt.title(f"Importancia de características (SHAP)")
        plt.xlabel("Importancia media (absoluta)")
        plt.ylabel("Características")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{model_name}_shap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[XAI] Gráfico de importancia SHAP guardado en: {plot_path}")

        # LIME
        try:
            if not hasattr(pipeline, "predict_proba"):
                print(f"[INFO] LIME no es compatible con el pipeline del modelo {model_name} (no tiene predict_proba)")
                continue

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_scaled_for_explainer, # Use scaled training data
                feature_names=feature_names, # Use feature names from X_test
                class_names=["No Seizure", "Seizure"],
                mode="classification",
                discretize_continuous=True
            )
            
            # Calculate LIME importances for a sample of test instances and average them
            all_lime_importances = []
            # Use a sample of test data for LIME, as it can be computationally intensive
            num_lime_samples = min(50, len(X_test_scaled_for_explainer))
            for idx in range(num_lime_samples):
                data_for_lime = X_test_scaled_for_explainer[idx]
                exp = explainer.explain_instance(
                    data_for_lime,
                    pipeline.predict_proba, # LIME calls predict_proba of the pipeline
                    num_features=len(feature_names), # Explain all selected features
                    top_labels=1 # Only need explanation for the predicted class
                )
                # Extract weights for the positive class (label=1)
                lime_weights = dict(exp.as_list(label=1))
                # Convert to a series with correct feature names and ensure all features are present
                current_lime_series = pd.Series(lime_weights)
                # Align with all feature names, fill missing with 0
                current_lime_series = current_lime_series.reindex(feature_names, fill_value=0)
                all_lime_importances.append(current_lime_series)

            if not all_lime_importances:
                print(f"[ERROR] No se generaron explicaciones LIME para {model_name}.")
                continue

            # Average the absolute LIME importances
            avg_abs_lime_importances = pd.concat(all_lime_importances, axis=1).abs().mean(axis=1)
            lime_series = avg_abs_lime_importances.sort_values(ascending=False).head(top_n)

            plt.figure(figsize=(8, 6))
            sns.barplot(x=lime_series.values, y=lime_series.index, palette="Greys")
            plt.title(f"Importancia LIME (Promedio) - {model_name.upper()}")
            plt.xlabel("Importancia media (absoluta)")
            plt.ylabel("Características")
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f"{model_name}_lime.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[XAI] Gráfico LIME guardado en: {plot_path}")

        except Exception as e:
            print(f"[ERROR] Fallo al generar gráfico LIME para {model_name}: {e}")
            continue

def main():
    # Rutas de los CSV procesados
    features_train_csv = "data/processed/features_train.csv"
    labels_train_csv = "data/processed/labels_train.csv"
    features_val_csv = "data/processed/features_val.csv"
    labels_val_csv = "data/processed/labels_val.csv"
    features_test_csv = "data/processed/features_test.csv"
    labels_test_csv = "data/processed/labels_test.csv"

    # TRAIN
    if os.path.exists(features_train_csv) and os.path.exists(labels_train_csv):
        print("[INFO] Cargando features y labels de TRAIN desde CSV...")
        X_train_raw = pd.read_csv(features_train_csv, index_col=0)
        y_train = pd.read_csv(labels_train_csv, index_col=0).squeeze()
    else:
        print("\nExtrayendo características de TRAIN con TSFRESH...")
        X_train_raw, y_train = extract_tsfresh_features(file_path_train, mode="train")

    X_train, selected_columns = select_features(X_train_raw, y_train, k=50)
    print("\nFeatures seleccionadas:", list(selected_columns))
    print("Shape después de selección:", X_train.shape, "y el head:", X_train.head())

    # VALIDACIÓN
    if os.path.exists(features_val_csv) and os.path.exists(labels_val_csv):
        print("[INFO] Cargando features y labels de VALIDACIÓN desde CSV...")
        X_val_raw = pd.read_csv(features_val_csv, index_col=0)
        y_val = pd.read_csv(labels_val_csv, index_col=0).squeeze()
    else:
        print("Extrayendo características de VALIDACIÓN con TSFRESH...")
        X_val_raw, y_val = extract_tsfresh_features(file_path_val, mode="val")
    X_val = apply_feature_selection(X_val_raw, selected_columns)

    # TEST
    if os.path.exists(features_test_csv) and os.path.exists(labels_test_csv):
        print("[INFO] Cargando features y labels de TEST desde CSV...")
        X_test_raw = pd.read_csv(features_test_csv, index_col=0)
        y_test = pd.read_csv(labels_test_csv, index_col=0).squeeze()
    else:
        print("Extrayendo características de TEST con TSFRESH...")
        X_test_raw, y_test = extract_tsfresh_features(file_path_test, mode="test")
    X_test = apply_feature_selection(X_test_raw, selected_columns)

    modelos = {
        "lr": "Logistic Regression",
        "knn": "K-Nearest Neighbors",
        "svc": "Support Vector Classifier",
        "rf": "Random Forest",
        "xgb": "XGBoost",
    }

    print("\nModelos disponibles:")
    for key, name in modelos.items():
        print(f" - {key}: {name}")

    selected = input("\nEscribe los modelos que deseas probar (separados por coma, ej: lr,rf,xgb): ")
    selected_models = [s.strip() for s in selected.split(",")]

    pipelines, val_results, best_model = train_and_validate_models(X_train, y_train, X_val, y_val, selected_models)
    print("\nEvaluando TODOS los modelos en TEST para comparación:")
    all_test_results = {}

    for model_name, pipeline in pipelines.items():
        test_metrics = evaluate_on_test(pipeline, X_test, y_test)
        all_test_results[model_name] = test_metrics
        print(f"\nMétricas en Test para {model_name}:")
        for metric, value in test_metrics.items():
            print(f" - {metric}: {value:.4f}")

    plot_and_save_metrics(all_test_results)
    print("\nGenerando gráficos de XAI para cada modelo...")
    generate_xai_barplots(pipelines, X_test, X_train)

if __name__ == "__main__":
    main()