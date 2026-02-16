"""
Script rápido para probar XAI (SHAP y LIME)
"""
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("🧪 TEST RÁPIDO XAI - SHAP & LIME")
print("="*60)

# 1. Cargar datos
print("\n1️⃣ Cargando datos...")
X_train = pd.read_csv('../../data/processed/features_train_sample.csv', index_col=0)
y_train = pd.read_csv('../../data/processed/labels_train_sample.csv', index_col=0).squeeze()
X_test = pd.read_csv('../../data/processed/features_test_sample.csv', index_col=0)
y_test = pd.read_csv('../../data/processed/labels_test_sample.csv', index_col=0).squeeze()

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Seleccionar features (reducir para velocidad)
print("\n2️⃣ Seleccionando top 20 features...")
selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"   Features seleccionadas: {len(selected_features)}")

# 3. Entrenar modelo simple
print("\n3️⃣ Entrenando Logistic Regression...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(X_train_selected, y_train)
score = pipeline.score(X_test_selected, y_test)
print(f"   Accuracy en test: {score:.4f}")

# 4. Probar SHAP
print("\n4️⃣ Probando SHAP (10 muestras de background)...")
try:
    background = X_train_selected[:10]
    explainer = shap.KernelExplainer(pipeline.predict_proba, background)
    shap_values = explainer.shap_values(X_test_selected[:5])  # Solo 5 muestras de test
    
    # Handle different formats
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Binary classification: [class_0_values, class_1_values]
        shap_for_class1 = shap_values[1]
    else:
        # Single array
        shap_for_class1 = shap_values
    
    # For predict_proba output (n_samples, n_features, n_classes)
    # We want class 1 (seizure)
    if shap_for_class1.ndim == 3:
        shap_for_class1 = shap_for_class1[:, :, 1]
    
    mean_shap = np.abs(shap_for_class1).mean(axis=0)
    
    # Plot
    importances = pd.Series(mean_shap, index=selected_features)
    importances = importances.sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="Greys")
    plt.title("SHAP Feature Importance (Top 10)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    
    Path('../../images/graphs').mkdir(parents=True, exist_ok=True)
    plt.savefig('../../images/graphs/test_shap_quick.png', dpi=150)
    plt.close()
    
    print("   ✅ SHAP funcionó! Guardado en images/graphs/test_shap_quick.png")
    
except Exception as e:
    print(f"   ❌ SHAP falló: {e}")

# 5. Probar LIME
print("\n5️⃣ Probando LIME (3 instancias)...")
try:
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_selected[:50],  # Solo 50 para velocidad
        feature_names=list(selected_features),
        class_names=["No Seizure", "Seizure"],
        mode="classification"
    )
    
    # Explicar 3 instancias
    all_weights = []
    for i in range(3):
        exp = explainer.explain_instance(
            X_test_selected[i],
            pipeline.predict_proba,
            num_features=20
        )
        
        label = exp.available_labels()[0]
        weights = dict(exp.as_list(label=label))
        
        # Extract feature names
        lime_dict = {}
        for feat_desc, weight in weights.items():
            for feat in selected_features:
                if feat in feat_desc:
                    lime_dict[feat] = weight
                    break
        
        series = pd.Series(lime_dict).reindex(selected_features, fill_value=0)
        all_weights.append(series)
    
    # Average
    avg_lime = pd.concat(all_weights, axis=1).abs().mean(axis=1)
    lime_top = avg_lime.sort_values(ascending=False).head(10)
    
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=lime_top.values, y=lime_top.index, palette="Greys")
    plt.title("LIME Feature Importance (Top 10)")
    plt.xlabel("Mean |LIME weight|")
    plt.tight_layout()
    
    Path('../../images/graphs').mkdir(parents=True, exist_ok=True)
    plt.savefig('../../images/graphs/test_lime_quick.png', dpi=150)
    plt.close()
    
    print("   ✅ LIME funcionó! Guardado en images/graphs/test_lime_quick.png")
    
except Exception as e:
    print(f"   ❌ LIME falló: {e}")

print("\n" + "="*60)
print("✅ TEST COMPLETADO!")
print("Revisa images/graphs/test_shap_quick.png y test_lime_quick.png")
print("="*60)
