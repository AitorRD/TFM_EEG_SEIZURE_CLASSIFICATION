# Sistema Unificado de Experimentos ML/DL

Este sistema permite ejecutar experimentos de **Machine Learning tradicional** y **Deep Learning** de manera unificada, configurados mediante archivos YAML.

## 📁 Estructura de Archivos

```
experimentation/classic/
├── config.yaml              # Configuración para ML tradicional
├── config_dl.yaml           # Configuración para Deep Learning
├── ml_experiments.py        # Script unificado ML/DL
├── dl_models.py            # Arquitecturas de DL (PyTorch)
├── machine_learning.py     # (Deprecated - usar ml_experiments.py)
└── ml_with_cv_and_tuning.py # (Deprecated - usar ml_experiments.py)
```

## 🚀 Instalación de Dependencias

### Para ML tradicional:
```bash
pip install pandas numpy scikit-learn xgboost optuna tsfresh shap lime matplotlib seaborn pyyaml
```

### Para Deep Learning (adicional):
```bash
pip install torch torchvision skorch
```

## 📝 Uso

### 1. Experimentos de Machine Learning Tradicional

**Configuración:** Edita `config.yaml`

```yaml
experiment:
  type: "ml"  # Tipo de experimento

# Activar/desactivar modelos
models:
  lr:
    enabled: true  # Logistic Regression
  rf:
    enabled: true  # Random Forest
  xgb:
    enabled: true  # XGBoost
  svc:
    enabled: false # SVC (desactivado)
  knn:
    enabled: false # KNN (desactivado)

# Configuración de Optuna
optuna:
  enabled: true  # Activar optimización
  n_trials: 100

# Validación cruzada
cross_validation:
  enabled: true
  n_folds: 5
```

**Ejecutar:**
```bash
python experimentation/classic/ml_experiments.py
```

### 2. Experimentos de Deep Learning

**Configuración:** Edita `config_dl.yaml` o cambia `config.yaml`

```yaml
experiment:
  type: "dl"  # Tipo de experimento
  device: "cuda"  # o "cpu"

deep_learning:
  enabled: true
  epochs: 50
  batch_size: 32
  data_format: "raw"  # 'raw' para Transformer/LSTM, 'features' para CNN

# Activar modelo DL
dl_models:
  transformer:
    enabled: true
  lstm:
    enabled: false
  cnn:
    enabled: false
```

**Ejecutar:**
```bash
# Opción 1: Usar config_dl.yaml
python experimentation/classic/ml_experiments.py

# Modificar para usar otro config
# En ml_experiments.py, cambiar main():
# experiment = MLExperiment(config_path="experimentation/classic/config_dl.yaml")
```

## 🎯 Características Principales

### Machine Learning
- ✅ **Modelos soportados:** LR, RF, SVC, KNN, XGBoost
- ✅ **Extracción de features:** TSFRESH automática
- ✅ **Selección de features:** SelectKBest configurable
- ✅ **Validación cruzada:** K-Fold estratificada
- ✅ **Optimización:** Optuna con espacios de búsqueda personalizables
- ✅ **XAI:** SHAP y LIME para explicabilidad
- ✅ **Class weighting:** Automático para datos desbalanceados

### Deep Learning
- ✅ **Modelos soportados:** Transformer, LSTM, GRU, CNN 1D
- ✅ **Framework:** PyTorch con skorch (compatible con sklearn)
- ✅ **Early Stopping:** Detiene entrenamiento automático
- ✅ **LR Scheduling:** Reduce learning rate cuando plateaus
- ✅ **Optimización:** Optuna con hiperparámetros de DL
- ✅ **Datos raw:** Trabaja directamente con ventanas temporales
- ✅ **Class weighting:** Automático en loss function

## 📊 Configuración Detallada

### Espacios de Búsqueda Optuna

**Para ML:**
```yaml
models:
  rf:
    optuna_search_space:
      n_estimators:
        type: "int"
        low: 50
        high: 300
      max_depth:
        type: "int"
        low: 10
        high: 50
```

**Para DL:**
```yaml
dl_models:
  transformer:
    optuna_search_space:
      lr:
        type: "loguniform"
        low: 0.00001
        high: 0.001
      d_model:
        type: "categorical"
        choices: [32, 64, 128]
```

### Early Stopping (DL)

```yaml
deep_learning:
  early_stopping:
    enabled: true
    patience: 10        # Épocas sin mejora
    min_delta: 0.001    # Mejora mínima requerida
    monitor: "val_f1"   # Métrica a monitorear
```

### Learning Rate Scheduler (DL)

```yaml
deep_learning:
  lr_scheduler:
    enabled: true
    type: "ReduceLROnPlateau"
    factor: 0.5      # LR se multiplica por 0.5
    patience: 5      # Épocas antes de reducir
    mode: "max"      # max para f1, min para loss
```

## 🔧 Modificar Arquitecturas DL

Edita `dl_models.py` para personalizar arquitecturas:

```python
class EEGTransformer(nn.Module):
    def __init__(self, input_dim=19, d_model=64, ...):
        super().__init__()
        # Tu arquitectura personalizada aquí
```

## 📈 Resultados

Los experimentos generan automáticamente:

- **Métricas:** Tabla comparativa en `images/results/`
- **Optuna:** CSV con trials y visualizaciones HTML
- **XAI (ML):** Gráficos SHAP y LIME en `images/xai/`
- **Features:** Features seleccionadas en `data/processed/`

## 💡 Ejemplos de Uso

### Comparar todos los modelos ML
```yaml
# config.yaml
experiment:
  type: "ml"

models:
  lr: {enabled: true}
  rf: {enabled: true}
  svc: {enabled: true}
  knn: {enabled: true}
  xgb: {enabled: true}

optuna:
  enabled: false  # Usar parámetros por defecto
```

### Optimizar solo Random Forest
```yaml
models:
  rf: {enabled: true}
  lr: {enabled: false}
  # ... resto desactivado

optuna:
  enabled: true
  n_trials: 100
```

### Entrenar Transformer con optimización
```yaml
# config_dl.yaml
experiment:
  type: "dl"

deep_learning:
  enabled: true
  epochs: 50

dl_models:
  transformer: {enabled: true}

optuna:
  enabled: true
  n_trials: 30  # DL es más lento
```

### Usar CNN sobre features TSFRESH
```yaml
deep_learning:
  enabled: true
  data_format: "features"  # No 'raw'

dl_models:
  cnn: {enabled: true}

feature_extraction:
  enabled: true  # Extraer features primero

feature_selection:
  enabled: true
  k: 50
```

## 🐛 Troubleshooting

### Error: "PyTorch/skorch no disponible"
```bash
pip install torch skorch
```

### Error: "CUDA out of memory"
```yaml
# Reducir batch_size
deep_learning:
  batch_size: 16  # o 8
```

O usar CPU:
```yaml
experiment:
  device: "cpu"
```

### Error: "No se encontraron ventanas con longitud 3000"
Verifica que tus datos tengan ventanas del tamaño correcto:
```python
df.groupby("window_id").size().value_counts()
```

### Optuna muy lento para DL
Reduce n_trials:
```yaml
optuna:
  n_trials: 10  # En lugar de 100
```

## 🎓 Mejores Prácticas

1. **Prueba primero sin Optuna:** Usa parámetros por defecto para verificar que todo funciona
2. **ML antes que DL:** Los modelos ML son más rápidos para iterar
3. **Validación cruzada en ML, no en DL:** CV es muy costoso computacionalmente para DL
4. **Early Stopping siempre activo:** Evita overfitting en DL
5. **Guarda checkpoints:** Para experimentos largos de DL

## 📚 Recursos

- **Optuna:** https://optuna.org/
- **skorch:** https://skorch.readthedocs.io/
- **TSFRESH:** https://tsfresh.readthedocs.io/
- **SHAP:** https://shap.readthedocs.io/

## 🤝 Contribuir

Para agregar un nuevo modelo:

1. **ML:** Agrega el modelo en `create_default_pipeline()` y en `create_optuna_objective()`
2. **DL:** Crea la arquitectura en `dl_models.py` y agrégala en `create_dl_model()`
3. **Config:** Agrega la configuración en `config.yaml` o `config_dl.yaml`

---

**Autor:** Sistema unificado de experimentos ML/DL  
**Fecha:** 2026  
**Versión:** 1.0
