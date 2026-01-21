# TFM_EGG_SEIZURE
![alt tag](https://github.com/AitorRD/TFM_EGG_SEIZURE/blob/main/images/canales.jpg)

## Data
The database consists of 14 folders containing EEG recordings in EDF format (European Data Format). Each folder refers to a specific subject including between 1 and 5 data files with a maximum size of 2.11 GB each, and a text file containing information on data and seizures. The edf files contain signals recorded on the same or different days and the seizure events are chronologically ordered. All dates in the .edf files are de-identified.

### Data Avaibility
[here](https://physionet.org/content/siena-scalp-eeg/1.0.0/)

El pipeline del proyecto de clasificación de crisis epilépticas (EEG Seizure Classification) es el siguiente:

## Pipeline del Proyecto
1. Adquisición de Datos
Fuente: Base de datos Siena Scalp EEG
Formato: Archivos EDF (European Data Format)
Contenido: 14 pacientes con 1-5 archivos EEG cada uno
2. Procesamiento de Datos
2.1 Conversión EDF → CSV (convertion-edf-csv.py)
Lee archivos EDF con MNE
Extrae 19 canales EEG (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2)
Aplica clipping: recorta ±30 minutos alrededor de cada crisis
Genera archivos CSV por sesión
2.2 Concatenación (concat.py)
Agrupa sesiones de pacientes
División estratificada: 70% train / 15% val / 15% test
Salida: train.csv, val.csv, test.csv
2.3 Ventaneo (window.py)
Tamaño de ventana: 10 segundos (1000 muestras @ 100 Hz)
Overlap: 25%
Mantiene integridad por sesión
Salida: dataset_windowed_*.csv
3. Experimentación
3.1 Modelos Clásicos (machine_learning.py)
Extracción de características: tsfresh (features temporales)
Selección de features: SelectKBest (top 50)
Modelos: Logistic Regression, Random Forest, SVM, KNN, XGBoost
Validación: Métricas (accuracy, precision, recall, F1)
3.2 Modelos Deep Learning
CNN: CNN.py
RNN: RNN.py
Transformer: transformers.py
Arquitectura: TransformerEncoder (d_model=64, 4 heads, 2 layers)
Input: 19 canales × 3000 timesteps
Training con class weights balanceados
4. Evaluación y Visualización
Métricas: Accuracy, Precision, Recall, F1-Score
Gráficos: graphs.py, barplot.py
Estadísticas: stats.py
XAI: Interpretabilidad con SHAP/LIME