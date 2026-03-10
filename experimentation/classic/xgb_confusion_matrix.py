"""
Confusion Matrix for XGBoost model (standalone)
Loads pipeline and selector, predicts on test set, and plots confusion matrix.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONFIGURATION
SUFFIX = ""
PIPELINE_PATH = f"data/processed/models/pipeline_xgb{SUFFIX}.joblib"
SELECTOR_PATH = f"data/processed/models/selector_xgb{SUFFIX}.joblib"
FEATURES_TEST_PATH = f"data/processed/features_test{SUFFIX}.csv"
LABELS_TEST_PATH = f"data/processed/labels_test{SUFFIX}.csv"
OUTPUT_PATH = f"images/xai/xgb_confusion_matrix{SUFFIX}.png"

# Load pipeline and selector
pipeline = joblib.load(PIPELINE_PATH)
selector = joblib.load(SELECTOR_PATH)

# Load features and labels
X_test_full = pd.read_csv(FEATURES_TEST_PATH, index_col=0)
y_test = pd.read_csv(LABELS_TEST_PATH, index_col=0).values.ravel()

# Reorder columns and select features
expected_cols = list(selector.feature_names_in_)
X_test_full = X_test_full[expected_cols]
X_test = selector.transform(X_test_full)

# Predict
y_pred = pipeline.predict(X_test)

# Confusion matrix (standard: 0 first)
cm = confusion_matrix(y_test, y_pred)

# Reorder to TP FP / FN TN layout (positive class = 1 first)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    cm_display = np.array([[tp, fp], [fn, tn]])
    display_labels = [1, 0]
else:
    cm_display = cm
    labels = pipeline.classes_ if hasattr(pipeline, 'classes_') else np.unique(y_test)
    display_labels = labels

fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=display_labels)
disp.plot(ax=ax, cmap='Blues', colorbar=False)

# Add TP/TN/FP/FN labels to each cell
if cm.shape == (2, 2):
    cell_labels = [["TP", "FP"], ["FN", "TN"]]
    for i in range(2):
        for j in range(2):
            disp.text_[i, j].set_text(f"{cm_display[i, j]}\n({cell_labels[i][j]})")

# Increase font sizes
plt.title('Confusion Matrix — XGBoost', fontsize=20, fontweight='bold')
ax.set_xlabel(ax.get_xlabel(), fontsize=18)
ax.set_ylabel(ax.get_ylabel(), fontsize=18)
ax.tick_params(axis='both', labelsize=16)
for text_obj in disp.text_.ravel():
    text_obj.set_fontsize(20)
plt.tight_layout()
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Confusion matrix saved: {OUTPUT_PATH}")

# Mostrar TP, FP, TN, FN (binario)
if cm.shape == (2, 2):
	tn, fp, fn, tp = cm.ravel()
	print(f"\nConfusion Matrix (raw counts):")
	print(f"TN: {tn}  FP: {fp}")
	print(f"FN: {fn}  TP: {tp}")
else:
	print("\nAdvertencia: La matriz de confusión no es binaria, no se muestran TP/FP/FN/TN.")
