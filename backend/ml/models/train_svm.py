"""
train_svm_model.py
Trains and evaluates an SVM model using the preprocessed finger-to-key dataset.
Exports model in .pkl, .onnx, and .pth formats, and logs metrics for comparison.
"""

import os
import time
import joblib
import inspect
import torch
import pandas as pd
import json
import csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import GridSearchCV
from radon.metrics import mi_visit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ====================================================
# LOAD DATA
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(RESULTS_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(RESULTS_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(RESULTS_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(RESULTS_DIR, "y_test.csv")).values.ravel()

print(f"Loaded dataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

# ====================================================
# TRAIN SVM (RBF)
# ====================================================
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}

print("Training SVM with grid search (this may take a few minutes)...")

start_train = time.time()
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
end_train = time.time()

best_svm = grid_search.best_estimator_
train_time = end_train - start_train

print(f"Best Parameters: {grid_search.best_params_}")

# ====================================================
# EVALUATE MODEL
# ====================================================
y_pred = best_svm.predict(X_test)
y_proba = best_svm.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

# Inference time (ms per sample)
start_inf = time.time()
_ = best_svm.predict(X_test)
end_inf = time.time()
inference_time = (end_inf - start_inf) / len(X_test) * 1000

# Maintainability index (of this script)
code_str = inspect.getsource(__import__(__name__))
mi_score = mi_visit(code_str, True)

# Save .pkl (scikit-learn format)
pkl_path = os.path.join(RESULTS_DIR, "svm_model.pkl")
joblib.dump(best_svm, pkl_path)
file_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)

# Mean Average Precision
mAP = average_precision_score(y_test, y_proba)

print("\nMODEL PERFORMANCE METRICS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Inference Time: {inference_time:.6f} ms/sample")
print(f"Training Time: {train_time:.4f} s")
print(f"Maintainability Index: {mi_score:.2f}")
print(f"Storage (pkl): {file_size_mb:.4f} MB")
print(f"Mean Average Precision (mAP): {mAP:.4f}")

# ====================================================
# EXPORT TO ONNX
# ====================================================
print("\nExporting to ONNX format...")
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(best_svm, initial_types=initial_type)
onnx_path = os.path.join(RESULTS_DIR, "svm_model.onnx")
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Saved ONNX model: {onnx_path}")

# ====================================================
# EXPORT TO PyTorch (.pth)
# ====================================================
print("\nExporting to PyTorch format...")

class SVMPyTorchWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        preds = self.model.predict_proba(x_np)[:, 1]
        return torch.tensor(preds, dtype=torch.float32)

svm_torch = SVMPyTorchWrapper(best_svm)
pth_path = os.path.join(RESULTS_DIR, "svm_model.pth")
torch.save(svm_torch, pth_path)
print(f"Saved PyTorch model: {pth_path}")

# ====================================================
# SAVE METRICS TO CSV FOR MODEL COMPARISON
# ====================================================
metrics_csv = os.path.join(RESULTS_DIR, "model_comparison_metrics.csv")

metrics_data = {
    "Model": "SVM",
    "Accuracy": round(accuracy, 4),
    "Error Rate": round(error_rate, 4),
    "Inference Time (ms/sample)": round(inference_time, 6),
    "Training Time (s)": round(train_time, 4),
    "Mean Average Precision": round(mAP, 4),
    "Maintainability Index": round(mi_score, 2),
    "Storage (MB)": round(file_size_mb, 4)
}

file_exists = os.path.isfile(metrics_csv)
with open(metrics_csv, mode='a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metrics_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics_data)

print(f"\nMetrics appended to: {metrics_csv}")

# ====================================================
# SAVE METRICS AS JSON (OPTIONAL)
# ====================================================
metrics_json = os.path.join(RESULTS_DIR, "svm_metrics.json")
with open(metrics_json, "w") as f:
    json.dump(metrics_data, f, indent=4)
print(f"Saved JSON metrics: {metrics_json}")

# ====================================================
# SUMMARY
# ====================================================
print("\nTRAINING AND EXPORT COMPLETE")
print(f"Results saved in: {RESULTS_DIR}")
