# backend/ml/evaluation/evaluation_summary.py
import os
import json
import pandas as pd

# -----------------------
# Paths (with root finder)
# -----------------------
def find_project_root() -> str:
    """
    Try current dir and up to 4 parents to locate a folder that contains 'backend/ml/dataset/raw'.
    If not found, return current working directory.
    """
    cwd = os.getcwd()
    candidates = [cwd]
    cur = cwd
    for _ in range(4):
        cur = os.path.dirname(cur)
        if cur and cur not in candidates:
            candidates.append(cur)
    for base in candidates:
        raw_dir = os.path.join(base, "backend", "ml", "dataset", "raw")
        if os.path.isdir(raw_dir):
            return base
    return cwd

PROJECT_ROOT = find_project_root()
EVAL_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "evaluation")

# -----------------------
# Helpers
# -----------------------
def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def coalesce_float(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return default

# -----------------------
# Collect metrics per model
# -----------------------
models = [
    ("TCN",          "tcn_metrics.json",          "tcn_train.json"),
    ("BiLSTM+CRF",   "bilstm_metrics.json",       "bilstm_train.json"),
    ("Transformer",  "transformer_metrics.json",  "transformer_train.json"),
]

rows = []
for name, metrics_file, train_file in models:
    m = load_json(os.path.join(EVAL_DIR, metrics_file))
    t = load_json(os.path.join(EVAL_DIR, train_file))

    # Accept your latest field names from evaluation scripts:
    # precision, recall, f1_score, avg_inference_ms_per_seq,
    # storage_onnx_mb, storage_checkpoint_mb, training_time_seconds,
    # num_sequences, seq_len
    row = {
        "Model": name,
        "precision":       coalesce_float(m, ["precision"]),
        "recall":          coalesce_float(m, ["recall"]),
        "f1_score":        coalesce_float(m, ["f1_score"]),
        "avg_inference_ms_per_seq": coalesce_float(m, ["avg_inference_ms_per_seq"]),
        "storage_onnx_mb":          coalesce_float(m, ["storage_onnx_mb"]),
        "storage_checkpoint_mb":    coalesce_float(m, ["storage_checkpoint_mb"]),
        # prefer metric file if it already merged training time, else fallback to train json
        "training_time_seconds":    coalesce_float(m, ["training_time_seconds"], default=coalesce_float(t, ["training_time_seconds"])),
        "num_sequences":            coalesce_float(m, ["num_sequences"]),
        "seq_len":                  coalesce_float(m, ["seq_len"]),
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Clean up column order for readability
cols_order = [
    "Model",
    "precision", "recall", "f1_score",
    "avg_inference_ms_per_seq",
    "storage_onnx_mb", "storage_checkpoint_mb",
    "training_time_seconds",
    "num_sequences", "seq_len",
]
df = df[[c for c in cols_order if c in df.columns]]

print("\nModel comparison:\n")
print(df.to_string(index=False))

# Maintainability (optional)
mi_report = load_json(os.path.join(EVAL_DIR, "maintainability_report.json"))
avg_mi = mi_report.get("average_MI")
if avg_mi is not None:
    print(f"\nAverage Maintainability Index: {avg_mi}")
else:
    print("\nAverage Maintainability Index: (report not found)")

# Save CSV
out_csv = os.path.join(EVAL_DIR, "model_comparison.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# Optional: simple comparison plot
try:
    import matplotlib.pyplot as plt

    # Choose a subset of metrics to visualize
    plot_cols = ["precision", "f1_score", "avg_inference_ms_per_seq", "training_time_seconds", "storage_onnx_mb"]
    plot_cols = [c for c in plot_cols if c in df.columns]

    if len(plot_cols) >= 2:
        # Normalize non-accuracy metrics for a fair visual (except precision/f1)
        df_plot = df.set_index("Model")[plot_cols].copy()
        # Keep precision/f1 as is; normalize timing/storage columns only
        to_norm = [c for c in df_plot.columns if c not in ("precision", "f1_score")]
        for c in to_norm:
            col = df_plot[c]
            if col.notna().any() and col.max() != col.min():
                df_plot[c] = (col - col.min()) / (col.max() - col.min())

        ax = df_plot.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Model Comparison (precision/f1 raw; other metrics normalized)")
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        out_png = os.path.join(EVAL_DIR, "model_comparison.png")
        plt.savefig(out_png)
        print(f"Saved: {out_png}")
    else:
        print("Skipping plot (insufficient metrics to visualize).")
except Exception as e:
    print(f"Skipping plot (matplotlib not available or error occurred): {e}")