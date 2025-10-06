#!/usr/bin/env python
# coding: utf-8
"""
Compare models using the JSON artifacts produced by your training/evaluation
pipeline and save a CSV + PNG.

Outputs:
  - backend/ml/evaluation/model_comparison.csv
  - backend/ml/evaluation/model_comparison.png
"""

import os, json, math
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Helpers / paths
# -----------------------
def find_project_root() -> str:
    cwd = os.getcwd()
    candidates = [cwd]
    cur = cwd
    for _ in range(4):
        cur = os.path.dirname(cur)
        if cur and cur not in candidates:
            candidates.append(cur)
    for base in candidates:
        if os.path.isdir(os.path.join(base, "backend", "ml", "dataset", "raw")):
            return base
    return cwd

PROJECT_ROOT = find_project_root()
EVAL_DIR  = os.path.join(PROJECT_ROOT, "backend", "ml", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

def jload(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def coalesce_float(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return default

# -----------------------
# Model sources (filenames)
# -----------------------
SPECS = [
    {
        "name": "TCN",
        "metrics": "tcn_metrics.json",
        "train":   "tcn_train.json",
        # files to average for per-model MI (best-effort)
        "mi_files_contains": ["train_TCN.py", "evaluation_TCN.py"],
    },
    {
        "name": "BiLSTM-CRF",
        "metrics": "bilstm_metrics.json",
        "train":   "bilstm_train.json",
        "mi_files_contains": ["train_BiLSTM_CRF.py", "evaluation_BiLSTM_CRF.py"],
    },
    {
        "name": "Transformer",
        "metrics": "transformer_metrics.json",
        "train":   "transformer_train.json",
        "mi_files_contains": ["train_Transformer.py", "evaluation_Transformer.py"],
    },
]

# -----------------------
# Maintainability: map per model from maintainability_report.json
# - Uses per-file MI if available; falls back to overall average
# -----------------------
maint_report = jload(os.path.join(EVAL_DIR, "maintainability_report.json"))
overall_mi = maint_report.get("average_MI")
per_file_mi = {}
if "files" in maint_report:
    for row in maint_report["files"]:
        path = row.get("file")
        mi   = row.get("maintainability_index")
        if path and mi is not None:
            per_file_mi[path.replace("\\", "/")] = float(mi)

def model_mi(spec) -> float | None:
    # Average MI for files that contain any of the markers in the path
    markers = spec.get("mi_files_contains", [])
    matches = []
    for fpath, mi in per_file_mi.items():
        if any(m in fpath for m in markers):
            matches.append(mi)
    if matches:
        return round(sum(matches) / len(matches), 2)
    return overall_mi if overall_mi is not None else None

# -----------------------
# Build rows
# -----------------------
rows = []
for spec in SPECS:
    m = jload(os.path.join(EVAL_DIR, spec["metrics"]))
    t = jload(os.path.join(EVAL_DIR, spec["train"]))

    precision = coalesce_float(m, ["precision"])
    inf_ms    = coalesce_float(m, ["avg_inference_ms_per_seq"])
    ttime     = coalesce_float(m, ["training_time_seconds"], default=coalesce_float(t, ["training_time_seconds"]))
    onnx_mb   = coalesce_float(m, ["storage_onnx_mb"], default=0.0) or 0.0
    ckpt_mb   = coalesce_float(m, ["storage_checkpoint_mb"], default=0.0) or 0.0
    storage_total_mb = round(onnx_mb + ckpt_mb, 3)

    rows.append({
        "Model": spec["name"],
        "precision": precision,
        "training_time_seconds": ttime,
        "maintainability_index": model_mi(spec),
        "inference_time_ms": inf_ms,
        "storage_total_mb": storage_total_mb,
        # keep raw parts too (optional)
        "storage_onnx_mb": onnx_mb,
        "storage_checkpoint_mb": ckpt_mb,
    })

df = pd.DataFrame(rows)

# Sort by precision (desc) as a default view
if "precision" in df.columns and df["precision"].notna().any():
    df = df.sort_values(by="precision", ascending=False).reset_index(drop=True)

# Save table
csv_path = os.path.join(EVAL_DIR, "model_comparison.csv")
df.to_csv(csv_path, index=False)
print("Saved table:", csv_path)
print(df.to_string(index=False))

# -----------------------
# Plot PNG: 5 panels (precision, training time, MI, inference time, storage)
# -----------------------
try:
    # Prepare data (use 0 for missing so bars render)
    plot_df = df.fillna(0.0).copy()

    metrics = [
        ("precision", "Precision"),
        ("training_time_seconds", "Training Time (s)"),
        ("maintainability_index", "Maintainability Index"),
        ("inference_time_ms", "Inference Time (ms/seq)"),
        ("storage_total_mb", "Storage (MB, ONNX+CKPT)"),
    ]
    # Create a 2x3 grid and use 5 axes
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        ax.bar(plot_df["Model"], plot_df[col])
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=0)

        # value labels on top of bars
        for idx, v in enumerate(plot_df[col].tolist()):
            try:
                ax.text(idx, v, f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8, rotation=0)
            except Exception:
                pass

    # Hide the sixth (unused) subplot
    axes[-1].axis("off")

    plt.tight_layout()
    png_path = os.path.join(EVAL_DIR, "model_comparison.png")
    plt.savefig(png_path, dpi=150)
    print("Saved plot:", png_path)
except Exception as e:
    print("Plot skipped due to error:", e)
