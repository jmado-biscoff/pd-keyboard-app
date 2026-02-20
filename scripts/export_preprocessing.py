# One-time script to export encoder and scaler data from pickle to JSON
# Run from project root: python scripts/export_preprocessing.py

import os
import json
import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "frontend", "public", "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pickle files
encoder = joblib.load(os.path.join(RESULTS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(RESULTS_DIR, "scaler.pkl"))

# Export to JSON
data = {
    "encoder": {
        "categories": [cat.tolist() for cat in encoder.categories_]
    },
    "scaler": {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
}

output_path = os.path.join(OUTPUT_DIR, "preprocessing.json")
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Exported preprocessing data to {output_path}")
print(f"  Encoder categories: {[len(c) for c in data['encoder']['categories']]} features")
print(f"  Scaler: {len(data['scaler']['mean'])} numeric features")
