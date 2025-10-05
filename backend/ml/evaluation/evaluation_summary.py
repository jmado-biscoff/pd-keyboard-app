# backend/ml/evaluation/evaluation_summary.py
import os, json, pandas as pd

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
EVAL_DIR  = os.path.join(PROJECT_ROOT, "backend", "ml", "evaluation")

def j(name):
    p = os.path.join(EVAL_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else {}

rows = []
rows.append({"Model":"TCN"}         | j("tcn_eval_from_onnx.json")        | j("tcn_train.json"))
rows.append({"Model":"BiLSTM+CRF"}  | j("bilstm_eval_from_onnx.json")      | j("bilstm_train.json"))
rows.append({"Model":"Transformer"} | j("transformer_eval_from_onnx.json") | j("transformer_train.json"))

df = pd.DataFrame(rows)
print("\nModel comparison:\n", df)

mi = j("maintainability_report.json").get("average_MI")
print("\nAverage Maintainability Index:", mi)

out_csv = os.path.join(EVAL_DIR, "model_comparison.csv")
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
