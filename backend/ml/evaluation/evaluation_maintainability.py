# backend/ml/evaluation/evaluation_maintainability.py
import os, json, glob
from radon.metrics import mi_visit

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
os.makedirs(EVAL_DIR, exist_ok=True)

PATTERNS = [
    os.path.join(PROJECT_ROOT, "backend/**/*.py"),
    os.path.join(PROJECT_ROOT, "frontend/**/*.py"),
]

files = []
for pat in PATTERNS:
    files.extend(glob.glob(pat, recursive=True))
files = [f for f in files if ("site-packages" not in f and "__pycache__" not in f)]

def file_mi(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        return float(mi_visit(src, multi=True))
    except Exception:
        return None

rows = [{"file": f, "maintainability_index": round(mi, 2)}
        for f in files if (mi := file_mi(f)) is not None]

avg_mi = round(sum(r["maintainability_index"] for r in rows) / max(1, len(rows)), 2) if rows else None

out = {"average_MI": avg_mi, "files": rows}
out_path = os.path.join(EVAL_DIR, "maintainability_report.json")
with open(out_path, "w") as fp: json.dump(out, fp, indent=2)
print("Average MI:", avg_mi)
print(f"Saved: {out_path}")
