# backend/ml/evaluation/evaluation_TCN.py
import os, json, time, numpy as np
from sklearn.metrics import precision_recall_fscore_support

# -----------------------
# Paths & hyperparams
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
DATA_DIR  = os.path.join(PROJECT_ROOT, "backend", "ml", "dataset", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "models")
EVAL_DIR  = os.path.join(PROJECT_ROOT, "backend", "ml", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

ONNX_PATH = os.path.join(MODEL_DIR, "tcn_model.onnx")
PT_PATH   = os.path.join(MODEL_DIR, "tcn_model.pt")
OUT_JSON  = os.path.join(EVAL_DIR, "tcn_eval_from_onnx.json")
TRAIN_JSON= os.path.join(EVAL_DIR, "tcn_train.json")

# -----------------------
# Load test data
# -----------------------
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.int64)

# -----------------------
# ONNX inference
# -----------------------
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name

def run_inference(session, X, batch=64):
    preds, n_batches = [], 0
    t0 = time.perf_counter()
    for i in range(0, len(X), batch):
        xb = X[i:i+batch]
        logits = session.run(None, {in_name: xb})[0]  # [B,T,C]
        preds.append(logits.argmax(axis=-1))          # [B,T]
        n_batches += 1
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / max(1, n_batches) * 1000.0
    return np.vstack(preds), round(avg_ms, 3)

preds, avg_ms = run_inference(sess, X_test, batch=64)
y_true = y_test.reshape(-1)
y_pred = preds.reshape(-1)

prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

def size_mb(p): return round(os.path.getsize(p)/1024/1024, 3) if os.path.exists(p) else None
train_time = None
if os.path.exists(TRAIN_JSON):
    try: train_time = json.load(open(TRAIN_JSON))["training_time_seconds"]
    except: pass

result = {
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "avg_inference_ms_per_seq": float(avg_ms),
    "storage_onnx_mb": size_mb(ONNX_PATH),
    "storage_checkpoint_mb": size_mb(PT_PATH),
    "training_time_seconds": float(train_time) if train_time is not None else None,
    "num_sequences": int(X_test.shape[0]),
    "seq_len": int(X_test.shape[1]),
}
with open(OUT_JSON, "w") as f: json.dump(result, f, indent=2)
print(f"Saved: {OUT_JSON}")