#!/usr/bin/env python
# coding: utf-8
import os, json, time, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support

# ---------- paths ----------
def find_project_root() -> str:
    cwd = os.getcwd(); c=[cwd]; cur=cwd
    for _ in range(4):
        cur = os.path.dirname(cur)
        if cur and cur not in c: c.append(cur)
    for base in c:
        if os.path.isdir(os.path.join(base,"backend","ml","dataset","raw")):
            return base
    return cwd

ROOT     = find_project_root()
DATA_DIR = os.path.join(ROOT, "backend", "ml", "dataset", "processed")
MODEL_DIR= os.path.join(ROOT, "backend", "ml", "models")
EVAL_DIR = os.path.join(ROOT, "backend", "ml", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

def must(p):
    if not os.path.exists(p): raise FileNotFoundError(p)
    return p

# ---------- data ----------
X_test = np.load(must(os.path.join(DATA_DIR,"X_test.npy"))).astype(np.float32)
y_test = np.load(must(os.path.join(DATA_DIR,"y_test.npy"))).astype(np.int64)
B, T, F = X_test.shape
NUM_CLASSES = int(y_test.max() + 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- model (same as training) ----------
try:
    from torchcrf import CRF
except Exception as e:
    raise RuntimeError("Install dependency: pip install pytorch-crf") from e

class BiLSTMEmitter(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=(dropout if layers>1 else 0.0), bidirectional=True)
        self.proj = nn.Linear(hidden*2, classes)
    def forward(self, x):
        y,_ = self.lstm(x); return self.proj(y)

class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, classes=2, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.emitter = BiLSTMEmitter(input_dim, hidden, layers, dropout, classes)
        self.crf = CRF(classes, batch_first=True)

PT_PATH = os.path.join(MODEL_DIR,"bilstm_crf_model.pt")
model = BiLSTMCRF(F, classes=NUM_CLASSES, hidden=64, layers=1, dropout=0.1).to(DEVICE)
model.load_state_dict(torch.load(must(PT_PATH), map_location=DEVICE))
model.eval()

# ---------- accuracy via CRF decode ----------
loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=128)
all_p, all_t = [], []
def full_mask(bsz, t, device): return torch.ones(bsz, t, dtype=torch.bool, device=device)
with torch.no_grad():
    for xb,yb in loader:
        xb = xb.to(DEVICE)
        mask = full_mask(xb.size(0), xb.size(1), DEVICE)
        emissions = model.emitter(xb)
        paths = model.crf.decode(emissions, mask=mask)
        preds = np.array([p[:xb.size(1)] for p in paths], dtype=np.int64)
        all_p.append(preds); all_t.append(yb.numpy())
y_pred = np.concatenate(all_p, axis=0).reshape(-1)
y_true = np.concatenate(all_t, axis=0).reshape(-1)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

# ---------- latency (use ONNX emitter) ----------
avg_ms = None
ONNX_PATH = os.path.join(MODEL_DIR,"bilstm_emitter.onnx")
if os.path.exists(ONNX_PATH):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    _ = sess.run(None, {in_name: X_test[:8]})
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, {in_name: X_test})
    elapsed = (time.perf_counter()-t0)/iters
    avg_ms = (elapsed * 1000.0) / B
else:
    # fallback: time PyTorch emitter forward
    with torch.no_grad():
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model.emitter(torch.tensor(X_test).to(DEVICE))
        elapsed = (time.perf_counter()-t0)/iters
        avg_ms = (elapsed * 1000.0) / B

def size_mb(p): return round(os.path.getsize(p)/1e6,3) if os.path.exists(p) else None

# ---------- merge training time ----------
train_time = None
train_json = os.path.join(EVAL_DIR, "bilstm_train.json")
if os.path.exists(train_json):
    with open(train_json, "r", encoding="utf-8") as f:
        tr = json.load(f)
    train_time = tr.get("training_time_seconds")

# ---------- save ----------
OUT_JSON = os.path.join(EVAL_DIR, "bilstm_metrics.json")
result = {
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "avg_inference_ms_per_seq": float(avg_ms) if avg_ms is not None else None,
    "storage_onnx_mb": size_mb(ONNX_PATH),
    "storage_checkpoint_mb": size_mb(PT_PATH),
    "training_time_seconds": float(train_time) if train_time is not None else None,
    "num_sequences": int(B),
    "seq_len": int(T)
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {OUT_JSON}")