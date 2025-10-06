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

# ---------- model (same as training defaults) ----------
class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp=chomp
    def forward(self,x): return x[:, :, :-self.chomp] if self.chomp>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, dilation, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,out_ch,k,stride=stride,padding=padding,dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.BatchNorm1d(out_ch),
            nn.Conv1d(out_ch,out_ch,k,stride=stride,padding=padding,dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.BatchNorm1d(out_ch),
        )
        self.down = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else None
        self.relu = nn.ReLU()
    def forward(self,x):
        out=self.net(x); res=x if self.down is None else self.down(x)
        return self.relu(out+res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, channels=(32,32,32), kernel_size=3, dilations=(1,2,4)):
        super().__init__()
        layers=[]; in_c=input_size
        for c,d in zip(channels,dilations):
            pad=(kernel_size-1)*d
            layers.append(TemporalBlock(in_c,c,kernel_size,1,d,pad))
            in_c=c
        self.t = nn.Sequential(*layers)
        self.h = nn.Conv1d(in_c, output_size, 1)
    def forward(self,x):
        x=x.transpose(1,2)      # [B,F,T]
        h=self.t(x)             # [B,C,T]
        y=self.h(h).transpose(1,2)  # [B,T,C]
        return y

# ---------- accuracy via PyTorch checkpoint ----------
PT_PATH = os.path.join(MODEL_DIR,"tcn_model.pt")
model = TCN(F, NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(must(PT_PATH), map_location=DEVICE))
model.eval()

loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64)
all_p, all_t = [], []
with torch.no_grad():
    for xb,yb in loader:
        xb = xb.to(DEVICE)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        all_p.append(pred); all_t.append(yb.numpy())
y_pred = np.concatenate(all_p, axis=0).reshape(-1)
y_true = np.concatenate(all_t, axis=0).reshape(-1)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

# ---------- latency (prefer ONNX) ----------
def size_mb(p): return round(os.path.getsize(p)/1e6,3) if os.path.exists(p) else None

avg_ms = None
ONNX_PATH = os.path.join(MODEL_DIR,"tcn_model.onnx")
if os.path.exists(ONNX_PATH):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    _ = sess.run(None, {in_name: X_test[:8]})  # warmup
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, {in_name: X_test})
    elapsed = (time.perf_counter()-t0)/iters
    avg_ms = (elapsed * 1000.0) / B
else:
    # fallback: time PyTorch forward
    with torch.no_grad():
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(torch.tensor(X_test).to(DEVICE))
        elapsed = (time.perf_counter()-t0)/iters
        avg_ms = (elapsed * 1000.0) / B

# ---------- merge training time ----------
train_time = None
train_json = os.path.join(EVAL_DIR, "tcn_train.json")
if os.path.exists(train_json):
    with open(train_json, "r", encoding="utf-8") as f:
        tr = json.load(f)
    train_time = tr.get("training_time_seconds")

# ---------- save ----------
OUT_JSON = os.path.join(EVAL_DIR, "tcn_metrics.json")
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
