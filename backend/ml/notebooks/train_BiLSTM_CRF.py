#!/usr/bin/env python
# coding: utf-8
import os, json, time, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support

# -----------------------
# Root & paths
# -----------------------
def find_project_root() -> str:
    cwd = os.getcwd(); c=[cwd]; cur=cwd
    for _ in range(4):
        cur = os.path.dirname(cur)
        if cur and cur not in c: c.append(cur)
    for base in c:
        if os.path.isdir(os.path.join(base,"backend","ml","dataset","raw")): return base
    return cwd

PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "dataset", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "models")
EVAL_DIR  = os.path.join(PROJECT_ROOT, "backend", "ml", "evaluation")
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(EVAL_DIR, exist_ok=True)

def must(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")
    return p

# -----------------------
# Config (smaller = faster)
# -----------------------
BATCH_SIZE=128
EPOCHS=30
PATIENCE=4
LR=1e-3
HIDDEN=64
LAYERS=1
DROPOUT=0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load data
# -----------------------
X_train = np.load(must(os.path.join(DATA_DIR,"X_train.npy"))).astype(np.float32)
y_train = np.load(must(os.path.join(DATA_DIR,"y_train.npy"))).astype(np.int64)
X_val   = np.load(must(os.path.join(DATA_DIR,"X_val.npy"))).astype(np.float32)
y_val   = np.load(must(os.path.join(DATA_DIR,"y_val.npy"))).astype(np.int64)
X_test  = np.load(must(os.path.join(DATA_DIR,"X_test.npy"))).astype(np.float32)
y_test  = np.load(must(os.path.join(DATA_DIR,"y_test.npy"))).astype(np.int64)

SEQ_LEN = int(X_train.shape[1]); INPUT_SIZE = int(X_train.shape[2])
NUM_CLASSES = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

# -----------------------
# Model (BiLSTM + CRF)
# -----------------------
try:
    from torchcrf import CRF  # pip install pytorch-crf
except Exception as e:
    raise RuntimeError("Install dependency: pip install pytorch-crf") from e

class BiLSTMEmitter(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=(dropout if layers>1 else 0.0), bidirectional=True)
        self.proj = nn.Linear(hidden*2, classes)
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.proj(y)

class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, classes=2, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.emitter = BiLSTMEmitter(input_dim, hidden, layers, dropout, classes)
        self.crf = CRF(classes, batch_first=True)
    def nll(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

def full_mask(bsz, T, device): return torch.ones(bsz, T, dtype=torch.bool, device=device)

# -----------------------
# Data loaders
# -----------------------
def to_tensors(X,y): return torch.tensor(X), torch.tensor(y)
Xtr_t,ytr_t = to_tensors(X_train,y_train)
Xva_t,yva_t = to_tensors(X_val,y_val)
Xte_t,yte_t = to_tensors(X_test,y_test)
train_loader = DataLoader(TensorDataset(Xtr_t,ytr_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva_t,yva_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xte_t,yte_t), batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Train (with total time)
# -----------------------
model = BiLSTMCRF(INPUT_SIZE, classes=NUM_CLASSES, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val=float("inf"); patience=0
best_path = os.path.join(MODEL_DIR,"bilstm_crf_model.pt")

start_time = time.time()
for epoch in range(EPOCHS):
    # train
    model.train(); tr=0.0
    for xb,yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        mask = full_mask(xb.size(0), xb.size(1), DEVICE)
        optimizer.zero_grad()
        emissions = model.emitter(xb)
        loss = model.nll(emissions, yb, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr += loss.item() * xb.size(0)
    tr /= len(train_loader.dataset)

    # val
    model.eval(); va=0.0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mask = full_mask(xb.size(0), xb.size(1), DEVICE)
            emissions = model.emitter(xb)
            loss = model.nll(emissions, yb, mask)
            va += loss.item() * xb.size(0)
    va /= len(val_loader.dataset)

    print(f"Epoch {epoch+1:02d} | train {tr:.4f} | val {va:.4f}")
    if va < best_val:
        best_val = va; patience=0
        torch.save(model.state_dict(), best_path)
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early stopping.")
            break
end_time = time.time()
train_time = end_time - start_time
print(f"Total training time: {train_time:.2f} seconds")
with open(os.path.join(EVAL_DIR,"bilstm_train.json"), "w", encoding="utf-8") as f:
    json.dump({"training_time_seconds": float(train_time)}, f, indent=2)

# -----------------------
# (optional) quick eval & ONNX emitter export
# -----------------------
model.load_state_dict(torch.load(best_path, map_location=DEVICE)); model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for xb,yb in test_loader:
        xb = xb.to(DEVICE)
        mask = full_mask(xb.size(0), xb.size(1), DEVICE)
        emissions = model.emitter(xb)
        paths = model.crf.decode(emissions, mask=mask)   # list of [T]
        T = xb.size(1)
        preds = np.array([p[:T] for p in paths], dtype=np.int64)
        all_p.append(preds); all_t.append(yb.cpu().numpy())
y_pred = np.concatenate(all_p,axis=0).reshape(-1)
y_true = np.concatenate(all_t,axis=0).reshape(-1)
from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
with open(os.path.join(EVAL_DIR,"bilstm_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"precision":float(prec),"recall":float(rec),"f1_score":float(f1)}, f, indent=2)

# Export ONNX for emitter (fast runtime)
dummy = torch.randn(1, SEQ_LEN, INPUT_SIZE, device=DEVICE)
onnx_path = os.path.join(MODEL_DIR,"bilstm_emitter.onnx")
import onnx  # ensure installed
torch.onnx.export(
    model.emitter, dummy, onnx_path,
    input_names=["input"], output_names=["emissions"],
    dynamic_axes={"input":{0:"batch",1:"seq_len"}, "emissions":{0:"batch",1:"seq_len"}},
    opset_version=14
)
