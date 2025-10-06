#!/usr/bin/env python
# coding: utf-8
import os, json, time, numpy as np, torch
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
# Load data
# -----------------------
X_train = np.load(must(os.path.join(DATA_DIR,"X_train.npy"))).astype(np.float32)
y_train = np.load(must(os.path.join(DATA_DIR,"y_train.npy"))).astype(np.int64)
X_val   = np.load(must(os.path.join(DATA_DIR,"X_val.npy"))).astype(np.float32)
y_val   = np.load(must(os.path.join(DATA_DIR,"y_val.npy"))).astype(np.int64)
X_test  = np.load(must(os.path.join(DATA_DIR,"X_test.npy"))).astype(np.float32)
y_test  = np.load(must(os.path.join(DATA_DIR,"y_test.npy"))).astype(np.int64)

BATCH_SIZE=64; EPOCHS=50; PATIENCE=5; LR=1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = int(X_train.shape[1]); INPUT_SIZE = int(X_train.shape[2])
NUM_CLASSES = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

# -----------------------
# Model (TCN)
# -----------------------
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
            layers.append(TemporalBlock(in_c,c,kernel_size,1,d,pad)); in_c=c
        self.t = nn.Sequential(*layers)
        self.h = nn.Conv1d(in_c, output_size, 1)
    def forward(self,x):
        x=x.transpose(1,2)      # [B,F,T]
        h=self.t(x)             # [B,C,T]
        y=self.h(h).transpose(1,2)  # [B,T,C]
        return y

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
model = TCN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val=float("inf"); patience=0
best_path = os.path.join(MODEL_DIR,"tcn_model.pt")

start_time = time.time()
for epoch in range(EPOCHS):
    model.train(); tr=0.0
    for xb,yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits.reshape(-1, NUM_CLASSES), yb.reshape(-1))
        loss.backward(); optimizer.step()
        tr += loss.item() * xb.size(0)
    tr /= len(train_loader.dataset)

    model.eval(); va=0.0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits=model(xb)
            loss=criterion(logits.reshape(-1,NUM_CLASSES), yb.reshape(-1))
            va += loss.item() * xb.size(0)
    va /= len(val_loader.dataset)
    print(f"Epoch {epoch+1:02d} | train {tr:.4f} | val {va:.4f}")

    if va < best_val:
        best_val = va; patience = 0
        torch.save(model.state_dict(), best_path)
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early stopping.")
            break
end_time = time.time()
train_time = end_time - start_time
print(f"Total training time: {train_time:.2f} seconds")
with open(os.path.join(EVAL_DIR,"tcn_train.json"), "w", encoding="utf-8") as f:
    json.dump({"training_time_seconds": float(train_time)}, f, indent=2)

# -----------------------
# (optional) quick eval & ONNX export
# -----------------------
model.load_state_dict(torch.load(best_path, map_location=DEVICE)); model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for xb,yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        all_p.append(pred); all_t.append(yb.numpy())
all_p = np.concatenate(all_p,axis=0).reshape(-1)
all_t = np.concatenate(all_t,axis=0).reshape(-1)
prec, rec, f1, _ = precision_recall_fscore_support(all_t, all_p, average="binary")
with open(os.path.join(EVAL_DIR,"tcn_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"precision":float(prec),"recall":float(rec),"f1_score":float(f1)}, f, indent=2)

dummy = torch.randn(1, SEQ_LEN, INPUT_SIZE, device=DEVICE)
onnx_path = os.path.join(MODEL_DIR,"tcn_model.onnx")
import onnx  # ensure installed
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":{0:"batch",1:"seq_len"},"output":{0:"batch",1:"seq_len"}},
    opset_version=14
)
