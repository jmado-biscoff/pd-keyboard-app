# ============================================================
# train_bilstm_crf.py
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import time, json, os, csv, sys
from sklearn.metrics import precision_score

# Allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.preprocessing import load_data

EPOCHS = 40
LR = 5e-4
SEQ_LEN = 32
BATCH_SIZE = 32

# Dynamically resolve dataset path
data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "raw", "synthetic_finger_key_dataset.csv")
)
train_loader, test_loader, input_dim = load_data(
    data_path, sequence_length=SEQ_LEN, batch_size=BATCH_SIZE
)

class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMCRF(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

start_time = time.time()
early_stop = False

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    if round(avg_loss, 4) == 0.0000:
        print("Training stopped early because loss reached 0.0000")
        early_stop = True
        break

training_time = time.time() - start_time

# ============================================================
# EVALUATION
# ============================================================
model.eval()
y_true, y_pred = [], []
start_infer = time.time()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        preds = model(X).cpu().numpy()
        y_pred.extend((preds > 0.5).astype(int).flatten())
        y_true.extend(y.numpy())

# Convert inference time to milliseconds per sample
inference_time_ms = ((time.time() - start_infer) / len(y_true)) * 1000
precision = precision_score(y_true, y_pred, zero_division=0)

# ============================================================
# STORAGE & MAINTAINABILITY
# ============================================================
save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved", "bilstm_crf_model.pth")
)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
storage_mb = os.path.getsize(save_path) / (1024 * 1024)
param_count = sum(p.numel() for p in model.parameters())
maintainability_index = max(0, 100 - (param_count / 1e5) * 5)

# ============================================================
# Export ONNX version
# ============================================================
onnx_path = save_path.replace(".pth", ".onnx")
dummy_input = torch.randn(1, SEQ_LEN, input_dim).to(device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print(f"ONNX model exported to: {onnx_path}")

# ============================================================
# METRICS
# ============================================================
metrics = {
    "model": "BiLSTM-CRF",
    "training_time_sec": round(training_time, 3),
    "inference_time_ms_per_sample": round(inference_time_ms, 3),
    "precision": round(float(precision), 4),
    "storage_MB": round(storage_mb, 4),
    "maintainability_index": round(maintainability_index, 2),
    "early_stopped": early_stop
}

# ============================================================
# SAVE METRICS
# ============================================================
json_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved", "bilstm_crf_metrics.json")
)
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=4)

csv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved", "model_results.csv")
)
header = [
    "model", "training_time_sec", "inference_time_ms_per_sample",
    "precision", "storage_MB", "maintainability_index", "early_stopped"
]
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if write_header:
        writer.writeheader()
    writer.writerow(metrics)

print("BiLSTM-CRF model and metrics saved successfully.")
print(f"Model saved to: {save_path}")
print(f"Metrics saved to: {json_path}")
print(f"Results appended to: {csv_path}")
