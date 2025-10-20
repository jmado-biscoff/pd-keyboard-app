# ============================================================
# train_mlp.py
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
BATCH_SIZE = 32

# Dynamically resolve dataset path
data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "raw", "synthetic_finger_key_dataset.csv")
)
train_loader, test_loader, input_dim = load_data(data_path, batch_size=BATCH_SIZE)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(input_dim).to(device)
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

    # Early stop condition
    if round(avg_loss, 4) == 0.0000:
        print("Training stopped early because loss reached 0.0000")
        early_stop = True
        break

training_time = time.time() - start_time

# Evaluation
model.eval()
y_true, y_pred = [], []
start_infer = time.time()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        preds = model(X).cpu().numpy()
        y_pred.extend((preds > 0.5).astype(int).flatten())
        y_true.extend(y.numpy())

# =======================================
# Modified: Save inference time in ms
# =======================================
inference_time_ms = ((time.time() - start_infer) / len(y_true)) * 1000
precision = precision_score(y_true, y_pred, zero_division=0)

# Storage & Maintainability
save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved", "mlp_model.pth")
)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
storage_mb = os.path.getsize(save_path) / (1024 * 1024)
param_count = sum(p.numel() for p in model.parameters())
maintainability_index = max(0, 100 - (param_count / 1e5) * 5)

# =======================================
# Added: Export ONNX
# =======================================
onnx_path = save_path.replace(".pth", ".onnx")
dummy_input = torch.randn(1, input_dim).to(device)
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

# Save metrics
metrics = {
    "model": "MLP",
    "training_time_sec": round(training_time, 3),
    "inference_time_ms_per_sample": round(inference_time_ms, 3),  # Updated
    "precision": round(float(precision), 4),
    "storage_MB": round(storage_mb, 4),
    "maintainability_index": round(maintainability_index, 2),
    "early_stopped": early_stop
}

json_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved", "mlp_metrics.json")
)
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=4)

csv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved", "model_results.csv")
)
header = [
    "model", "training_time_sec", "inference_time_ms_per_sample",  # Updated
    "precision", "storage_MB", "maintainability_index", "early_stopped"
]
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if write_header:
        writer.writeheader()
    writer.writerow(metrics)

print("MLP model and metrics saved successfully.")
print(f"Model saved to: {save_path}")
print(f"Metrics saved to: {json_path}")
print(f"Results appended to: {csv_path}")
