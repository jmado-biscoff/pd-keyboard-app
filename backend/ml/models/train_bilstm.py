# ============================================================
# train_bilstm_crf.py — BiLSTM with 5-Fold Cross-Validation (Clean Version)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os, time, json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ============================================================
# CONFIGURATION
# ============================================================
EPOCHS = 60
LR = 1e-4
SEQ_LEN = 32
BATCH_SIZE = 32
HIDDEN_DIM = 128
DROPOUT = 0.3
N_SPLITS = 5  # Number of folds

data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "raw", "synthetic_finger_key_dataset.csv")
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# ============================================================
# VALIDATION
# ============================================================
required_cols = ["pressed_key", "finger_name", "hand_used", "is_correct"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ============================================================
# ENCODE FEATURES
# ============================================================
label_encoders = {}
for col in ["pressed_key", "finger_name", "hand_used"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df[["pressed_key", "finger_name", "hand_used"]].values
y = df["is_correct"].astype(int).values

# ============================================================
# SEQUENCE CREATION FUNCTION
# ============================================================
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(0, len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

# ============================================================
# MODEL DEFINITION
# ============================================================
class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# ============================================================
# CROSS-VALIDATION TRAINING
# ============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_metrics = []
best_f1 = 0
fold_idx = 1
input_dim = X.shape[1]

for train_index, test_index in skf.split(X, y):
    print(f"\n========== Fold {fold_idx}/{N_SPLITS} ==========")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train_seq, y_train_seq = make_sequences(X_train, y_train, SEQ_LEN)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, SEQ_LEN)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_seq, y_train_seq),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_seq, y_test_seq),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Initialize new model for each fold
    model = BiLSTMCRF(input_dim, HIDDEN_DIM, DROPOUT).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation per epoch
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                preds = model(Xb).cpu().numpy()
                y_pred.extend((preds > 0.5).astype(int).flatten())
                y_true.extend(yb.numpy())

        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        print(f"Fold {fold_idx} | Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Final evaluation per fold
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = model(Xb).cpu().numpy()
            y_pred.extend((preds > 0.5).astype(int).flatten())
            y_true.extend(yb.numpy())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Fold {fold_idx} — Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    fold_metrics.append({
        "fold": fold_idx,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved"))
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "bilstm_highacc_bestfold.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Best model updated (Fold {fold_idx})")

    fold_idx += 1

# ============================================================
# AGGREGATE METRICS
# ============================================================
avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
avg_f1 = np.mean([m["f1"] for m in fold_metrics])
avg_prec = np.mean([m["precision"] for m in fold_metrics])
avg_rec = np.mean([m["recall"] for m in fold_metrics])

print("\n========== Cross-Validation Summary ==========")
print(f"Avg Accuracy: {avg_acc:.4f} | Avg F1: {avg_f1:.4f} | Avg Precision: {avg_prec:.4f} | Avg Recall: {avg_rec:.4f}")

metrics = {
    "model": "BiLSTM-CRF (High Accuracy + 5-Fold CV)",
    "folds": N_SPLITS,
    "avg_accuracy": round(float(avg_acc), 4),
    "avg_precision": round(float(avg_prec), 4),
    "avg_recall": round(float(avg_rec), 4),
    "avg_f1_score": round(float(avg_f1), 4)
}

save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved"))
os.makedirs(save_dir, exist_ok=True)
json_path = os.path.join(save_dir, "bilstm_highacc_cv_metrics.json")
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nTraining complete. Cross-validation metrics saved to {json_path}")
