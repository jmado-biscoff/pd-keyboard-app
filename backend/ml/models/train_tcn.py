"""
train_tcn.py — Train Temporal Convolutional Network (TCN)
for Finger-to-Key Correctness Classification
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_PATH = r"E:\pd-keyboard-app\backend\ml\dataset\raw\synthetic_finger_key_dataset_fixed.csv"
SAVE_DIR = r"E:\pd-keyboard-app\backend\ml\saved"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQUENCE_LEN = 32
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
PATIENCE = 7  # early stopping patience

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. DATASET PREPARATION
# ============================================================
class FingerSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(path, seq_len=32):
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = [
        "finger_x", "finger_y",
        "key_x1", "key_y1", "key_x2", "key_y2",
        "dx", "dy", "distance",
        "norm_dx", "norm_dy", "norm_distance"
    ]
    X = df[feature_cols].values
    y = df["is_correct"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    seq_data, seq_labels = [], []
    for i in range(len(X) - seq_len):
        seq_data.append(X[i:i+seq_len])
        seq_labels.append(y[i+seq_len-1])
    seq_data, seq_labels = np.array(seq_data), np.array(seq_labels)

    return seq_data, seq_labels


# ============================================================
# 3. MODEL DEFINITION
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,  # ensures same length
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        # Match sequence lengths in case of off-by-one due to padding
        if out.size(2) != residual.size(2):
            min_len = min(out.size(2), residual.size(2))
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len]

        return torch.relu(out + residual)

class FingerFitTCN(nn.Module):
    def __init__(self, input_dim=12, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_dim, 64, 3, 1),
            TCNBlock(64, 128, 3, 2),
            TCNBlock(128, 128, 3, 4),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, features, seq_len)
        x = self.tcn(x)
        x = x.squeeze(-1)
        return self.fc(x)


# ============================================================
# 4. TRAINING UTILITIES
# ============================================================
def train_one_fold(model, train_loader, val_loader, optimizer, criterion, fold_idx):
    best_acc = 0.0
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                preds_cls = torch.argmax(preds, dim=1)
                all_preds.extend(preds_cls.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"Learning rate reduced to {new_lr:.6f}")

        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"tcn_best_fold{fold_idx+1}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    return best_acc

# ============================================================
# 5. MAIN TRAINING LOOP (K-FOLD)
# ============================================================
def main():
    print("Loading dataset...")
    X, y = load_dataset(DATA_PATH, SEQUENCE_LEN)
    print(f"Dataset shape: {X.shape}, Labels: {y.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_acc = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n===== Fold {fold+1} / 5 =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = FingerSequenceDataset(X_train, y_train)
        val_ds = FingerSequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = FingerFitTCN(input_dim=X.shape[2], num_classes=2).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR)
        class_counts = np.bincount(y_train)
        weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
        weights = weights / weights.sum()  # normalize
        criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

        best_acc = train_one_fold(model, train_loader, val_loader, optimizer, criterion, fold)
        all_fold_acc.append(best_acc)
        print(f"Fold {fold+1} Best Accuracy: {best_acc*100:.2f}%")

    avg_acc = np.mean(all_fold_acc)
    print(f"\nAverage Cross-Validation Accuracy: {avg_acc*100:.2f}%")

    best_fold = np.argmax(all_fold_acc) + 1
    best_model_path = os.path.join(SAVE_DIR, f"tcn_best_fold{best_fold}.pth")
    final_model_path = os.path.join(SAVE_DIR, "tcn_best.pth")
    os.replace(best_model_path, final_model_path)
    print(f"Saved final best model → {final_model_path}")


if __name__ == "__main__":
    main()
