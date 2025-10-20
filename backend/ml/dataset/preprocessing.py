# ============================================================
# preprocessing.py
# For dataset with header:
# timestamp, pressed_key, finger_name, hand_used, finger_x, finger_y,
# key_x1, key_y1, key_x2, key_y2, dx, dy, distance, norm_dx, norm_dy,
# norm_distance, is_correct
# ============================================================

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os


def load_data(csv_path, sequence_length=None, batch_size=32):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Define label column
    # ------------------------------------------------------------
    label_col = "is_correct"
    if label_col not in df.columns:
        raise KeyError(f"Expected '{label_col}' column in dataset, but found {list(df.columns)}")

    # ------------------------------------------------------------
    # Encode categorical features
    # ------------------------------------------------------------
    categorical_cols = ["pressed_key", "finger_name", "hand_used"]
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # ------------------------------------------------------------
    # Split features and label
    # ------------------------------------------------------------
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    # ------------------------------------------------------------
    # Scale numeric features
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_np = np.array(X_scaled, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)

    # ------------------------------------------------------------
    # Case 1: Frame-based (for MLP)
    # ------------------------------------------------------------
    if sequence_length is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
        )
        train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        print(f"Frame dataset loaded - Features: {X_train.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
        return (
            DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size, shuffle=False),
            X_train.shape[1],
        )

    # ------------------------------------------------------------
    # Case 2: Sequential (for TCN or BiLSTM)
    # ------------------------------------------------------------
    def make_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len - 1])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = make_sequences(X_np, y_np, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    print(f"Sequential dataset loaded - Features: {X_seq.shape[2]} | Sequence length: {sequence_length}")
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
        X_seq.shape[2],
    )
