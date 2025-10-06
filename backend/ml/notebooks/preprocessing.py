#!/usr/bin/env python
# coding: utf-8
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Project root & paths
# ---------------------------
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

PROJECT_ROOT  = find_project_root()
RAW_DIR       = os.path.join(PROJECT_ROOT, "backend", "ml", "dataset", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "dataset", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def coalesce_col(df: pd.DataFrame, *names):
    """Return the first matching column name or None."""
    for n in names:
        if n in df.columns:
            return n
    return None

def require_cols(df: pd.DataFrame, needed):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ---------------------------
# 1) Load & concat raw CSVs
# ---------------------------
raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
if not raw_files:
    raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}")

dfs = []
for fp in raw_files:
    # robust read; skip malformed lines if any
    df = pd.read_csv(fp, on_bad_lines="skip")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(dfs)} files -> {data.shape[0]} rows")

# ---------------------------
# 2) Normalize column names (robust)
# ---------------------------
# Try to find press/release time columns (case-insensitive variants)
press_col   = coalesce_col(data, "Press_Time", "press_time", "press", "down_time", "key_down")
release_col = coalesce_col(data, "Release_Time", "release_time", "release", "up_time", "key_up")
key_col     = coalesce_col(data, "Key", "key", "scan_code", "key_code")  # optional
user_col    = coalesce_col(data, "User_ID", "user_id", "user", "participant", "subject")  # optional
sess_col    = coalesce_col(data, "Session_ID", "session_id", "session")  # optional

require_cols(data, [press_col, release_col])

# Keep a compact frame with normalized names
df = pd.DataFrame({
    "Press_Time":   data[press_col].astype(float),
    "Release_Time": data[release_col].astype(float),
})
if key_col:  df["Key"]       = data[key_col]
if user_col: df["User_ID"]   = data[user_col]
if sess_col: df["Session_ID"]= data[sess_col]

# Drop rows with missing press/release
df = df.dropna(subset=["Press_Time", "Release_Time"]).copy()

# Fix obvious issues: if Release < Press, swap or set hold to NaN
# We'll correct holds later by clipping to >=0
# Sort within sessions if we have IDs
if user_col and sess_col:
    df.sort_values(by=["User_ID", "Session_ID", "Press_Time", "Release_Time"], inplace=True, kind="mergesort")
elif user_col:
    df.sort_values(by=["User_ID", "Press_Time", "Release_Time"], inplace=True, kind="mergesort")
else:
    df.sort_values(by=["Press_Time", "Release_Time"], inplace=True, kind="mergesort")

# ---------------------------
# 3) Temporal features: Hold, DD, UD (per session if available)
# ---------------------------
def add_dd_ud(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["Hold_Time"] = (g["Release_Time"] - g["Press_Time"]).clip(lower=0)

    # Previous press and release (to compute intervals to current press)
    g["Prev_Press"]   = g["Press_Time"].shift(1)
    g["Prev_Release"] = g["Release_Time"].shift(1)

    # DD: current Press - previous Press
    g["DD"] = (g["Press_Time"] - g["Prev_Press"]).astype(float)
    # UD: current Press - previous Release
    g["UD"] = (g["Press_Time"] - g["Prev_Release"]).astype(float)

    # Fill first-row NaNs with group medians (fallback to overall medians if needed)
    for col in ["DD", "UD"]:
        med = g[col].median()
        if pd.isna(med):
            med = df[col].median() if col in df.columns else 0.0
        g[col] = g[col].fillna(med)

    # Clean extreme negatives (shouldn’t happen after sorting; still guard)
    g["DD"] = g["DD"].clip(lower=0)
    g["UD"] = g["UD"].clip(lower=0)
    return g.drop(columns=["Prev_Press", "Prev_Release"])

if user_col and sess_col:
    df = df.groupby(["User_ID", "Session_ID"], group_keys=False).apply(add_dd_ud)
elif user_col:
    df = df.groupby(["User_ID"], group_keys=False).apply(add_dd_ud)
else:
    df = add_dd_ud(df)

# Keep only the features we will use
FEATURES = ["Hold_Time", "DD", "UD"]
X_raw = df[FEATURES].astype(float).to_numpy()

# ---------------------------
# 4) Labels (example): good/bad based on Hold median
#    Replace with your true labels when ready.
# ---------------------------
hold_med = np.median(X_raw[:, 0])
y_raw = (X_raw[:, 0] > hold_med).astype(np.int64)

# ---------------------------
# 5) Scale + window to sequences
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

SEQ_LEN = 32
if X_scaled.shape[0] <= SEQ_LEN:
    raise ValueError(f"Not enough rows ({X_scaled.shape[0]}) to build sequences of length {SEQ_LEN}")

num_samples = X_scaled.shape[0] - SEQ_LEN
X_seq = np.stack([X_scaled[i:i+SEQ_LEN] for i in range(num_samples)], axis=0)  # [N, 32, F]
y_seq = np.stack([y_raw[i:i+SEQ_LEN]   for i in range(num_samples)], axis=0)  # [N, 32]

# Split: 70/15/15 (train/val/test)
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.30, random_state=42, shuffle=True)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True)

# ---------------------------
# 6) Save arrays & config
# ---------------------------
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

with open(os.path.join(PROCESSED_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "seq_len": SEQ_LEN,
            "features": FEATURES,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist()
        },
        f,
        indent=2
    )

print("Preprocessing complete.")
print("Saved to:", PROCESSED_DIR)
print("Shapes ->",
      "X_train", X_train.shape,
      "X_val",   X_val.shape,
      "X_test",  X_test.shape)
