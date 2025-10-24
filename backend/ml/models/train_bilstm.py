# ============================================================
# train_bilstm_correctness_full_11feat_v2.py
# Author: PD Team Seven
# Purpose: Train 11-feature BiLSTM model (8 numeric + 3 categorical)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import joblib, os, time

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = r"E:\pd-keyboard-app\backend\ml\dataset\raw\synthetic_finger_key_dataset_fixed.csv"
SAVE_DIR  = r"E:\pd-keyboard-app\backend\ml\saved"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
BATCH_SIZE = 64
LR = 1e-3
K_FOLDS = 5
PRINT_INTERVAL = 5

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset with {len(df)} samples and {len(df.columns)} columns.")

# Expected numeric + categorical columns
num_features = ["dx","dy","norm_dx","norm_dy","norm_distance"]
cat_features = ["pressed_key","finger_name","hand_used"]

# Add derived geometric features
df["angle"] = np.degrees(np.arctan2(df["dy"], df["dx"]))
df["abs_dx"] = df["dx"].abs()
df["abs_dy"] = df["dy"].abs()

extra_numeric = ["angle","abs_dx","abs_dy"]
num_features += extra_numeric  # total = 8 numeric
print(f"🧩 Numeric features: {num_features}")

# ============================================================
# ENCODERS (categorical)
# ============================================================
all_keys = list("abcdefghijklmnopqrstuvwxyz") + [
    "space","enter","shift","tab","caps-lock","ctrl","alt","backspace",
    "semicolon","comma","period","slash","quote","minus","equal",
    "bracket-left","bracket-right","backslash","backtick",
    "1","2","3","4","5","6","7","8","9","0"
]
all_fingers = ["thumb","index","middle","ring","pinky"]
all_hands = ["left","right"]

encoders = {}
for col, classes in zip(cat_features,[all_keys,all_fingers,all_hands]):
    le = LabelEncoder()
    le.fit(classes)
    encoders[col] = le
    df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
    df[col] = le.transform(df[col])

joblib.dump(encoders, os.path.join(SAVE_DIR,"encoders.pkl"))
print("✅ Encoders saved.")

# ============================================================
# SCALER (fit 8 numeric features)
# ============================================================
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
joblib.dump(scaler, os.path.join(SAVE_DIR,"scaler_8feat.pkl"))
print("✅ Scaler (8 features) saved.")

# ============================================================
# DATASET PREPARATION
# ============================================================
X = df[num_features + cat_features].values
y = df["is_correct"].values

class FingerKeyDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

dataset = FingerKeyDataset(X,y)

# ============================================================
# MODEL
# ============================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self,input_dim=11,hidden_dim=256,num_layers=3,dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,hidden_dim,num_layers=num_layers,
            batch_first=True,bidirectional=True,dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_dim*2,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.fc3(out)
        return self.sigmoid(out)

# ============================================================
# TRAINING LOOP WITH K-FOLD
# ============================================================
criterion = nn.BCELoss()
skf = StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=42)
best_acc, best_path = 0.0, os.path.join(SAVE_DIR,"bilstm_spatial_11feat.pth")

for fold,(train_idx,val_idx) in enumerate(skf.split(X,y)):
    print(f"\n==========================")
    print(f"🧩 Fold {fold+1}/{K_FOLDS}")
    print("==========================")
    train_loader = DataLoader(Subset(dataset,train_idx),batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(Subset(dataset,val_idx),batch_size=BATCH_SIZE,shuffle=False)

    model = BiLSTMClassifier(input_dim=11).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for Xb,yb in train_loader:
            Xb,yb = Xb.to(DEVICE),yb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out,yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out>0.5).eq(yb).sum().item()
            total += yb.size(0)
        train_acc = correct/total
        avg_loss = total_loss/len(train_loader)

        # Validation
        model.eval()
        preds,truths=[],[]
        with torch.no_grad():
            for Xb,yb in val_loader:
                Xb,yb = Xb.to(DEVICE),yb.to(DEVICE).unsqueeze(1)
                out = model(Xb)
                preds.extend((out>0.5).float().cpu().numpy().flatten())
                truths.extend(yb.cpu().numpy().flatten())
        acc = accuracy_score(truths,preds)

        if epoch % PRINT_INTERVAL == 0 or epoch == 1:
            print(f"Fold {fold+1} | Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | TrainAcc: {train_acc*100:.2f}% | ValAcc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

print(f"\n✅ Training complete.")
print(f"✅ Best validation accuracy: {best_acc*100:.2f}%")
print(f"✅ Best model saved to: {best_path}")
