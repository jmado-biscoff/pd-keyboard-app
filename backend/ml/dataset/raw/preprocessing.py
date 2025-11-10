"""
preprocess_finger_key.py
Preprocesses the finger-to-key dataset for training and evaluation.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib  # to save encoder/scaler

# 📂 Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "finger_key_dataset_.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "results")

# 🧩 Load dataset
data = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

# 🎯 Define feature and target columns
categorical_features = ["pressed_key", "finger_name", "hand_used"]
numeric_features = ["dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance"]
target_column = "is_correct"  # adjust if needed

# Separate input and label
X = data[categorical_features + numeric_features]
y = data[target_column]

# 🔡 Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cats = encoder.fit_transform(X[categorical_features])
encoded_cols = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols)

# 🔢 Scale numeric features
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(X[numeric_features])
scaled_df = pd.DataFrame(scaled_nums, columns=[f"{col}_scaled" for col in numeric_features])

# 🧩 Combine processed features
X_processed = pd.concat([encoded_df, scaled_df], axis=1)

# 🧪 Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 💾 Save outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

# 💾 Save encoder & scaler for inference
joblib.dump(encoder, os.path.join(OUTPUT_DIR, "encoder.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print("Preprocessing complete!")
print(f"Processed data & models saved to: {OUTPUT_DIR}")
