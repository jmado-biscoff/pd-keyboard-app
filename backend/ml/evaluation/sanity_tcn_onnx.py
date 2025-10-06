import os, json, numpy as np, onnxruntime as ort

# ---- Paths (adjust if needed)
MODEL_PATH = "backend/ml/models/tcn_model.onnx"
FEAT_CFG   = "backend/ml/dataset/processed/feature_config.json"
X_TEST     = "backend/ml/dataset/processed/X_test.npy"

assert os.path.exists(MODEL_PATH), f"Missing ONNX: {MODEL_PATH}"
assert os.path.exists(FEAT_CFG),   f"Missing feature config: {FEAT_CFG}"
assert os.path.exists(X_TEST),     f"Missing: {X_TEST}"

# ---- Load
cfg = json.load(open(FEAT_CFG, "r", encoding="utf-8"))
seq_len = int(cfg["seq_len"])
features = cfg["features"]
print(f"Feature config -> seq_len={seq_len}, features={features}")

X = np.load(X_TEST).astype(np.float32)   # shape: [N, seq_len, n_features]
N, T, F = X.shape
print("X_test shape:", X.shape)

# ---- Pick one sequence
sample = X[0:1]   # [1, T, F]
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

inp_name  = sess.get_inputs()[0].name
out_name  = sess.get_outputs()[0].name
print("ONNX IO:", inp_name, "->", out_name)

# ---- Run
out = sess.run([out_name], {inp_name: sample})[0]   # [1, T, C]
print("Output shape:", out.shape)
print("First 5 logits (t=0..4):")
print(out[0, :5, :])
