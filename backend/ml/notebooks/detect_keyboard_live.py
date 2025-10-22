import cv2
import torch
from ultralytics import YOLO
import mediapipe as mp
import time
import numpy as np
import os
import csv
from datetime import datetime
from pynput.keyboard import Key, Listener
import sys

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL_PATH = "runs/train/keyboard_key_detector/weights/best.pt"
BILSTM_MODEL_PATH = r"D:\pd-keyboard-app\backend\ml\saved\bilstm_highacc_bestfold.pth"  # your trained model
CONF_THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
SEQ_LEN = 1  # since we predict per press, no sequence buffer

SAVE_DIR = r"D:\pd-keyboard-app\backend\ml\results_csv"
os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"finger_key_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ============================================================
# CSV HEADER
# ============================================================
csv_header = [
    "timestamp", "pressed_key", "finger_name", "hand_used",
    "finger_x", "finger_y", "key_x1", "key_y1", "key_x2", "key_y2",
    "dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance",
    "predicted_label"
]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_header)
print(f"CSV logging enabled → {csv_path}")

# ============================================================
# LOAD YOLO + MEDIAPIPE
# ============================================================
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(DEVICE)
print(f"YOLOv8 model loaded on {DEVICE}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ============================================================
# LOAD BILSTM MODEL
# ============================================================
class BiLSTMCRF(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=2,
                                  dropout=dropout, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

bilstm_model = BiLSTMCRF(input_dim=3)  # since input is pressed_key, finger_name, hand_used (encoded)
bilstm_model.load_state_dict(torch.load(BILSTM_MODEL_PATH, map_location=DEVICE))
bilstm_model.to(DEVICE)
bilstm_model.eval()
print("BiLSTM model loaded for correctness prediction.")

# ============================================================
# SPECIAL KEY MAP
# ============================================================
SPECIAL_KEY_MAP = {
    ';': 'semicolon', ':': 'semicolon', ',': 'comma', '.': 'period', '/': 'slash',
    '?': 'slash', '[': 'bracket-left', ']': 'bracket-right', '\\': 'backslash',
    "'": 'quote', '"': 'quote', '`': 'backtick', '-': 'minus', '_': 'minus',
    '=': 'equal', '+': 'equal', Key.space: 'space', Key.enter: 'enter',
    Key.backspace: 'backspace', Key.tab: 'tab', Key.caps_lock: 'caps-lock'
}

pressed_key, last_logged_key = None, None

def on_press(key):
    global pressed_key
    if key in SPECIAL_KEY_MAP:
        pressed_key = SPECIAL_KEY_MAP[key]
    else:
        try:
            pressed_key = key.char.lower() if key.char else None
        except AttributeError:
            pressed_key = SPECIAL_KEY_MAP.get(str(key), str(key).replace("Key.", "").lower())

def on_release(key):
    global pressed_key
    pressed_key = None

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

# ============================================================
# CAMERA SETUP
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# ============================================================
# KEYBOARD CALIBRATION
# ============================================================
print("Calibrating keyboard layout... remove hands from frame.")
key_positions = {}
for _ in range(20):
    ret, frame = cap.read()
    if not ret:
        continue
    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        if label.lower() == "keyboard":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        key_positions[label] = (x1, y1, x2, y2)
print(f"Locked {len(key_positions)} key boxes:", list(key_positions.keys()))

# ============================================================
# LIVE LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb)
    detected_fingers = []

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, hand_label in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            hand_used = "Left" if hand_label.classification[0].label == "Right" else "Right"
            color = (0, 128, 255) if hand_used == "Left" else (255, 128, 0)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for name, idx in {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}.items():
                fx = int(hand_landmarks.landmark[idx].x * FRAME_WIDTH)
                fy = int(hand_landmarks.landmark[idx].y * FRAME_HEIGHT)
                detected_fingers.append((name, fx, fy, hand_used))
                cv2.circle(frame, (fx, fy), 6, color, -1)

    # ========================================================
    # WHEN KEY IS PRESSED → RUN INFERENCE
    # ========================================================
    if pressed_key and pressed_key != last_logged_key:
        last_logged_key = pressed_key
        if pressed_key in key_positions:
            x1, y1, x2, y2 = key_positions[pressed_key]
            key_cx, key_cy = (x1 + x2)//2, (y1 + y2)//2

            min_dist = float("inf")
            fx, fy, closest_finger, hand_used = 0, 0, None, None
            for (fname, cx, cy, hand_label) in detected_fingers:
                dist = np.sqrt((cx - key_cx)**2 + (cy - key_cy)**2)
                if dist < min_dist:
                    min_dist, fx, fy, closest_finger, hand_used = dist, cx, cy, fname, hand_label

            if closest_finger:
                # Encode categorical features manually (0/1 style)
                pressed_enc = hash(pressed_key) % 1000 / 1000
                finger_enc = hash(closest_finger) % 1000 / 1000
                hand_enc = 1.0 if hand_used == "Right" else 0.0

                # Shape: (1, seq_len, 3)
                x_tensor = torch.tensor([[[pressed_enc, finger_enc, hand_enc]]], dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    pred = bilstm_model(x_tensor).item()
                pred_label = "Correct" if pred > 0.5 else "Incorrect"
                color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)

                cv2.putText(frame, pred_label, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                # Log CSV
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        time.time(), pressed_key, closest_finger, hand_used,
                        fx, fy, x1, y1, x2, y2, 0, 0, min_dist, 0, 0, 0, pred_label
                    ])
                print(f"{pressed_key} | {hand_used}-{closest_finger} → {pred_label} ({pred:.3f})")

    elif not pressed_key:
        last_logged_key = None

    cv2.imshow("YOLO + BiLSTM Typing Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()
print(f"Session complete. Data saved to: {csv_path}")
