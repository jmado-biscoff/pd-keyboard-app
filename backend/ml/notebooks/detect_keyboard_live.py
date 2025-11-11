# ============================================================
# detect_keyboard_live_rule_based_letters_punct.py — FULL INTEGRATION (YOLO + Mediapipe + SVM + Encoder + Scaler)
# ============================================================

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import os, csv, time, json, sys
from datetime import datetime
import keyboard
import threading
import joblib
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL_PATH = r"D:\pd-keyboard-app\backend\ml\notebooks\runs\train\keyboard_key_detector\weights\best.pt"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
DEBOUNCE_INTERVAL = 0.25

SAVE_DIR = r"D:\pd-keyboard-app\backend\ml\testing"
RESULTS_DIR = r"D:\pd-keyboard-app\backend\ml\results"

os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"finger_key_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

EXPECTED_PATH = os.path.join(SAVE_DIR, "expected_words.json")

# ============================================================
# LOAD EXPECTED KEYS
# ============================================================
expected_keys = []
if os.path.exists(EXPECTED_PATH):
    with open(EXPECTED_PATH, "r") as f:
        try:
            data = json.load(f)
            for word in data.get("words", []):
                expected_keys.extend([ch.lower() for ch in word])
            print(f"✅ Loaded {len(expected_keys)} expected keys (SPACE excluded)")
        except Exception as e:
            print(f"⚠️ Failed to load expected keys: {e}")
else:
    print("⚠️ No expected_words.json found — proceeding without target sequence.")
current_expected_index = 0

# ============================================================
# LOAD MODELS
# ============================================================
try:
    svm_model = joblib.load(os.path.join(RESULTS_DIR, "svm_model.pkl"))
    encoder = joblib.load(os.path.join(RESULTS_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(RESULTS_DIR, "scaler.pkl"))
    print("✅ SVM, Encoder, and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model/encoder/scaler: {e}")
    sys.exit(1)

# ============================================================
# INIT YOLO + MEDIAPIPE
# ============================================================
print("Loading YOLOv8 keyboard model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLOv8 model loaded successfully.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ============================================================
# TOUCH TYPING MAP
# ============================================================
TOUCH_TYPING_MAP = {
    "a": ("left", "pinky"), "s": ("left", "ring"), "d": ("left", "middle"), "f": ("left", "index"),
    "g": ("left", "index"), "h": ("right", "index"), "j": ("right", "index"), "k": ("right", "middle"),
    "l": ("right", "ring"), "q": ("left", "pinky"), "w": ("left", "ring"), "e": ("left", "middle"),
    "r": ("left", "index"), "t": ("left", "index"), "y": ("right", "index"), "u": ("right", "index"),
    "i": ("right", "middle"), "o": ("right", "ring"), "p": ("right", "pinky"), "z": ("left", "pinky"),
    "x": ("left", "ring"), "c": ("left", "middle"), "v": ("left", "index"), "b": ("left", "index"),
    "n": ("right", "index"), "m": ("right", "index"),
}

# ============================================================
# KEYBOARD INPUT HANDLER
# ============================================================
pressed_key = None
last_logged_key = None
last_sent_time = 0

def key_listener():
    global pressed_key
    last_key = None
    while True:
        event = keyboard.read_event(suppress=False)
        if event.event_type == keyboard.KEY_DOWN:
            name = event.name
            if len(name) == 1:
                key_name = name
            elif name in ["space", "enter", "backspace"]:
                key_name = name
            else:
                key_name = None

            if key_name and key_name != last_key:
                pressed_key = key_name
                last_key = key_name
        elif event.event_type == keyboard.KEY_UP:
            pressed_key = None
            last_key = None

threading.Thread(target=key_listener, daemon=True).start()

# ============================================================
# CAMERA INITIALIZATION + CALIBRATION
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("❌ Cannot open camera.")
    sys.exit()

# ============================================================
# KEYBOARD CALIBRATION + TEMPORARY BOUNDING BOX DISPLAY
# ============================================================
print("🔧 Calibrating keyboard layout... remove hands from frame.")
key_positions = {}
for _ in range(20):
    ret, frame = cap.read()
    if not ret:
        continue
    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id].lower()
        if label == "keyboard":
            continue
        key_positions[label] = tuple(map(int, box.xyxy[0]))

print(f"✅ Locked {len(key_positions)} key boxes detected.")
sys.stdout.flush()

# ✅ Display bounding boxes for 3 seconds for visual confirmation
start_time = time.time()
while time.time() - start_time < 5:  # show for 3 seconds
    ret, frame = cap.read()
    if not ret:
        continue
    for key, (x1, y1, x2, y2) in key_positions.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, key.upper(), (x1 + 3, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "Calibration complete — starting in 3s...",
                (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("YOLO + Mediapipe + SVM Typing Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CSV HEADER
# ============================================================
csv_header = [
    "timestamp", "pressed_key", "finger_name", "hand_used",
    "dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance",
    "rule_based_label", "ml_label"
]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_header)

# ============================================================
# MAIN DETECTION LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb)
    detected_fingers = []

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            x_mean = np.mean([lm.x for lm in hand_landmarks.landmark])
            hand_used = "left" if x_mean < 0.5 else "right"
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for name, idx in {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}.items():
                fx = int(hand_landmarks.landmark[idx].x * FRAME_WIDTH)
                fy = int(hand_landmarks.landmark[idx].y * FRAME_HEIGHT)
                detected_fingers.append((name, fx, fy, hand_used))
                cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1)

    expected_key = expected_keys[current_expected_index] if current_expected_index < len(expected_keys) else None

    now = time.time()
    if pressed_key and (pressed_key != last_logged_key or (now - last_sent_time) > DEBOUNCE_INTERVAL):
        key = pressed_key.lower()
        last_logged_key = key
        last_sent_time = now

        if key == "space" or key not in TOUCH_TYPING_MAP:
            continue

        if key in key_positions and detected_fingers:
            x1, y1, x2, y2 = key_positions[key]
            key_cx, key_cy = (x1 + x2) // 2, (y1 + y2) // 2

            fx, fy, closest_finger, hand_used = None, None, None, None
            min_dist = float("inf")
            for fname, cx, cy, hlabel in detected_fingers:
                dist = np.sqrt((cx - key_cx)**2 + (cy - key_cy)**2)
                if dist < min_dist:
                    min_dist, fx, fy, closest_finger, hand_used = dist, cx, cy, fname, hlabel

            if closest_finger:
                dx, dy = fx - key_cx, fy - key_cy
                distance = np.sqrt(dx**2 + dy**2)
                norm_dx, norm_dy = dx / FRAME_WIDTH, dy / FRAME_HEIGHT
                norm_distance = distance / np.sqrt(FRAME_WIDTH**2 + FRAME_HEIGHT**2)

                # Rule-based label
                expected_hand, expected_finger = TOUCH_TYPING_MAP.get(key, ("unknown", "unknown"))
                rule_based_correct = (expected_finger == closest_finger and (expected_hand == hand_used))
                rule_label = "Correct" if rule_based_correct else "Incorrect"

                # ============================================================
                # ML INFERENCE USING ENCODER + SCALER + SVM
                # ============================================================
                cat_df = pd.DataFrame([[key.upper(), closest_finger.title(), hand_used.title()]],
                                    columns=["pressed_key", "finger_name", "hand_used"])

                num_df = pd.DataFrame([[dx, dy, min_dist, norm_dx, norm_dy, norm_distance]],
                                    columns=["dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance"])

                encoded = pd.DataFrame(encoder.transform(cat_df),
                                       columns=encoder.get_feature_names_out(["pressed_key", "finger_name", "hand_used"]))
                scaled = pd.DataFrame(scaler.transform(num_df),
                                      columns=[f"{col}_scaled" for col in num_df.columns])

                features = pd.concat([encoded, scaled], axis=1)

                ml_pred = svm_model.predict(features)[0]
                ml_label = "Correct" if ml_pred == 1 else "Incorrect"

                # Display feedback
                color = (0, 255, 0) if ml_label == "Correct" else (0, 0, 255)
                cv2.putText(frame, f"{key.upper()} | ML: {ml_label} | RB: {rule_label}",
                            (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                # Save to CSV
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        time.time(), key, closest_finger, hand_used,
                        dx, dy, distance, norm_dx, norm_dy, norm_distance,
                        rule_label, ml_label
                    ])

                print(json.dumps({
                    "key": key.upper(),
                    "finger": closest_finger,
                    "hand": hand_used,
                    "rule_label": rule_label,
                    "ml_label": ml_label
                }))
                sys.stdout.flush()

    elif not pressed_key:
        last_logged_key = None

    cv2.imshow("YOLO + Mediapipe + SVM Typing Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Session complete. Data saved to: {csv_path}")
