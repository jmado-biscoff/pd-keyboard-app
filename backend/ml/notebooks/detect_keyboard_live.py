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

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "runs/train/keyboard_key_detector/weights/best.pt"
CONF_THRESHOLD = 0.4
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CAMERA_INDEX = 1
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

SAVE_DIR = r"C:\Users\Julienne\pd-keyboard-app\backend\ml\results_csv"
os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"finger_key_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ============================================================
# CSV HEADER
# ============================================================
csv_header = [
    "timestamp",
    "pressed_key",
    "finger_name",
    "hand_used",
    "finger_x",
    "finger_y",
    "key_x1",
    "key_y1",
    "key_x2",
    "key_y2",
    "dx",
    "dy",
    "distance",
    "norm_dx",
    "norm_dy",
    "norm_distance"
]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_header)
print(f"📁 CSV logging enabled. Saving live data to:\n{csv_path}")

# ============================================================
# LOAD MODELS
# ============================================================
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"✅ Loaded YOLOv8 model on {DEVICE}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ============================================================
# CAMERA SETUP
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()

# ============================================================
# SPECIAL KEY MAP
# ============================================================
SPECIAL_KEY_MAP = {
    ';': 'semicolon',
    ':': 'semicolon',
    ',': 'comma',
    '.': 'period',
    '/': 'slash',
    '?': 'slash',
    '[': 'bracket-left',
    ']': 'bracket-right',
    '\\': 'backslash',
    "'": 'quote',
    '"': 'quote',
    '`': 'backtick',
    '-': 'minus',
    '_': 'minus',
    '=': 'equal',
    '+': 'equal',

    # Modifier and functional keys
    Key.shift: 'shift-left',
    Key.shift_l: 'shift-left',
    Key.shift_r: 'shift-right',
    Key.ctrl_l: 'ctrl-left',
    Key.ctrl_r: 'ctrl-right',
    Key.alt_l: 'alt-left',
    Key.alt_r: 'alt-right',
    Key.space: 'space',
    Key.enter: 'enter',
    Key.backspace: 'backspace',
    Key.tab: 'tab',
    Key.caps_lock: 'caps-lock'
}

# ============================================================
# KEYBOARD EVENT LISTENER
# ============================================================
pressed_key = None
last_logged_key = None

def on_press(key):
    """Handle hardware key and character mapping"""
    global pressed_key

    # 1️⃣ Handle special Key enums directly
    if key in SPECIAL_KEY_MAP:
        pressed_key = SPECIAL_KEY_MAP[key]
        return

    # 2️⃣ Handle printable characters and punctuation
    try:
        char = key.char
        if char in SPECIAL_KEY_MAP:
            pressed_key = SPECIAL_KEY_MAP[char]
        elif char is not None:
            pressed_key = char.lower()
    except AttributeError:
        # Fallback for unhandled keys
        pressed_key = SPECIAL_KEY_MAP.get(str(key), str(key).replace("Key.", "").lower())

def on_release(key):
    global pressed_key
    pressed_key = None

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

# ============================================================
# CALIBRATION PHASE
# ============================================================
print("🧭 Calibrating keyboard layout... (remove your hands)")
key_positions = {}
for i in range(20):
    ret, frame = cap.read()
    if not ret:
        continue
    results = model.predict(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        if label.lower() == "keyboard":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > 0.3 * (FRAME_WIDTH * FRAME_HEIGHT):
            continue
        key_positions[label] = (x1, y1, x2, y2)

print(f"✅ Locked {len(key_positions)} key boxes:", list(key_positions.keys()))
print("✅ Calibration complete. Bounding boxes are fixed.")

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    assert w == FRAME_WIDTH and h == FRAME_HEIGHT, "Frame size mismatch — check camera resolution."

    # ============================================================
    # HAND DETECTION (NORMALIZED SCALE)
    # ============================================================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb)
    detected_fingers = []

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, hand_label in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            raw_label = hand_label.classification[0].label
            label_str = "Left" if raw_label == "Right" else "Right"  # flipped for top-down

            color = (0, 128, 255) if label_str == "Left" else (255, 128, 0)
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=color, thickness=2)
            )

            fingertip_indices = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
            for fname, idx in fingertip_indices.items():
                fx = int(hand_landmarks.landmark[idx].x * FRAME_WIDTH)
                fy = int(hand_landmarks.landmark[idx].y * FRAME_HEIGHT)
                detected_fingers.append((fname, fx, fy, label_str))
                cv2.circle(frame, (fx, fy), 6, color, -1)
                cv2.putText(frame, f"{label_str[0]}-{fname[0]}", (fx + 5, fy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ============================================================
    # LOGGING ON KEY PRESS
    # ============================================================
    if pressed_key and pressed_key != last_logged_key:
        last_logged_key = pressed_key

        if pressed_key in key_positions:
            x1, y1, x2, y2 = key_positions[pressed_key]
            key_cx, key_cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Find nearest fingertip
            min_dist = float("inf")
            closest_finger, hand_used, fx, fy = None, None, 0, 0
            for (fname, cx, cy, hand_label) in detected_fingers:
                dist = np.sqrt((cx - key_cx)**2 + (cy - key_cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_finger, hand_used, fx, fy = fname, hand_label, cx, cy

            if closest_finger:
                dx = fx - key_cx
                dy = fy - key_cy
                distance = np.sqrt(dx**2 + dy**2)

                # Normalize distances
                norm_dx = dx / FRAME_WIDTH
                norm_dy = dy / FRAME_HEIGHT
                norm_distance = distance / np.sqrt(FRAME_WIDTH**2 + FRAME_HEIGHT**2)

                timestamp = time.time()
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        timestamp,
                        pressed_key,
                        closest_finger,
                        hand_used,
                        fx,
                        fy,
                        x1,
                        y1,
                        x2,
                        y2,
                        dx,
                        dy,
                        distance,
                        norm_dx,
                        norm_dy,
                        norm_distance
                    ])
                print(f"🖱 Logged: {pressed_key} | {hand_used}-{closest_finger} | Dist: {distance:.2f}px | Norm: {norm_distance:.4f}")

    elif not pressed_key:
        last_logged_key = None

    cv2.imshow("Keyboard-Finger Data Logger (Normalized + Punctuation Fix)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()
print(f"✅ Data saved to: {csv_path}")
