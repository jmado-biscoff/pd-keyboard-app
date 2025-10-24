# ============================================================
# detect_keyboard_live_rule_based_letters_punct.py — FINAL VERSION (with expected key logic)
# ============================================================

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import os, csv, time, json, sys
from datetime import datetime
import keyboard
import threading

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL_PATH = r"runs/train/keyboard_key_detector/weights/best.pt"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 1
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

SAVE_DIR = r"E:\pd-keyboard-app\backend\ml\results_csv"
EXPECTED_PATH = os.path.join(SAVE_DIR, "expected_words.json")
os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(
    SAVE_DIR, f"finger_key_rulebased_letters_punct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

# ============================================================
# LOAD EXPECTED KEYS
# ============================================================
expected_keys = []
if os.path.exists(EXPECTED_PATH):
    with open(EXPECTED_PATH, "r") as f:
        try:
            data = json.load(f)
            words = data.get("words", [])
            for i, word in enumerate(words):
                for ch in word:
                    expected_keys.append(ch.lower())
                if i < len(words) - 1:
                    expected_keys.append("space")
            print(f"✅ Loaded {len(expected_keys)} expected keys from expected_words.json")
        except Exception as e:
            print(f"⚠️ Failed to load expected keys: {e}")
else:
    print("⚠️ No expected_words.json found — proceeding without target sequence.")
current_expected_index = 0

# ============================================================
# CSV HEADER
# ============================================================
csv_header = [
    "timestamp", "pressed_key", "finger_name", "hand_used",
    "finger_x", "finger_y", "key_x1", "key_y1", "key_x2", "key_y2",
    "dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance",
    "rule_based_label"
]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_header)
print(f"✅ CSV logging enabled → {csv_path}")
sys.stdout.flush()

# ============================================================
# LOAD YOLO + MEDIAPIPE
# ============================================================
print("Loading YOLOv8 keyboard model...")
sys.stdout.flush()
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLOv8 model loaded successfully.")
sys.stdout.flush()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ============================================================
# RULE-BASED TOUCH TYPING MAP
# ============================================================
TOUCH_TYPING_MAP = {
    "a": ("left", "pinky"), "s": ("left", "ring"), "d": ("left", "middle"), "f": ("left", "index"),
    "g": ("left", "index"), "h": ("right", "index"), "j": ("right", "index"), "k": ("right", "middle"),
    "l": ("right", "ring"),
    "q": ("left", "pinky"), "w": ("left", "ring"), "e": ("left", "middle"), "r": ("left", "index"),
    "t": ("left", "index"), "y": ("right", "index"), "u": ("right", "index"), "i": ("right", "middle"),
    "o": ("right", "ring"), "p": ("right", "pinky"),
    "z": ("left", "pinky"), "x": ("left", "ring"), "c": ("left", "middle"), "v": ("left", "index"),
    "b": ("left", "index"), "n": ("right", "index"), "m": ("right", "index"),
    "1": ("left", "pinky"), "2": ("left", "ring"), "3": ("left", "middle"), "4": ("left", "index"),
    "5": ("left", "index"), "6": ("right", "index"), "7": ("right", "index"), "8": ("right", "middle"),
    "9": ("right", "ring"), "0": ("right", "pinky"),
    "space": ("both", "thumb"), "enter": ("right", "pinky"), "tab": ("left", "pinky"),
    "caps-lock": ("left", "pinky"), "shift": ("left", "pinky"), "ctrl": ("left", "pinky"),
    "alt": ("right", "thumb"), "backspace": ("right", "pinky"),
    ",": ("right", "middle"), ".": ("right", "ring"), "/": ("right", "pinky"),
    "?": ("right", "pinky"), "!": ("left", "pinky")
}

# ============================================================
# INPUT HANDLER
# ============================================================
pressed_key = None
last_logged_key = None

def key_listener():
    global pressed_key
    while True:
        event = keyboard.read_event(suppress=False)
        if event.event_type == keyboard.KEY_DOWN:
            name = event.name
            if name == "space":
                pressed_key = "space"
            elif name == "enter":
                pressed_key = "enter"
            elif name == "backspace":
                pressed_key = "backspace"
            elif len(name) == 1:
                pressed_key = name
            else:
                pressed_key = None
        elif event.event_type == keyboard.KEY_UP:
            pressed_key = None

listener_thread = threading.Thread(target=key_listener, daemon=True)
listener_thread.start()

# ============================================================
# CAMERA SETUP
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    sys.stdout.flush()
    exit()

# ============================================================
# KEYBOARD CALIBRATION
# ============================================================
print("🔧 Calibrating keyboard layout... remove hands from frame.")
sys.stdout.flush()
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

# ============================================================
# LIVE LOOP — RULE-BASED + EXPECTED KEY FEEDBACK
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

    # Get current expected key
    expected_key = None
    if current_expected_index < len(expected_keys):
        expected_key = expected_keys[current_expected_index]

    if pressed_key and pressed_key != last_logged_key:
        last_logged_key = pressed_key
        key = pressed_key.lower()

        if key in TOUCH_TYPING_MAP and key in key_positions and detected_fingers:
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
                norm_dx, norm_dy = dx / FRAME_WIDTH, dy / FRAME_HEIGHT
                norm_dist = min_dist / np.sqrt(FRAME_WIDTH**2 + FRAME_HEIGHT**2)

                expected_hand, expected_finger = TOUCH_TYPING_MAP.get(key, ("unknown", "unknown"))
                rule_based_correct = (expected_finger == closest_finger and (expected_hand == hand_used or expected_hand == "both"))

                # ✅ JSON correctness based on expected key matching
                json_correct = (expected_key == key) if expected_key else True

                # ✅ Combine rule-based + expected-key correctness (both must be true)
                overall_correct = rule_based_correct and json_correct
                rule_label = "Correct" if overall_correct else "Incorrect"
                color = (0, 255, 0) if overall_correct else (0, 0, 255)

                cv2.putText(frame, f"{key.upper()} | Exp: {expected_key or '-'} | {rule_label}",
                            (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                # Log CSV
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        time.time(), key, closest_finger, hand_used,
                        fx, fy, x1, y1, x2, y2, dx, dy, min_dist,
                        norm_dx, norm_dy, norm_dist, rule_label
                    ])

                # ✅ JSON result (for Node)
                result = {
                    "expected_key": expected_key.upper() if expected_key else None,
                    "key": key.upper(),
                    "finger": closest_finger,
                    "hand": hand_used,
                    "correct": overall_correct
                }
                print(json.dumps(result))
                sys.stdout.flush()

                # ✅ Advance expected pointer
                if expected_key is not None:
                    current_expected_index += 1

    elif not pressed_key:
        last_logged_key = None

    cv2.imshow("YOLO + Rule-Based Typing Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Session complete. Data saved to: {csv_path}")
sys.stdout.flush()
