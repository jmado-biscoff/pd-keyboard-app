# ============================================================
# detect_keyboard_live_rule_based_letters_punct.py — AUTO-RECALIBRATING VERSION
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
DRIFT_THRESHOLD = 5  # pixels — small movement triggers recalibration

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
            for word in words:
                for ch in word:
                    expected_keys.append(ch.lower())
            print(f"✅ Loaded {len(expected_keys)} expected keys (SPACE excluded)")
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
}

# ============================================================
# KEYBOARD INPUT HANDLER
# ============================================================
pressed_key = None
last_logged_key = None
last_sent_time = 0
DEBOUNCE_INTERVAL = 0.25  # seconds

def key_listener():
    global pressed_key
    last_key = None
    while True:
        event = keyboard.read_event(suppress=False)
        if event.event_type == keyboard.KEY_DOWN:
            name = event.name
            key_name = name if len(name) == 1 else (name if name in ["space", "enter", "backspace", "shift", "ctrl", "tab", "caps lock"] else None)
            if key_name and key_name != last_key:
                pressed_key = key_name
                last_key = key_name
        elif event.event_type == keyboard.KEY_UP:
            pressed_key = None
            last_key = None

threading.Thread(target=key_listener, daemon=True).start()

# ============================================================
# CAMERA SETUP
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    sys.exit()

# ============================================================
# CALIBRATION FUNCTION
# ============================================================
def calibrate_keyboard():
    print("🔧 Calibrating keyboard layout... remove hands from frame.")
    sys.stdout.flush()
    detected = {}
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
            detected[label] = tuple(map(int, box.xyxy[0]))
    print(f"✅ Locked {len(detected)} key boxes detected.")
    sys.stdout.flush()
    return detected

key_positions = calibrate_keyboard()
prev_keyboard_box = None

# ============================================================
# FUNCTION: CHECK FOR DRIFT
# ============================================================
def has_drifted(prev_box, new_box):
    if prev_box is None or new_box is None:
        return True
    diff = np.abs(np.array(prev_box) - np.array(new_box))
    return np.any(diff > DRIFT_THRESHOLD)

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    # Track keyboard position
    current_keyboard_box = None
    for box in results[0].boxes:
        label = results[0].names[int(box.cls[0])].lower()
        if label == "keyboard":
            current_keyboard_box = tuple(map(int, box.xyxy[0]))
            break

    # Auto recalibration if keyboard moved slightly
    if has_drifted(prev_keyboard_box, current_keyboard_box):
        print("⚙️ Keyboard moved — recalibrating...")
        key_positions = calibrate_keyboard()
        prev_keyboard_box = current_keyboard_box
    elif current_keyboard_box is not None:
        prev_keyboard_box = current_keyboard_box

    # Draw bounding boxes for keys
    for key, (x1, y1, x2, y2) in key_positions.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 1)
        cv2.putText(frame, key.upper(), (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 50), 1)

    # Process hands
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

    # Detect and log keypresses
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
                norm_dx, norm_dy = dx / FRAME_WIDTH, dy / FRAME_HEIGHT
                norm_dist = min_dist / np.sqrt(FRAME_WIDTH**2 + FRAME_HEIGHT**2)
                expected_hand, expected_finger = TOUCH_TYPING_MAP.get(key, ("unknown", "unknown"))
                rule_based_correct = (expected_finger == closest_finger and (expected_hand == hand_used or expected_hand == "both"))
                json_correct = (expected_key == key) if expected_key else True
                overall_correct = rule_based_correct and json_correct
                rule_label = "Correct" if overall_correct else "Incorrect"
                color = (0, 255, 0) if overall_correct else (0, 0, 255)

                # Display feedback
                cv2.putText(frame, f"{key.upper()} | Exp: {expected_key or '-'} | {rule_label}",
                            (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                # Save dataset row
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        time.time(), key, closest_finger, hand_used,
                        fx, fy, x1, y1, x2, y2, dx, dy, min_dist,
                        norm_dx, norm_dy, norm_dist, rule_label
                    ])

                # Console output
                result = {
                    "expected_key": expected_key.upper() if expected_key else None,
                    "key": key.upper(),
                    "finger": closest_finger,
                    "hand": hand_used,
                    "correct": overall_correct
                }
                print(json.dumps(result))
                sys.stdout.flush()

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
