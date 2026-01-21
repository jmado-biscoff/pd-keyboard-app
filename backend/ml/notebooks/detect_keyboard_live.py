# ============================================================
# detect_keyboard_live.py — FULL INTEGRATION
# YOLO + Mediapipe + SVM + Encoder + Scaler
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
import threading
import sys

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL_PATH = r"E:\pd-keyboard-app\backend\ml\notebooks\runs\train\keyboard_key_detector\weights\best.pt"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 1
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
DEBOUNCE_INTERVAL = 0.25
calibration_done = False
calibration_frame = None
latest_frame = None

SAVE_DIR = r"E:\pd-keyboard-app\backend\ml\testing"
RESULTS_DIR = r"E:\pd-keyboard-app\backend\ml\results"

os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(
    SAVE_DIR,
    f"finger_key_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
)

# ✅ Path B – where your backend writes expected_words.json
EXPECTED_PATH = r"E:\pd-keyboard-app\backend\ml\results_csv\expected_words.json"

WINDOW_NAME = "YOLO + Mediapipe + SVM Typing Feedback"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ============================================================
# UNIVERSAL EXPECTED WORDS PARSER (WORDS or LETTERS or MIXED)
# ============================================================

def is_allowed_key(label: str) -> bool:
    return (len(label) == 1 and label.isalpha()) or label == "space"

expected_keys: list[str] = []

if os.path.exists(EXPECTED_PATH):
    try:
        with open(EXPECTED_PATH, "r") as f:
            data = json.load(f)
            raw_items = data.get("words", [])

            for item in raw_items:
                if not item:
                    continue

                # If it's a string (word or single letter)
                if isinstance(item, str):
                    # Remove spaces inside (e.g. "hello world" -> "helloworld")
                    cleaned = item.replace(" ", "")
                    # Extend with each character lowercased
                    expected_keys.extend(list(cleaned.lower()))

                # If it's a list of characters (future-proof)
                elif isinstance(item, list):
                    expected_keys.extend([str(ch).lower() for ch in item])

        print(f"✅ Parsed {len(expected_keys)} expected characters from expected_words.json")
        print(f"   Sequence: {expected_keys}")

    except Exception as e:
        print(f"❌ Failed to parse expected_words.json: {e}")
else:
    print(f"⚠️ expected_words.json NOT FOUND at: {EXPECTED_PATH}")

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
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ============================================================
# KEYBOARD INPUT HANDLER
# ============================================================
pressed_key = None
last_logged_key = None
last_sent_time = 0.0


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

def mouse_callback(event, x, y, flags, param):
    global calibration_done, calibration_frame, latest_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if latest_frame is not None:
            calibration_done = True
            calibration_frame = latest_frame.copy()
            print("🖱️ Calibration locked by mouse click")


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
print("🖱️ Click anywhere to CALIBRATE keyboard layout")

key_positions = {}
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while not calibration_done:
    ret, frame = cap.read()
    if not ret:
        continue

    latest_frame = frame.copy() 

    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    temp_positions = {}
    temp_conf = {}
    YOLO_LABEL_FIX = {
        "y": "z",
        "z": "y",
    }

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        raw_label = results[0].names[cls_id].lower()
        label = YOLO_LABEL_FIX.get(raw_label, raw_label)
        conf = float(box.conf[0])

        if not is_allowed_key(label):
            continue

        # ✅ keep ONLY the highest-confidence box per key label
        if label not in temp_conf or conf > temp_conf[label]:
            temp_conf[label] = conf
            temp_positions[label] = tuple(map(int, box.xyxy[0]))

    # 🔍 Draw LIVE preview (after deduplication)
    for label, (x1, y1, x2, y2) in temp_positions.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(
            frame,
            label.upper(),
            (x1 + 3, y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )

    cv2.putText(
        frame,
        "CLICK to calibrate keyboard",
        (40, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

    # When clicked → freeze
    if calibration_done:
        key_positions = temp_positions.copy()
        break

locked_key_count = len(key_positions)

if locked_key_count == 0:
    print(json.dumps({
        "status": "error",
        "reason": "NO_KEYBOARD",
        "message": "No keyboard detected."
    }))
    sys.stdout.flush()
    sys.exit(1)

elif locked_key_count < 27:
    print(json.dumps({
        "status": "error",
        "reason": "PARTIAL_KEYBOARD",
        "message": "Some keys were not detected. Please adjust the orientation of your keyboard.",
        "detected_keys": locked_key_count
    }))
    sys.stdout.flush()
    sys.exit(1)

else:
    print(json.dumps({
        "status": "ok",
        "locked_keys": locked_key_count
    }))
    sys.stdout.flush()

# ============================================================
# CSV HEADER — now includes expected_key instead of rule_based_label
# ============================================================
csv_header = [
    "timestamp",
    "pressed_key",
    "finger_name",
    "hand_used",
    "dx",
    "dy",
    "distance",
    "norm_dx",
    "norm_dy",
    "norm_distance",
    "expected_key",
    "ml_label",
]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_header)

# ============================================================
# EXIT LISTENER (from Node.js) — MUST be before main loop
# ============================================================

exit_flag = False

def exit_listener():
    global exit_flag
    for line in sys.stdin:
        if "EXIT" in line.strip().upper():
            print("🛑 EXIT received — closing OpenCV")
            exit_flag = True
            break

# Start listener thread
threading.Thread(target=exit_listener, daemon=True).start()

# ============================================================
# MAIN DETECTION LOOP
# ============================================================
while True:
    if exit_flag:
        break

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
            for name, idx in {
                "thumb": 4,
                "index": 8,
                "middle": 12,
                "ring": 16,
                "pinky": 20,
            }.items():
                fx = int(hand_landmarks.landmark[idx].x * FRAME_WIDTH)
                fy = int(hand_landmarks.landmark[idx].y * FRAME_HEIGHT)
                detected_fingers.append((name, fx, fy, hand_used))
                cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1)

    # Expected key from sequence (can be None if we run out)
    expected_key = (
        expected_keys[current_expected_index]
        if current_expected_index < len(expected_keys)
        else None
    )

    now = time.time()
    if pressed_key and (
        pressed_key != last_logged_key or (now - last_sent_time) > DEBOUNCE_INTERVAL
    ):
        key = pressed_key.lower()
        last_logged_key = key
        last_sent_time = now

        # Skip space and any YOLO label that isn't a detected key bbox
        if key == "space" or key not in key_positions:
            # Advance expected index per input even if space/unknown?
            # If you do NOT want that, comment out the next lines.
            continue

        if detected_fingers:
            x1, y1, x2, y2 = key_positions[key]
            key_cx, key_cy = (x1 + x2) // 2, (y1 + y2) // 2

            fx, fy, closest_finger, used_hand = None, None, None, None
            min_dist = float("inf")
            for fname, cx, cy, hlabel in detected_fingers:
                dist = np.sqrt((cx - key_cx) ** 2 + (cy - key_cy) ** 2)
                if dist < min_dist:
                    min_dist, fx, fy, closest_finger, used_hand = dist, cx, cy, fname, hlabel

            if closest_finger:
                dx, dy = fx - key_cx, fy - key_cy
                distance = np.sqrt(dx**2 + dy**2)
                norm_dx, norm_dy = dx / FRAME_WIDTH, dy / FRAME_HEIGHT
                norm_distance = distance / np.sqrt(
                    FRAME_WIDTH**2 + FRAME_HEIGHT**2
                )

                # ====================================================
                # ML INFERENCE USING ENCODER + SCALER + SVM
                # ====================================================
                cat_df = pd.DataFrame(
                    [[key.upper(), closest_finger.title(), used_hand.title()]],
                    columns=["pressed_key", "finger_name", "hand_used"],
                )

                num_df = pd.DataFrame(
                    [[dx, dy, min_dist, norm_dx, norm_dy, norm_distance]],
                    columns=[
                        "dx",
                        "dy",
                        "distance",
                        "norm_dx",
                        "norm_dy",
                        "norm_distance",
                    ],
                )

                encoded = pd.DataFrame(
                    encoder.transform(cat_df),
                    columns=encoder.get_feature_names_out(
                        ["pressed_key", "finger_name", "hand_used"]
                    ),
                )
                scaled = pd.DataFrame(
                    scaler.transform(num_df),
                    columns=[f"{col}_scaled" for col in num_df.columns],
                )

                features = pd.concat([encoded, scaled], axis=1)

                ml_pred = svm_model.predict(features)[0]
                ml_label = "Correct" if ml_pred == 1 else "Incorrect"

                # Display feedback (ML only)
                color = (0, 255, 0) if ml_label == "Correct" else (0, 0, 255)
                display_text = f"{key.upper()} | ML: {ml_label}"
                if expected_key is not None:
                    display_text += f" | Expected: {expected_key.upper()}"
                cv2.putText(
                    frame,
                    display_text,
                    (40, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    color,
                    2,
                )

                # Save to CSV
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            time.time(),
                            key,
                            closest_finger,
                            used_hand,
                            dx,
                            dy,
                            distance,
                            norm_dx,
                            norm_dy,
                            norm_distance,
                            expected_key,
                            ml_label,
                        ]
                    )

                # Print JSON for Node/PlaySession
                print(
                    json.dumps(
                        {
                            "key": key.upper(),
                            "finger": closest_finger,
                            "hand": used_hand,
                            "expected_key": expected_key.upper() if expected_key else None,
                            "ml_label": ml_label,
                        }
                    )
                )
                sys.stdout.flush()

                # ✅ Advance expected key index per input (correct or incorrect)
                current_expected_index += 1

    elif not pressed_key:
        last_logged_key = None

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Session complete. Data saved to: {csv_path}")