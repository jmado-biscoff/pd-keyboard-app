# Headless keyboard detection: YOLO + Mediapipe + SVM
# Streams detection results and visualization frames via stdout JSON

import warnings
import os, sys

# Suppress sklearn version warnings (must run before ML library imports)
warnings.filterwarnings("ignore", message=r".*InconsistentVersion.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.*")

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import csv, time, json, base64
from datetime import datetime
import keyboard
import threading
import joblib
import pandas as pd
import sklearn
from collections import deque

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)

YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, "runs", "train", "keyboard_key_detector", "weights", "best.pt")
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 1
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
DEBOUNCE_INTERVAL = 0.25

# Separate visual FPS from inference for smooth rendering
VISUAL_FPS = 60
INFERENCE_FPS = 20
FRAME_INTERVAL = 1.0 / VISUAL_FPS

SAVE_DIR = os.path.join(ML_DIR, "testing")
RESULTS_DIR = os.path.join(ML_DIR, "results")

# Ground truth validation - overrides ML predictions if hand/finger incorrect
# Prevents false positives from wrong-handed or wrong-fingered key presses
KEY_TO_EXPECTED_HAND = {
    'q': 'left', 'w': 'left', 'e': 'left', 'r': 'left', 't': 'left',
    'a': 'left', 's': 'left', 'd': 'left', 'f': 'left', 'g': 'left',
    'z': 'left', 'x': 'left', 'c': 'left', 'v': 'left', 'b': 'left',
    'y': 'right', 'u': 'right', 'i': 'right', 'o': 'right', 'p': 'right',
    'h': 'right', 'j': 'right', 'k': 'right', 'l': 'right',
    'n': 'right', 'm': 'right',
}

KEY_TO_EXPECTED_FINGER = {
    'q': 'pinky', 'a': 'pinky', 'z': 'pinky',
    'w': 'ring', 's': 'ring', 'x': 'ring',
    'e': 'middle', 'd': 'middle', 'c': 'middle',
    'r': 'index', 't': 'index', 'f': 'index', 'g': 'index', 'v': 'index', 'b': 'index',
    'y': 'index', 'u': 'index', 'h': 'index', 'j': 'index', 'n': 'index', 'm': 'index',
    'i': 'middle', 'k': 'middle',
    'o': 'ring', 'l': 'ring',
    'p': 'pinky',
}

os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"finger_key_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
EXPECTED_PATH = os.path.join(ML_DIR, "results_csv", "expected_words.json")

def is_allowed_key(label: str) -> bool:
    return len(label) == 1 and label.isalpha()

expected_keys: list[str] = []
start_index = 0

if os.path.exists(EXPECTED_PATH):
    try:
        with open(EXPECTED_PATH, "r") as f:
            data = json.load(f)
            raw_items = data.get("words", [])
            start_index = data.get("startIndex", 0)
            for item in raw_items:
                if not item:
                    continue
                if isinstance(item, str):
                    expected_keys.extend(list(item.replace(" ", "").lower()))
                elif isinstance(item, list):
                    expected_keys.extend([str(ch).lower() for ch in item])
        print(f"Parsed {len(expected_keys)} expected characters starting at index {start_index}: {expected_keys[start_index:start_index+10] if start_index < len(expected_keys) else []}")
    except Exception as e:
        print(f"Failed to parse expected_words.json: {e}")
else:
    print(f"expected_words.json NOT FOUND at: {EXPECTED_PATH}")

current_expected_index = start_index

# Load models with sklearn version compatibility
def load_model(path: str, name: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)
    version_mismatch = any(
        issubclass(w.category, sklearn.exceptions.InconsistentVersionWarning)
        for w in caught
    )
    if version_mismatch:
        joblib.dump(model, path)
        print(f"Re-exported {name} with sklearn {sklearn.__version__}")
    return model

try:
    svm_model = load_model(os.path.join(RESULTS_DIR, "svm_model.pkl"), "SVM")
    encoder = load_model(os.path.join(RESULTS_DIR, "encoder.pkl"), "Encoder")
    scaler = load_model(os.path.join(RESULTS_DIR, "scaler.pkl"), "Scaler")
    print("SVM, Encoder, and Scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load models: {e}")
    sys.exit(1)

# Initialize YOLO + Mediapipe
print("Loading YOLOv8 keyboard model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLOv8 model loaded successfully.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils

# Keyboard input handler
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

# Exit listener for graceful shutdown
exit_flag = False

def exit_listener():
    global exit_flag
    for line in sys.stdin:
        if "EXIT" in line.strip().upper():
            print(json.dumps({"type": "status", "message": "EXIT received"}))
            sys.stdout.flush()
            exit_flag = True
            break

threading.Thread(target=exit_listener, daemon=True).start()

# Frame encoding for 60 FPS streaming
def encode_frame(frame, width=640, height=360, quality=75):
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return base64.b64encode(buf).decode('ascii')

# Camera initialization
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, VISUAL_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print(json.dumps({"type": "error", "status": "error", "reason": "NO_CAMERA", "message": "Cannot open camera."}))
    sys.stdout.flush()
    sys.exit(1)

# Use actual camera dimensions for MediaPipe landmark projection
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera active at {FRAME_WIDTH}x{FRAME_HEIGHT}")
sys.stdout.flush()

# FPS tracking
frame_times = deque(maxlen=60)
inference_times = deque(maxlen=30)
last_fps_report = time.time()

# Auto-calibration - 26 letter keys (a-z), no spacebar
YOLO_LABEL_FIX = {"y": "z", "z": "y"}
REQUIRED_KEYS = set(list('abcdefghijklmnopqrstuvwxyz'))

calibration_done = False
calibration_boxes = {}
calibration_confs = {}
key_positions = {}
last_frame_send_time = 0.0

# Motion detection and stuck timer for calibration reset
previous_center = None
last_count_change_time = time.time()
last_detected_count = 0
MOTION_THRESHOLD = 20
STUCK_TIMEOUT = 4.0

print("Starting automatic keyboard calibration...")
sys.stdout.flush()

while not calibration_done:
    if exit_flag:
        print(json.dumps({"type": "status", "message": "EXIT during calibration"}))
        sys.stdout.flush()
        cap.release()
        hands.close()
        sys.exit(0)

    ret, frame = cap.read()
    if not ret:
        continue

    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        raw_label = results[0].names[cls_id].lower()
        label = YOLO_LABEL_FIX.get(raw_label, raw_label)
        conf = float(box.conf[0])
        if not is_allowed_key(label):
            continue
        if label not in calibration_confs or conf > calibration_confs[label]:
            calibration_confs[label] = conf
            calibration_boxes[label] = tuple(map(int, box.xyxy[0]))

    # Motion detection - reset if keyboard moves
    if calibration_boxes:
        centers = []
        for x1, y1, x2, y2 in calibration_boxes.values():
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        avg_x = sum(cx for cx, cy in centers) // len(centers)
        avg_y = sum(cy for cx, cy in centers) // len(centers)
        current_center = (avg_x, avg_y)
        if previous_center is not None:
            dx = abs(current_center[0] - previous_center[0])
            dy = abs(current_center[1] - previous_center[1])
            motion_distance = (dx**2 + dy**2) ** 0.5
            if motion_distance > MOTION_THRESHOLD:
                print(json.dumps({"type": "status", "message": f"Keyboard moved ({motion_distance:.0f}px) - resetting"}), file=sys.stderr)
                calibration_boxes.clear()
                calibration_confs.clear()
                previous_center = None
                last_detected_count = 0
                last_count_change_time = time.time()
        previous_center = current_center

    # Stuck timer - reset if no progress for 4 seconds
    current_count = len(calibration_boxes)
    if current_count > last_detected_count:
        last_detected_count = current_count
        last_count_change_time = time.time()
    elif current_count > 0 and current_count < 27:
        time_since_change = time.time() - last_count_change_time
        if time_since_change > STUCK_TIMEOUT:
            print(json.dumps({"type": "status", "message": f"Calibration stuck at {current_count}/27 - resetting"}), file=sys.stderr)
            calibration_boxes.clear()
            calibration_confs.clear()
            previous_center = None
            last_detected_count = 0
            last_count_change_time = time.time()

    # Draw bounding boxes on visualization frame
    viz_frame = frame.copy()
    for label, (x1, y1, x2, y2) in calibration_boxes.items():
        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Stream progress at 60 FPS
    now = time.time()
    if now - last_frame_send_time >= FRAME_INTERVAL:
        last_frame_send_time = now
        detected_list = sorted(calibration_boxes.keys())
        frame_b64 = encode_frame(viz_frame, quality=75)
        print(json.dumps({
            "type": "calibration_progress",
            "detected": len(calibration_boxes),
            "required": len(REQUIRED_KEYS),
            "detected_keys": detected_list,
            "frame": frame_b64
        }))
        sys.stdout.flush()

    # Complete calibration when all keys detected
    if REQUIRED_KEYS.issubset(calibration_boxes.keys()):
        calibration_done = True
        key_positions = calibration_boxes.copy()
        print(json.dumps({"type": "calibration_done", "status": "ok", "locked_keys": len(key_positions)}))
        sys.stdout.flush()

# CSV header
csv_header = ["timestamp", "pressed_key", "finger_name", "hand_used", "dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance", "expected_key", "ml_label"]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_header)

# Threaded MediaPipe worker for non-blocking inference
import queue as _queue_mod

_mp_input: _queue_mod.Queue = _queue_mod.Queue(maxsize=1)
_latest_landmarks = None
_latest_handedness = None
_landmarks_lock = threading.Lock()

def _mp_worker():
    global _latest_landmarks, _latest_handedness
    while not exit_flag:
        try:
            rgb_frame = _mp_input.get(timeout=0.1)
        except _queue_mod.Empty:
            continue
        result = hands.process(rgb_frame)
        with _landmarks_lock:
            _latest_landmarks = result.multi_hand_landmarks
            _latest_handedness = result.multi_handedness
        _mp_input.task_done()

threading.Thread(target=_mp_worker, daemon=True).start()

# Main detection loop - 60 FPS visual + adaptive inference
last_frame_send_time = time.time()
last_inference_time = 0.0
inference_interval = 1.0 / INFERENCE_FPS

while True:
    loop_start = time.perf_counter()

    if exit_flag:
        break

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Adaptive inference - only run MediaPipe at INFERENCE_FPS to prevent blocking
    now_perf = time.perf_counter()
    if now_perf - last_inference_time >= inference_interval:
        last_inference_time = now_perf
        try:
            _mp_input.put_nowait(rgb)
            inference_times.append(now_perf)
        except _queue_mod.Full:
            pass

    # Get latest landmarks from worker thread
    with _landmarks_lock:
        _current_landmarks = _latest_landmarks
        _current_handedness = _latest_handedness

    detected_fingers = []
    fingertip_count = 0
    left_fingers_count = 0
    right_fingers_count = 0
    frame_h, frame_w = frame.shape[:2]

    # Hand detection with mismatched handedness support
    if _current_landmarks:
        num_hands = len(_current_landmarks)
        num_handedness = len(_current_handedness) if _current_handedness else 0
        for hand_idx, hand_landmarks in enumerate(_current_landmarks):
            # Use handedness if available, otherwise fall back to wrist position
            if hand_idx < num_handedness:
                handedness_label = _current_handedness[hand_idx].classification[0].label
                # Swap left/right to correct for MediaPipe's mirrored camera perspective
                if handedness_label == "Left":
                    hand_used = "right"
                elif handedness_label == "Right":
                    hand_used = "left"
                else:
                    hand_used = handedness_label.lower()
            else:
                wrist_x = hand_landmarks.landmark[0].x
                hand_used = "left" if wrist_x < 0.5 else "right"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw L/R indicator over wrist
            wrist_x_px = int(hand_landmarks.landmark[0].x * frame_w)
            wrist_y_px = int(hand_landmarks.landmark[0].y * frame_h)
            indicator_label = hand_used[0].upper()
            label_size = cv2.getTextSize(indicator_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            box_x1 = wrist_x_px - label_size[0] // 2 - 5
            box_y1 = wrist_y_px - 35
            box_x2 = wrist_x_px + label_size[0] // 2 + 5
            box_y2 = wrist_y_px - 5
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
            cv2.putText(frame, indicator_label, (wrist_x_px - label_size[0] // 2, wrist_y_px - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Extract fingertip positions with separate hand counting
            hand_finger_count = 0
            for name, idx in {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}.items():
                fx = int(hand_landmarks.landmark[idx].x * frame_w)
                fy = int(hand_landmarks.landmark[idx].y * frame_h)
                detected_fingers.append((name, fx, fy, hand_used))
                fingertip_count += 1
                hand_finger_count += 1
                cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1)

            # Accumulate counts per hand
            if hand_used == "left":
                left_fingers_count += hand_finger_count
            elif hand_used == "right":
                right_fingers_count += hand_finger_count

    # Draw locked key bounding boxes
    for label, (x1, y1, x2, y2) in key_positions.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 3)

    expected_key = expected_keys[current_expected_index] if current_expected_index < len(expected_keys) else None

    now = time.time()
    if pressed_key and (
        pressed_key != last_logged_key or (now - last_sent_time) > DEBOUNCE_INTERVAL
    ):
        key = pressed_key.lower()
        last_logged_key = key
        last_sent_time = now

        if key != "space" and key in key_positions and detected_fingers:
            x1, y1, x2, y2 = key_positions[key]
            key_cx, key_cy = (x1 + x2) // 2, (y1 + y2) // 2

            fx, fy, closest_finger, used_hand = None, None, None, None
            min_dist = float("inf")
            for fname, cx, cy, hlabel in detected_fingers:
                dist = np.sqrt((cx - key_cx) ** 2 + (cy - key_cy) ** 2)
                if dist < min_dist:
                    min_dist, fx, fy, closest_finger, used_hand = dist, cx, cy, fname, hlabel

            if closest_finger:
                # Lockdown mode: block detection if hands are out of position
                # This prevents incorrect metrics when user takes hands off keyboard
                if fingertip_count < 9:
                    # Hands not in position - stop processing and don't advance
                    continue

                dx, dy = fx - key_cx, fy - key_cy
                distance = np.sqrt(dx**2 + dy**2)
                norm_dx, norm_dy = dx / frame_w, dy / frame_h
                norm_distance = distance / np.sqrt(frame_w**2 + frame_h**2)

                # ML inference using encoder + scaler + SVM
                cat_df = pd.DataFrame([[key.upper(), closest_finger.title(), used_hand.title()]], columns=["pressed_key", "finger_name", "hand_used"])
                num_df = pd.DataFrame([[dx, dy, min_dist, norm_dx, norm_dy, norm_distance]], columns=["dx", "dy", "distance", "norm_dx", "norm_dy", "norm_distance"])
                encoded = pd.DataFrame(encoder.transform(cat_df), columns=encoder.get_feature_names_out(["pressed_key", "finger_name", "hand_used"]))
                scaled = pd.DataFrame(scaler.transform(num_df), columns=[f"{col}_scaled" for col in num_df.columns])
                features = pd.concat([encoded, scaled], axis=1)
                ml_pred = svm_model.predict(features)[0]
                ml_label_raw = "Correct" if ml_pred == 1 else "Incorrect"

                # Validation layer - override ML if hand/finger violates touch typing rules
                expected_hand = KEY_TO_EXPECTED_HAND.get(key, None)
                expected_finger = KEY_TO_EXPECTED_FINGER.get(key, None)
                hand_correct = (expected_hand is None) or (used_hand == expected_hand)
                finger_correct = (expected_finger is None) or (closest_finger == expected_finger)

                if ml_label_raw == "Correct" and hand_correct and finger_correct:
                    ml_label = "Correct"
                else:
                    ml_label = "Incorrect"
                    if not hand_correct:
                        print(json.dumps({"type": "debug", "message": f"Hand violation: expected {expected_hand}, got {used_hand} for key '{key}'"}), file=sys.stderr)
                    if not finger_correct:
                        print(json.dumps({"type": "debug", "message": f"Finger violation: expected {expected_finger}, got {closest_finger} for key '{key}'"}), file=sys.stderr)

                # Save to CSV
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([time.time(), key, closest_finger, used_hand, dx, dy, distance, norm_dx, norm_dy, norm_distance, expected_key, ml_label])

                # Stream detection event
                print(json.dumps({"type": "detection", "key": key.upper(), "finger": closest_finger, "hand": used_hand, "expected_key": expected_key.upper() if expected_key else None, "ml_label": ml_label, "fingertip_count": fingertip_count, "left_fingers_count": left_fingers_count, "right_fingers_count": right_fingers_count}))
                sys.stdout.flush()
                current_expected_index += 1

    elif not pressed_key:
        last_logged_key = None

    # Stream annotated frame at 60 FPS
    now_frame = time.time()
    if now_frame - last_frame_send_time >= FRAME_INTERVAL:
        last_frame_send_time = now_frame
        frame_b64 = encode_frame(frame, quality=75)
        print(json.dumps({"type": "frame", "frame": frame_b64, "fingertip_count": fingertip_count, "left_fingers_count": left_fingers_count, "right_fingers_count": right_fingers_count}))
        sys.stdout.flush()
        frame_times.append(time.perf_counter())

    # FPS monitoring - report every 5 seconds
    if time.time() - last_fps_report >= 5.0:
        last_fps_report = time.time()
        if len(frame_times) > 1:
            visual_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
        else:
            visual_fps = 0
        if len(inference_times) > 1:
            inf_fps = len(inference_times) / (inference_times[-1] - inference_times[0])
        else:
            inf_fps = 0
        print(json.dumps({"type": "fps_stats", "visual_fps": round(visual_fps, 1), "inference_fps": round(inf_fps, 1), "target_visual": VISUAL_FPS, "target_inference": INFERENCE_FPS}), file=sys.stderr)

# Cleanup
cap.release()
hands.close()
print(json.dumps({"type": "status", "message": f"Session complete. Data saved to: {csv_path}"}))
sys.stdout.flush()