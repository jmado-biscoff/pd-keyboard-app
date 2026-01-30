# ============================================================
# detect_keyboard_live.py — HEADLESS INTEGRATION
# YOLO + Mediapipe + SVM + Encoder + Scaler
# Runs headless (no GUI window) and streams detection results
# plus visualization frames to the backend via stdout JSON.
# ============================================================

import warnings
import os, sys

# ============================================================
# WARNING HYGIENE — must run before any ML library imports
# ============================================================
# Suppress sklearn's InconsistentVersionWarning (patch-level
# version differences within the same minor release are safe)
warnings.filterwarnings(
    "ignore",
    message=r".*InconsistentVersion.*",
)
# Suppress generic sklearn FutureWarnings from unpickling
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

# ============================================================
# CONFIGURATION — OPTIMIZED FOR 60 FPS VISUAL RENDERING
# ============================================================
YOLO_MODEL_PATH = r"E:\pd-keyboard-app\backend\ml\notebooks\runs\train\keyboard_key_detector\weights\best.pt"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 1
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
DEBOUNCE_INTERVAL = 0.25

# CRITICAL: Separate visual FPS from inference rate for smooth rendering
VISUAL_FPS = 60            # Target frame streaming rate for smooth browser rendering
INFERENCE_FPS = 20         # MediaPipe processing rate (lower = better performance)
FRAME_INTERVAL = 1.0 / VISUAL_FPS  # ~16.6ms — target frame send period

SAVE_DIR = r"E:\pd-keyboard-app\backend\ml\testing"
RESULTS_DIR = r"E:\pd-keyboard-app\backend\ml\results"

# ============================================================
# DETERMINISTIC VALIDATION RULES — GROUND TRUTH MAPPINGS
# ============================================================
# These mappings enforce hard constraints that override ML predictions.
# If the detected hand or finger doesn't match the expected values,
# the keystroke is ALWAYS marked as "Incorrect" regardless of SVM output.
#
# This prevents false positives where the SVM incorrectly classifies
# wrong-handed or wrong-fingered key presses as "Correct".
# ============================================================

# Which hand should press each key (QWERTY touch typing standard)
KEY_TO_EXPECTED_HAND = {
    # Left hand keys
    'q': 'left', 'w': 'left', 'e': 'left', 'r': 'left', 't': 'left',
    'a': 'left', 's': 'left', 'd': 'left', 'f': 'left', 'g': 'left',
    'z': 'left', 'x': 'left', 'c': 'left', 'v': 'left', 'b': 'left',

    # Right hand keys
    'y': 'right', 'u': 'right', 'i': 'right', 'o': 'right', 'p': 'right',
    'h': 'right', 'j': 'right', 'k': 'right', 'l': 'right',
    'n': 'right', 'm': 'right',
}

# Which finger should press each key (QWERTY touch typing standard)
KEY_TO_EXPECTED_FINGER = {
    # Left hand
    'q': 'pinky', 'a': 'pinky', 'z': 'pinky',
    'w': 'ring', 's': 'ring', 'x': 'ring',
    'e': 'middle', 'd': 'middle', 'c': 'middle',
    'r': 'index', 't': 'index', 'f': 'index', 'g': 'index', 'v': 'index', 'b': 'index',

    # Right hand
    'y': 'index', 'u': 'index', 'h': 'index', 'j': 'index', 'n': 'index', 'm': 'index',
    'i': 'middle', 'k': 'middle',
    'o': 'ring', 'l': 'ring',
    'p': 'pinky',
}

os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(
    SAVE_DIR,
    f"finger_key_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
)

# ✅ Path B – where your backend writes expected_words.json
EXPECTED_PATH = r"E:\pd-keyboard-app\backend\ml\results_csv\expected_words.json"

# ============================================================
# UNIVERSAL EXPECTED WORDS PARSER (WORDS or LETTERS or MIXED)
# ============================================================

def is_allowed_key(label: str) -> bool:
    # ✅ 26-KEY MODEL: Only allow single letter keys (a-z)
    return len(label) == 1 and label.isalpha()

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
# LOAD MODELS (with sklearn version consistency)
# ============================================================
def load_model(path: str, name: str):
    """Load a joblib model. If the pickle was created by a different
    sklearn patch version, re-export it so subsequent loads are silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)

    version_mismatch = any(
        issubclass(w.category, sklearn.exceptions.InconsistentVersionWarning)
        for w in caught
    )
    if version_mismatch:
        joblib.dump(model, path)
        print(f"✅ Re-exported {name} with sklearn {sklearn.__version__} "
              f"(was trained on a different patch version)")
    return model

try:
    svm_model = load_model(os.path.join(RESULTS_DIR, "svm_model.pkl"), "SVM")
    encoder = load_model(os.path.join(RESULTS_DIR, "encoder.pkl"), "Encoder")
    scaler = load_model(os.path.join(RESULTS_DIR, "scaler.pkl"), "Scaler")
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
    max_num_hands=2,
    min_detection_confidence=0.4,  # ✅ Lowered from 0.6 for aggressive detection
    min_tracking_confidence=0.4     # ✅ Lowered from 0.6 to keep tracking in poor lighting
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

# ============================================================
# EXIT LISTENER (from Node.js) — started early so EXIT works
# during calibration as well as the main detection loop
# ============================================================

exit_flag = False

def exit_listener():
    global exit_flag
    for line in sys.stdin:
        if "EXIT" in line.strip().upper():
            print(json.dumps({"type": "status", "message": "EXIT received — shutting down"}))
            sys.stdout.flush()
            exit_flag = True
            break

threading.Thread(target=exit_listener, daemon=True).start()

# ============================================================
# FRAME ENCODING HELPER — OPTIMIZED FOR 60 FPS
# ============================================================

def encode_frame(frame, width=640, height=360, quality=75):
    """
    Resize and encode a BGR frame as a base64 JPEG string.
    Optimized for 60 FPS streaming with minimal CPU overhead.

    Args:
        frame: BGR frame from OpenCV
        width, height: Target dimensions (smaller = faster, less bandwidth)
        quality: JPEG quality 1-100 (75 = good balance of quality/speed)

    Returns:
        Base64-encoded JPEG string (NO LOGGING - critical for 60 FPS)
    """
    # Use INTER_LINEAR for speed (INTER_AREA is slower but higher quality)
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # JPEG encoding with optimization flag for better compression
    _, buf = cv2.imencode(
        '.jpg',
        small,
        [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    )

    # Fast base64 encoding with ASCII output (faster than UTF-8)
    return base64.b64encode(buf).decode('ascii')

# ============================================================
# CAMERA INITIALIZATION — OPTIMIZED FOR 60 FPS
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, VISUAL_FPS)  # Request 60 FPS from camera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Minimize buffer lag
if not cap.isOpened():
    print(json.dumps({
        "type": "error",
        "status": "error",
        "reason": "NO_CAMERA",
        "message": "Cannot open camera."
    }))
    sys.stdout.flush()
    sys.exit(1)

# Read actual dimensions from the camera — the driver may not honour
# the requested resolution, so use what is actually delivered.
# This is critical for correct MediaPipe landmark projection on
# non-square ROIs: normalized (0–1) coordinates must be scaled by
# the real frame size, not the requested size.
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Camera active at {FRAME_WIDTH}x{FRAME_HEIGHT}")
sys.stdout.flush()

# ============================================================
# FPS TRACKING — For performance monitoring
# ============================================================
frame_times = deque(maxlen=60)  # Track last 60 frames for visual FPS
inference_times = deque(maxlen=30)  # Track last 30 inferences
last_fps_report = time.time()

# ════════════════════════════════════════════════════════════
# AUTO-CALIBRATION (headless) — 26-Key Model
# ════════════════════════════════════════════════════════════
# Accumulates key detections across frames until all 26 letter
# keys (a-z) have been seen at least once.
# Spacebar removed for improved calibration reliability.
# No user interaction or GUI window required.
# ════════════════════════════════════════════════════════════

YOLO_LABEL_FIX = {"y": "z", "z": "y"}
# ✅ 26-KEY MODEL: Spacebar removed from calibration
REQUIRED_KEYS = set(list('abcdefghijklmnopqrstuvwxyz'))  # a-z only (26 keys)

calibration_done = False
calibration_boxes = {}   # key_label -> (x1, y1, x2, y2)
calibration_confs = {}   # key_label -> best confidence seen
key_positions = {}
last_frame_send_time = 0.0

# ============================================================
# MOTION-SENSING RESET & STUCK TIMER
# ============================================================
previous_center = None  # (avg_x, avg_y) of all boxes in previous frame
last_count_change_time = time.time()  # Last time the box count increased
last_detected_count = 0  # Track progress to detect "stuck" state
MOTION_THRESHOLD = 20  # pixels - if keyboard moves more than this, reset
STUCK_TIMEOUT = 4.0    # seconds - if no progress for 4s, reset

print("🔄 Starting automatic keyboard calibration...")
sys.stdout.flush()

while not calibration_done:
    if exit_flag:
        print(json.dumps({"type": "status", "message": "EXIT during calibration"}))
        sys.stdout.flush()
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

        # Keep the highest-confidence bounding box per key across all frames
        if label not in calibration_confs or conf > calibration_confs[label]:
            calibration_confs[label] = conf
            calibration_boxes[label] = tuple(map(int, box.xyxy[0]))

    # ============================================================
    # MOTION DETECTION: Reset if keyboard moves
    # ============================================================
    if calibration_boxes:
        # Calculate average center of all detected boxes
        centers = []
        for x1, y1, x2, y2 in calibration_boxes.values():
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centers.append((center_x, center_y))

        avg_x = sum(cx for cx, cy in centers) // len(centers)
        avg_y = sum(cy for cx, cy in centers) // len(centers)
        current_center = (avg_x, avg_y)

        # Check if keyboard moved significantly
        if previous_center is not None:
            dx = abs(current_center[0] - previous_center[0])
            dy = abs(current_center[1] - previous_center[1])
            motion_distance = (dx**2 + dy**2) ** 0.5

            if motion_distance > MOTION_THRESHOLD:
                print(json.dumps({
                    "type": "status",
                    "message": f"🔄 Keyboard moved ({motion_distance:.0f}px) - resetting calibration"
                }), file=sys.stderr)
                calibration_boxes.clear()
                calibration_confs.clear()
                previous_center = None
                last_detected_count = 0
                last_count_change_time = time.time()

        previous_center = current_center

    # ============================================================
    # STUCK TIMER: Reset if no progress for 4 seconds
    # ============================================================
    current_count = len(calibration_boxes)
    if current_count > last_detected_count:
        # Progress made - update timer
        last_detected_count = current_count
        last_count_change_time = time.time()
    elif current_count > 0 and current_count < 27:
        # No progress and not complete - check if stuck
        time_since_change = time.time() - last_count_change_time
        if time_since_change > STUCK_TIMEOUT:
            print(json.dumps({
                "type": "status",
                "message": f"⚠️ Calibration stuck at {current_count}/27 for {STUCK_TIMEOUT}s - resetting"
            }), file=sys.stderr)
            calibration_boxes.clear()
            calibration_confs.clear()
            previous_center = None
            last_detected_count = 0
            last_count_change_time = time.time()

    # Draw current accumulated detections on a visualization frame
    # Bounding boxes only - no text labels (per user request)
    viz_frame = frame.copy()
    for label, (x1, y1, x2, y2) in calibration_boxes.items():
        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thickness: 3px for better visibility
        # Text overlay removed - bounding boxes provide sufficient visual feedback

    # Periodically stream progress + visualization frame to backend at 60 FPS
    now = time.time()
    if now - last_frame_send_time >= FRAME_INTERVAL:
        last_frame_send_time = now
        detected_list = sorted(calibration_boxes.keys())

        # Optimized encoding: quality=75 for calibration (good balance)
        frame_b64 = encode_frame(viz_frame, quality=75)

        print(json.dumps({
            "type": "calibration_progress",
            "detected": len(calibration_boxes),
            "required": len(REQUIRED_KEYS),
            "detected_keys": detected_list,
            "frame": frame_b64
        }))
        sys.stdout.flush()

    # Lock calibration once all required keys have been detected
    if REQUIRED_KEYS.issubset(calibration_boxes.keys()):
        calibration_done = True
        key_positions = calibration_boxes.copy()
        print(json.dumps({
            "type": "calibration_done",
            "status": "ok",
            "locked_keys": len(key_positions)
        }))
        sys.stdout.flush()

# ============================================================
# CSV HEADER
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
# THREADED MEDIAPIPE WORKER
# Runs hands.process() in a background thread so the main loop
# is never blocked waiting for landmark inference.  The input
# queue has maxsize=1: if the worker hasn't consumed the previous
# frame yet, put_nowait() drops the new one, keeping latency low.
# The main loop always reads the most recent landmarks without
# blocking, which may be from a prior frame — acceptable because
# hand position changes slowly relative to 30 FPS capture.
# ============================================================
import queue as _queue_mod

_mp_input: _queue_mod.Queue = _queue_mod.Queue(maxsize=1)
_latest_landmarks = None
_latest_handedness = None  # Store handedness classifications
_landmarks_lock = threading.Lock()

def _mp_worker():
    """Consume frames from _mp_input, run MediaPipe, publish landmarks AND handedness."""
    global _latest_landmarks, _latest_handedness
    while not exit_flag:
        try:
            rgb_frame = _mp_input.get(timeout=0.1)
        except _queue_mod.Empty:
            continue
        result = hands.process(rgb_frame)
        with _landmarks_lock:
            # Store both landmarks and handedness from the MediaPipe result
            # multi_handedness contains classification labels ("Left"/"Right")
            # for each detected hand, aligned by index with multi_hand_landmarks
            _latest_landmarks = result.multi_hand_landmarks
            _latest_handedness = result.multi_handedness
        _mp_input.task_done()

threading.Thread(target=_mp_worker, daemon=True).start()

# ============================================================
# MAIN DETECTION LOOP — 60 FPS VISUAL + ADAPTIVE INFERENCE
# Decouples frame capture/streaming from inference for smooth rendering.
# ============================================================
last_frame_send_time = time.time()  # Reset timer for main loop
last_inference_time = 0.0
inference_interval = 1.0 / INFERENCE_FPS  # e.g., 50ms for 20 FPS

while True:
    loop_start = time.perf_counter()

    if exit_flag:
        break

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ============================================================
    # ADAPTIVE INFERENCE — Only run MediaPipe at INFERENCE_FPS
    # This prevents MediaPipe from blocking 60 FPS visual rendering
    # ============================================================
    now_perf = time.perf_counter()
    if now_perf - last_inference_time >= inference_interval:
        last_inference_time = now_perf

        # Feed the MediaPipe worker (non-blocking; silently drops if
        # the worker hasn't finished the previous frame yet).
        try:
            _mp_input.put_nowait(rgb)
            inference_times.append(now_perf)  # Track inference FPS
        except _queue_mod.Full:
            pass  # Skip this frame if worker is busy

    # Snapshot the latest landmarks AND handedness published by the worker thread.
    # These may be from a prior frame — that's fine; hand position
    # changes slowly relative to the 30 FPS capture rate.
    with _landmarks_lock:
        _current_landmarks = _latest_landmarks
        _current_handedness = _latest_handedness

    detected_fingers = []
    fingertip_count = 0  # Track total fingertips detected (0-10)

    # Use actual frame dimensions for landmark projection so that
    # fingertip coordinates are correct regardless of aspect ratio.
    # MediaPipe normalized coords (0–1) are relative to the image
    # that was fed to process(), which is this frame.
    frame_h, frame_w = frame.shape[:2]

    # ════════════════════════════════════════════════════════════
    # ✅ ROBUST HAND DETECTION WITH MISMATCHED HANDEDNESS SUPPORT
    # ════════════════════════════════════════════════════════════
    # MediaPipe provides:
    # - multi_hand_landmarks: landmark positions for each hand
    # - multi_handedness: classification labels ("Left"/"Right") for each hand
    #
    # PROBLEM: Sometimes len(handedness) < len(landmarks) due to transient
    # detection issues. The old code would crash or skip hands entirely.
    #
    # FIX: Always iterate over ALL landmarks. Use handedness data when
    # available (up to min length), fall back to wrist x-coordinate for
    # any hands that don't have matching handedness metadata.
    # ════════════════════════════════════════════════════════════
    if _current_landmarks:
        num_hands = len(_current_landmarks)
        num_handedness = len(_current_handedness) if _current_handedness else 0

        for hand_idx, hand_landmarks in enumerate(_current_landmarks):
            # ✅ FIX: Only use handedness if available for this index
            if hand_idx < num_handedness:
                # Extract handedness label from MediaPipe's classification result
                handedness_label = _current_handedness[hand_idx].classification[0].label

                # ════════════════════════════════════════════════════════
                # HANDEDNESS CORRECTION FOR MIRRORED CAMERA FEED
                # ════════════════════════════════════════════════════════
                # MediaPipe returns handedness from the person's perspective
                # in the ORIGINAL (non-mirrored) image. However, typing trainer
                # camera feeds are typically mirrored (like looking in a mirror)
                # so the user's right hand appears on the left side of the screen.
                #
                # To match the user's ACTUAL hand to the detection:
                # - Swap "Left" ↔ "Right" to account for horizontal flip
                # ════════════════════════════════════════════════════════
                if handedness_label == "Left":
                    hand_used = "right"  # Person's left hand appears on right side (mirrored)
                elif handedness_label == "Right":
                    hand_used = "left"   # Person's right hand appears on left side (mirrored)
                else:
                    hand_used = handedness_label.lower()  # Fallback
            else:
                # ✅ FALLBACK: Handedness not available for this hand, use wrist position
                wrist_x = hand_landmarks.landmark[0].x
                hand_used = "left" if wrist_x < 0.5 else "right"

            # Draw hand skeleton on the visualization frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ✅ VISUAL DEBUGGING: Draw L/R indicator over wrist
            wrist_x_px = int(hand_landmarks.landmark[0].x * frame_w)
            wrist_y_px = int(hand_landmarks.landmark[0].y * frame_h)
            indicator_label = hand_used[0].upper()  # "L" or "R"

            # Draw small box behind label for visibility
            label_size = cv2.getTextSize(indicator_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            box_x1 = wrist_x_px - label_size[0] // 2 - 5
            box_y1 = wrist_y_px - 35
            box_x2 = wrist_x_px + label_size[0] // 2 + 5
            box_y2 = wrist_y_px - 5
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)  # Black background
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)   # Green border

            # Draw label text
            cv2.putText(
                frame,
                indicator_label,
                (wrist_x_px - label_size[0] // 2, wrist_y_px - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # Green text
                2,
                cv2.LINE_AA
            )

            # Extract fingertip positions for all five fingers
            # Landmark indices: thumb=4, index=8, middle=12, ring=16, pinky=20
            for name, idx in {
                "thumb": 4,
                "index": 8,
                "middle": 12,
                "ring": 16,
                "pinky": 20,
            }.items():
                # Convert normalized (0-1) coordinates to pixel coordinates
                fx = int(hand_landmarks.landmark[idx].x * frame_w)
                fy = int(hand_landmarks.landmark[idx].y * frame_h)

                # Store finger info: (name, x, y, hand_label)
                detected_fingers.append((name, fx, fy, hand_used))
                fingertip_count += 1  # Increment total fingertip count

                # Draw fingertip marker on visualization frame
                cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1)

    # Draw locked key bounding boxes on the frame
    # Bounding boxes only - no text labels (per user request)
    for label, (x1, y1, x2, y2) in key_positions.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 3)  # Thickness: 3px for better visibility
        # Text overlay removed - clean visualization without labels

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

        # Process only keys that have a locked bounding box (skip space / unknown)
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
                dx, dy = fx - key_cx, fy - key_cy
                distance = np.sqrt(dx**2 + dy**2)
                norm_dx, norm_dy = dx / frame_w, dy / frame_h
                norm_distance = distance / np.sqrt(
                    frame_w**2 + frame_h**2
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

                # ====================================================
                # ML PREDICTION (SVM classifier output)
                # ====================================================
                ml_pred = svm_model.predict(features)[0]
                ml_label_raw = "Correct" if ml_pred == 1 else "Incorrect"

                # ====================================================
                # DETERMINISTIC VALIDATION LAYER — HARD RULE ENFORCEMENT
                # ====================================================
                # This layer overrides the SVM prediction if the detected
                # hand or finger violates touch typing rules. This prevents
                # false positives where the ML model incorrectly classifies
                # wrong-handed or wrong-fingered key presses as "Correct".
                #
                # Validation checks (all must pass for "Correct"):
                # 1. Is the detected hand the expected hand for this key?
                # 2. Is the detected finger the expected finger for this key?
                #
                # If ANY check fails, force ml_label = "Incorrect"
                # ====================================================

                expected_hand = KEY_TO_EXPECTED_HAND.get(key, None)
                expected_finger = KEY_TO_EXPECTED_FINGER.get(key, None)

                # Check hand constraint
                hand_correct = (expected_hand is None) or (used_hand == expected_hand)

                # Check finger constraint
                finger_correct = (expected_finger is None) or (closest_finger == expected_finger)

                # Final label: only "Correct" if ML says correct AND rules pass
                if ml_label_raw == "Correct" and hand_correct and finger_correct:
                    ml_label = "Correct"
                else:
                    ml_label = "Incorrect"

                    # Log the specific violation for debugging
                    if not hand_correct:
                        print(json.dumps({
                            "type": "debug",
                            "message": f"Hand violation: expected {expected_hand}, got {used_hand} for key '{key}'"
                        }), file=sys.stderr)
                    if not finger_correct:
                        print(json.dumps({
                            "type": "debug",
                            "message": f"Finger violation: expected {expected_finger}, got {closest_finger} for key '{key}'"
                        }), file=sys.stderr)

                # ════════════════════════════════════════════════════════════
                # ML FEEDBACK — NO TEXT OVERLAY (per user request)
                # ════════════════════════════════════════════════════════════
                # ML correctness feedback is communicated to the frontend via
                # JSON output, not drawn on the camera feed. This keeps the
                # visualization clean and professional.
                # ════════════════════════════════════════════════════════════

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

                # Stream detection event to backend
                print(
                    json.dumps(
                        {
                            "type": "detection",
                            "key": key.upper(),
                            "finger": closest_finger,
                            "hand": used_hand,
                            "expected_key": expected_key.upper() if expected_key else None,
                            "ml_label": ml_label,
                            "fingertip_count": fingertip_count,  # 0-10 fingers detected
                        }
                    )
                )
                sys.stdout.flush()

                current_expected_index += 1

    elif not pressed_key:
        last_logged_key = None

    # ============================================================
    # STREAM ANNOTATED FRAME AT 60 FPS (VISUAL_FPS)
    # NO LOGGING OF RAW FRAME DATA — critical for performance
    # ============================================================
    now_frame = time.time()
    if now_frame - last_frame_send_time >= FRAME_INTERVAL:
        last_frame_send_time = now_frame

        # Optimized encoding: quality=75 provides good balance
        # (higher quality = more CPU, lower = artifacts)
        frame_b64 = encode_frame(frame, quality=75)

        # Output frame with fingertip count for 10-finger monitoring
        print(json.dumps({
            "type": "frame",
            "frame": frame_b64,
            "fingertip_count": fingertip_count  # 0-10 fingers detected
        }))
        sys.stdout.flush()

        # Track visual FPS
        frame_times.append(time.perf_counter())

    # ============================================================
    # FPS MONITORING — Report every 5 seconds for debugging
    # ============================================================
    if time.time() - last_fps_report >= 5.0:
        last_fps_report = time.time()

        # Calculate actual FPS from timestamps
        if len(frame_times) > 1:
            visual_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
        else:
            visual_fps = 0

        if len(inference_times) > 1:
            inf_fps = len(inference_times) / (inference_times[-1] - inference_times[0])
        else:
            inf_fps = 0

        # Report to stderr to avoid interfering with JSON stdout
        print(json.dumps({
            "type": "fps_stats",
            "visual_fps": round(visual_fps, 1),
            "inference_fps": round(inf_fps, 1),
            "target_visual": VISUAL_FPS,
            "target_inference": INFERENCE_FPS
        }), file=sys.stderr)

# ============================================================
# CLEANUP — release all hardware and ML resources
# ============================================================
cap.release()
hands.close()  # Release MediaPipe graph and associated memory
print(json.dumps({"type": "status", "message": f"Session complete. Data saved to: {csv_path}"}))
sys.stdout.flush()