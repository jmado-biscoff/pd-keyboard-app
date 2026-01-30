# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

### Required Software
- **Node.js** (v18+) and npm
- **MongoDB** (running locally on port 27017 or configured in `.env`)
- **Python 3.10+** with pip
- **Camera access** for ML detection features

### Environment Setup

Create a `.env` file in the project root with the following configuration:

```bash
# Frontend
VITE_API_URL=http://localhost:5000/api/auth

# Backend
PORT=5000
MONGO_URI=mongodb://localhost:27017/typing_app
JWT_SECRET=your_jwt_secret_here

# ML Models
MODEL_PATH=backend/ml/notebooks/runs/train/keyboard_key_detector/weights/best.onnx
FEATURE_CFG=backend/ml/dataset/processed/feature_config.json
```

### Initial Setup

```bash
# Install dependencies
npm install
cd frontend && npm install && cd ..
cd backend && npm install && cd ..

# Install Python ML dependencies (see ML Dependencies section below)

# Start MongoDB (if not running as service)
mongod

# Run development servers
npm run dev
```

## Commands

### Development (from repo root)

```bash
npm run dev              # Start frontend (port 5173) + backend (port 5000) concurrently
npm run dev:frontend     # Frontend only
npm run dev:backend      # Backend only (nodemon + ts-node)
npm run build            # Build frontend (Vite) then compile backend (tsc)
npm start                # Run compiled backend (node dist/server.js)
npm run lint             # ESLint on frontend
npm run preview          # Preview frontend production build
```

### ML Pipeline (Python, run from repo root)

```bash
# Preprocessing
python backend/ml/notebooks/preprocessing.py

# Training
python backend/ml/notebooks/train_TCN.py
python backend/ml/notebooks/train_BiLSTM_CRF.py
python backend/ml/notebooks/train_Transformer.py

# Evaluation
python backend/ml/evaluation/evaluation_TCN.py
python backend/ml/evaluation/evaluation_BiLSTM_CRF.py
python backend/ml/evaluation/evaluation_Transformer.py

# Maintainability analysis
python backend/ml/evaluation/evaluation_maintainability.py

# Compare all models (generates CSV + plot)
python backend/ml/evaluation/compare_models.py

# Live keyboard detection (spawned by backend on /api/detect/start)
python backend/ml/notebooks/detect_keyboard_live.py
```

### ML Dependencies (Python 3.10+)

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm matplotlib
pip install onnx onnxruntime
pip install radon
pip install ultralytics opencv-python mediapipe albumentations keyboard joblib
```

**CRITICAL - Path Configuration:** The `detect_keyboard_live.py` script contains hardcoded absolute paths. Before running on any machine, you MUST update these paths at the top of the script in the `CONFIGURATION` section (around lines 39-99):
- `YOLO_MODEL_PATH` — path to YOLO weights (line 39)
- `SAVE_DIR` — where CSV test results are saved (line 50)
- `RESULTS_DIR` — where pre-trained ML artifacts (SVM, KNN, RF) are stored (line 51)
- `EXPECTED_PATH` — where backend writes expected_words.json (line 99)

**Example fix for cross-platform compatibility:**
```python
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "notebooks/runs/train/keyboard_key_detector/weights/best.pt")
```

## Architecture Overview

This is a monorepo typing-tutor application with three layers:

### Frontend (React + Vite + TypeScript)
- **[frontend/](frontend/)** — React 19 SPA with Vite, Tailwind CSS, shadcn/ui, React Router v6, TanStack Query
- Routing split between student pages (Dashboard, Learn with 6 keyboard modules, Play, Settings) and teacher pages (Dashboard, Classroom management, Settings)
- Auth pages handle login/register with JWT stored client-side
- The `@` alias resolves to `./src` (configured in [frontend/vite.config.ts](frontend/vite.config.ts))
- Custom retro-styled components: PixelButton, PixelCard, PixelInput in `frontend/src/components/`
- **Play Session Components** (real-time typing with ML detection):
  - `VideoFeed` — Camera feed with ML detection visualization overlay
  - `TextPrompt` — Text display with character-level feedback (correct/incorrect highlighting)
  - `VirtualKeyboard` — Interactive keyboard with active key highlighting
  - `MetricsPanel` — Real-time WPM, accuracy, finger accuracy, timing variance
  - `ErrorQueue` — Live error feed showing incorrect keys/fingers
  - `CalibrationOverlay` — Guides user through 27-key calibration before typing
  - `DetectionErrorOverlay` — Handles camera/detection failures
  - `SessionComplete` — Post-session results and statistics

### Backend (Express + TypeScript + MongoDB)
- **[backend/src/](backend/src/)** — Express app with routes, models, middleware
- **Models:** User (role: student|teacher), Classroom (teacher owns, students request to join), Result (WPM, accuracy per session)
- **Routes:**
  - `authRoutes.ts` — register/login with JWT (7-day expiry)
  - `studentRoutes.ts` — join classroom, fetch enrolled classrooms
  - `teacherRoutes.ts` — create/manage classrooms, approve/reject student requests
  - `typingRoutes.ts` — returns random exercises by level (1=letters, 2=words, 3=phrases, 4=sentences) from text files in [backend/data/](backend/data/)
  - `detectionRoutes.ts` — spawns/kills the Python detection subprocess via `POST /start` and `/stop`, provides SSE stream at `GET /stream`, and accepts expected keys via `POST /expected`
  - `resultsRoutes.ts` — save and fetch typing session results
- **Middleware:** [backend/src/middleware/authMiddleware.ts](backend/src/middleware/authMiddleware.ts) verifies Bearer JWT tokens
- CORS is configured to allow `localhost:5173`

### ML Pipeline (Python / PyTorch)
- **[backend/ml/](backend/ml/)** — self-contained Python module for keyboard detection and typing analysis
- **Real-time detection** (`detect_keyboard_live.py`): Uses YOLOv8 for keyboard key bounding boxes, Mediapipe for hand/finger tracking, and SVM for classifying which finger presses which key. Outputs JSON per keystroke to stdout, which the backend reads. Each JSON event includes a base64-encoded frame showing the annotated camera feed with bounding boxes and hand landmarks.
- **Training pipeline** (notebooks/): TCN, BiLSTM-CRF, and Transformer models trained on keystroke time-series data, exported to ONNX
- **Pre-trained artifacts** in `backend/ml/results/` (SVM, KNN, RF pickle files) and `backend/ml/notebooks/runs/train/` (YOLO weights)
- Detection results are also saved to CSV in `backend/ml/testing/` with timestamped filenames
- **Data flow:** Python stdout → Backend line-buffer parser → SSE push → Frontend EventSource → React state updates → UI rendering (VideoFeed, VirtualKeyboard, MetricsPanel)

## Key Architectural Patterns

1. **Python subprocess spawning for ML inference:** The backend does not import Python ML code directly. On `POST /api/detect/start`, it spawns `detect_keyboard_live.py` as a child process and reads JSON from its stdout. `POST /api/detect/stop` kills the process. This isolates the Python/PyTorch dependency from the Node.js runtime.

   **Real-time communication:** The Python process outputs two types of JSON to stdout:
   - **Calibration events** (`event: "calibration_update"`) during initial setup
   - **Detection events** (`event: "detection"`) for each keystroke with key, finger, hand, correctness, and base64-encoded frame

   The backend parses these via line-buffered stdout and pushes them to the frontend via **Server-Sent Events (SSE)** at `GET /api/detect/stream`, eliminating the need for client polling.

2. **Level-based typing content:** Exercises are fetched from plain text files in [backend/data/](backend/data/) — `letters.txt`, `words.txt`, `phrases.txt`, `sentences.txt`. The `/api/typing/level/:id` route maps level numbers 1–4 to these files and returns a random sample.

3. **Classroom approval workflow:** Students request to join via classroom code; teachers must explicitly approve or reject requests. Pending requests are stored as a subdocument array on the Classroom model.

4. **Environment configuration:** All runtime config lives in the root `.env` file — `PORT`, `MONGO_URI`, `JWT_SECRET`, `MODEL_PATH` (YOLO), `FEATURE_CFG` (preprocessing config). The backend reads these via [backend/src/config/env.ts](backend/src/config/env.ts).

5. **Monorepo with root scripts:** The root `package.json` uses `concurrently` to run frontend and backend together. Frontend dev server proxies `/api/*` to `http://localhost:5000` (configured in [frontend/vite.config.ts](frontend/vite.config.ts)).

6. **Calibration workflow:** Before each typing session, users must complete a calibration phase where they press all 27 QWERTY letter keys (a-z + space) to allow YOLO to detect the keyboard bounding boxes. The Python script tracks progress via `calibration_update` events and signals completion with `calibration_done`. The frontend displays a `CalibrationOverlay` guiding users through this process.

7. **Deterministic validation with ground truth mappings:** The Python detection script enforces hardcoded `KEY_TO_EXPECTED_HAND` and `KEY_TO_EXPECTED_FINGER` dictionaries that define the correct touch-typing form. These override the SVM classifier to prevent false positives where incorrect hand/finger placement might be misclassified as correct. A keystroke is only marked correct if both the SVM predicts it AND the detected hand/finger match the ground truth.

8. **SSE (Server-Sent Events) for real-time streaming:** The backend maintains a set of SSE client connections at `GET /api/detect/stream`. When the Python process outputs JSON, the backend immediately pushes it to all connected clients without polling. This provides sub-100ms latency for keystroke feedback and frame updates.

## Additional Documentation

The repository contains several technical documentation files that provide implementation details:

- **[FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md)** — Details the calibration popup and live feed architecture
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** — Scrolling text animation implementation notes
- **[SCROLLING_TEXT_SOLUTION.md](SCROLLING_TEXT_SOLUTION.md)** — Specific solution details for text scrolling feature
- **[VIDEO_FEED_FIX.md](VIDEO_FEED_FIX.md)** — Video feed performance and rendering fixes
- **[FPS_FIX_SUMMARY.md](FPS_FIX_SUMMARY.md)** — Frame rate optimization documentation
- **[POST_CALIBRATION_OPTIMIZATION.md](POST_CALIBRATION_OPTIMIZATION.md)** — Performance improvements after calibration

These files contain implementation context and should be consulted when working on related features.

## Testing

The codebase includes test files using Vitest:
- `backend/src/__tests__/api.routes.test.ts` — Backend API route tests
- `frontend/src/pages/__test__/PlaySession.test.tsx` — Frontend component tests

**Note:** Test scripts are not currently configured in package.json. To run tests, you would need to add Vitest configuration and test scripts to the respective package.json files.

## Common Issues & Troubleshooting

### Camera Not Detected
- Check `CAMERA_INDEX` in `detect_keyboard_live.py` (line 41). Default is `1` — try `0` for built-in webcams or `2+` for multiple external cameras.
- Verify camera permissions are granted to the browser and Python process.

### MongoDB Connection Failed
- Ensure MongoDB is running: `mongod` or check if running as a system service.
- Verify `MONGO_URI` in `.env` matches your MongoDB configuration.

### Python Process Fails to Start
- Verify all Python dependencies are installed (see ML Dependencies section).
- Check that hardcoded paths in `detect_keyboard_live.py` exist and point to correct model files.
- Look for Python errors in the backend console output prefixed with `[PYTHON ERROR]`.

### YOLO Model Not Found
- Ensure YOLO weights exist at the path specified in `YOLO_MODEL_PATH`.
- Default location: `backend/ml/notebooks/runs/train/keyboard_key_detector/weights/best.pt`
- You may need to train the model first or download pre-trained weights.

### Port Already in Use
- Frontend (5173) or backend (5000) ports may be occupied.
- Kill existing processes or change ports in `.env` (backend) and `frontend/vite.config.ts` (dev server proxy).
