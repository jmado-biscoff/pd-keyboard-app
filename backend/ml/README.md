Machine Learning Module – Typing Behavior Analysis

This folder contains the Machine Learning pipeline for the Typing Behavior Web Application, which uses keystroke dynamics to analyze typing patterns and classify performance. The models are trained on time-series keystroke data (e.g., Press_Time, Release_Time, Hold_Time) and evaluated across multiple architectures.

Folder Structure
backend/ml
├── dataset/
│ ├── raw/ # Raw keystroke CSV files (per user/session)
│ ├── processed/ # Preprocessed NumPy arrays and feature configs
│
├── models/ # Trained model weights (.pt) and exported ONNX files (.onnx)
│
├── notebooks/
│ ├── preprocessing.py
│ ├── train_TCN.py
│ ├── train_BiLSTM_CRF.py
│ ├── train_Transformer.py
│
├── evaluation/
│ ├── evaluation_TCN.py
│ ├── evaluation_BiLSTM_CRF.py
│ ├── evaluation_Transformer.py
│ ├── evaluation_maintainability.py
│ ├── evaluation_summary.py
│ ├── compare_models.py
│
└── **init**.py

Dependencies

Install Python 3.10+ then install the required dependencies:

pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm matplotlib
pip install onnx onnxruntime
pip install radon

(Optional) Create a virtual environment:

python -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows

Data Preparation

Place all raw keystroke CSV files inside:

backend/ml/dataset/raw/

Run preprocessing to generate NumPy datasets:

python backend/ml/notebooks/preprocessing.py

This will create:

X_train.npy, X_val.npy, X_test.npy

y_train.npy, y_val.npy, y_test.npy

feature_config.json

inside backend/ml/dataset/processed/.

Model Training

Train each model:

python backend/ml/notebooks/train_TCN.py
python backend/ml/notebooks/train_BiLSTM_CRF.py
python backend/ml/notebooks/train_Transformer.py

Each script will:

Load processed data

Train the model

Measure and save training time

Save model weights (.pt) and ONNX exports (.onnx)

Save training metadata in evaluation/\*.json

Model Evaluation

After training, evaluate each model:

python backend/ml/evaluation/evaluation_TCN.py
python backend/ml/evaluation/evaluation_BiLSTM_CRF.py
python backend/ml/evaluation/evaluation_Transformer.py

Each evaluation script produces a \*\_metrics.json file with:

Precision, Recall, F1

Inference time

Model size (ONNX & checkpoint)

Training time (loaded from train JSON)

Maintainability Index

To analyze maintainability using radon:

python backend/ml/evaluation/evaluation_maintainability.py

This generates backend/ml/evaluation/maintainability_report.json containing MI scores per file.

Model Comparison

Generate a table and plot comparing TCN, BiLSTM-CRF, and Transformer:

python backend/ml/evaluation/compare_models.py

Outputs:

model_comparison.csv

model_comparison.png

Recommended Run Order

# 1. Preprocessing

python backend/ml/notebooks/preprocessing.py

# 2. Train models

python backend/ml/notebooks/train_TCN.py
python backend/ml/notebooks/train_BiLSTM_CRF.py
python backend/ml/notebooks/train_Transformer.py

# 3. Evaluate models

python backend/ml/evaluation/evaluation_TCN.py
python backend/ml/evaluation/evaluation_BiLSTM_CRF.py
python backend/ml/evaluation/evaluation_Transformer.py

# 4. Maintainability

python backend/ml/evaluation/evaluation_maintainability.py

# 5. Summary & comparison

python backend/ml/evaluation/compare_models.py
