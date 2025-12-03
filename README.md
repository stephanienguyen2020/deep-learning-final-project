## Multi-Level ASL Recognition System: Complete Pipeline

### Software Requirements

```bash
# Python 3.10+ recommended
python --version

# Install dependencies
pip install -r requirements.txt
# OR using uv
uv sync
```

### Required Packages

- TensorFlow 2.x
- MediaPipe
- OpenCV (cv2)
- NumPy, Pandas, scikit-learn
- FastAPI, Uvicorn (for API server)
- Jupyter Notebook/Lab

---

## Dataset Setup

### 1. Download the Kaggle Dataset

Download the ASL dataset from Kaggle:

- **URL**: https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset/data

### 2. Organize the Data

Place the downloaded images in the `data/` folder with the following structure:

```
data/
├── A/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── B/
│   └── ...
├── C/
│   └── ...
...
├── Z/
├── 0/
├── 1/
...
├── 9/
└── _/          # Space/neutral gesture
```

> **Note**: The `data/` folder is gitignored. You must download and set up the dataset locally.

---

## Step 1: Training Phase (Notebooks)

Run the notebooks in order. Each notebook builds on the output of the previous one.

### Notebook 1: Preprocessing (`notebooks/1.preprocessing.ipynb`)

**Purpose**: Extract hand landmarks from images using MediaPipe

**Input**: Raw images from `data/` folder

**Output**: `model/keypoint_classifier/keypoint2.csv`

```bash
# Open Jupyter and run the notebook
jupyter notebook notebooks/1.preprocessing.ipynb
```

**What it does**:

1. Loads images from each class folder (A-Z, 0-9, \_)
2. Uses MediaPipe Hands to detect 21 hand landmarks
3. Converts absolute coordinates to relative coordinates (normalized to wrist)
4. Outputs 42 features per sample (21 points × 2 coordinates)
5. Saves to CSV format: `[label, x0, y0, x1, y1, ..., x20, y20]`

---

### Notebook 2: Data Analysis & Model Training (`notebooks/2.data_analysis.ipynb`)

**Purpose**: Train the letter-level classifier (A-Z, 0-9)

**Input**: `model/keypoint_classifier/keypoint2.csv`

**Output**:

- `model/keypoint_classifier/keypoint_classifier.h5` (Keras model)
- `model/keypoint_classifier/keypoint_classifier.tflite` (TFLite for inference)

```bash
jupyter notebook notebooks/2.data_analysis.ipynb
```

**What it does**:

1. Loads preprocessed landmark data from CSV
2. Applies feature selection (removes low-variance features: 42 → 40 features)
3. Splits data: 75% training, 25% testing
4. Trains a Dense neural network:
   - Input: 40 features
   - Dense(128) + ReLU + Dropout(0.3)
   - Dense(64) + ReLU + Dropout(0.3)
   - Dense(37) + Softmax (37 classes)
5. Uses early stopping and model checkpointing
6. Converts to TFLite for efficient inference

**Expected Results**: ~96% test accuracy

---

### Notebook 3: Point History Classification (`notebooks/3.point_history_classification.ipynb`)

**Purpose**: Train the word-level gesture classifier (temporal motion patterns)

**Input**: `model/point_history_classifier/point_history.csv`

**Output**:

- `model/point_history_classifier/point_history_classifier.hdf5`
- `model/point_history_classifier/point_history_classifier.tflite`

```bash
jupyter notebook notebooks/3.point_history_classification.ipynb
```

**What it does**:

1. Tracks fingertip position over 16 frames (temporal sequence)
2. Input: 32 values (16 points × 2 coordinates)
3. Trains either Dense or LSTM network
4. Classifies motion patterns (e.g., circle, swipe, zigzag)

> **Note**: This notebook requires collecting point history data first. Use `app_with_point_history.py` with mode `2` to record gesture motions.

---

## Step 2: Deployment Options

After training, you have the `.tflite` models ready for inference.

### Create Label Files

Before running inference, create the label CSV files:

**`model/keypoint_classifier/keypoint_classifier_label.csv`**:

```csv
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
0
1
2
3
4
5
6
7
8
9
_
```

**`model/point_history_classifier/point_history_classifier_label.csv`**:

```csv
Stop
Clockwise
Counter Clockwise
Move
```

---

### Option A: FastAPI Backend Server

Start the API server that handles ASL predictions:

```bash
# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Features**:

- WebSocket endpoint at `/api/asl/predict`
- REST endpoints for health checks and available signs
- MongoDB integration for user management
- Prometheus metrics at `/metrics`
- Jaeger tracing (when enabled)

**API Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/asl/predict` | WebSocket | Real-time prediction |
| `/api/asl/signs` | GET | List available signs |
| `/api/asl/health` | GET | ASL service health |

---

### Option B: Real-time Client Application

Run the webcam client that connects to the API server:

```bash
# Connect to local server
python app.py --ws_url ws://localhost:8000/api/asl/predict

# Connect to remote server
python app.py --ws_url ws://your-server.com/api/asl/predict
```

**Command Line Options**:

```bash
python app.py \
    --device 0 \                    # Camera device index
    --width 960 \                   # Frame width
    --height 540 \                  # Frame height
    --ws_url ws://localhost:8000/api/asl/predict \
    --min_detection_confidence 0.7 \
    --min_tracking_confidence 0.5
```

**Controls**:

- `ESC`: Exit the application

---

### Option C: Standalone Mode with Point History

For local inference with gesture motion tracking:

```bash
python app_with_point_history.py
```

**Modes** (press number keys to switch):

- `0`: Normal prediction mode
- `1`: Record keypoint data (for training)
- `2`: Record point history data (for gesture training)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Kaggle Dataset]                                                    │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────────────┐                                            │
│  │  1.preprocessing    │  MediaPipe extracts 21 hand landmarks      │
│  │     .ipynb          │                                            │
│  └─────────┬───────────┘                                            │
│            │                                                         │
│            ▼                                                         │
│     keypoint2.csv (42 features per sample)                          │
│            │                                                         │
│            ▼                                                         │
│  ┌─────────────────────┐                                            │
│  │  2.data_analysis    │  Train Dense NN: 40→128→64→37              │
│  │     .ipynb          │                                            │
│  └─────────┬───────────┘                                            │
│            │                                                         │
│            ▼                                                         │
│     keypoint_classifier.tflite                                       │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│  │   Webcam     │ ───▶ │  MediaPipe   │ ───▶ │   app.py     │       │
│  │   Client     │      │  (21 points) │      │  (Client)    │       │
│  └──────────────┘      └──────────────┘      └──────┬───────┘       │
│                                                      │               │
│                                              WebSocket│               │
│                                                      ▼               │
│                                             ┌──────────────┐         │
│                                             │   main.py    │         │
│                                             │  (FastAPI)   │         │
│                                             └──────┬───────┘         │
│                                                    │                 │
│                                                    ▼                 │
│                                      ┌─────────────────────────┐    │
│                                      │  KeyPointClassifier     │    │
│                                      │  (TFLite Inference)     │    │
│                                      └─────────────────────────┘    │
│                                                    │                 │
│                                                    ▼                 │
│                                          Prediction: "A", "B", ...   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Goals Roadmap

| Level                            | Description                                 | Status         | Model                             |
| -------------------------------- | ------------------------------------------- | -------------- | --------------------------------- |
| **Letter-Level**                 | Recognize A-Z, 0-9 (37 classes)             | ✅ Implemented | `keypoint_classifier.tflite`      |
| **Word-Level (Reach Goal)**      | Recognize temporal letter sequences → words | 🔄 In Progress | `point_history_classifier.tflite` |
| **Sentence-Level (Super Reach)** | LLM integration for sentence generation     | 📋 Planned     | GPT/LLM API                       |

---

## Troubleshooting

### Common Issues

**1. "Model not found" error**

```
Ensure the .tflite files exist:
- model/keypoint_classifier/keypoint_classifier.tflite
- model/point_history_classifier/point_history_classifier.tflite
```

**2. "No hand detected"**

- Ensure good lighting
- Keep hand within camera frame
- Adjust `--min_detection_confidence` (lower value = more sensitive)

**3. WebSocket connection failed**

- Verify the server is running: `uvicorn main:app --reload`
- Check the WebSocket URL matches the server address

**4. Low accuracy predictions**

- Retrain with more diverse data
- Ensure hand orientation matches training data
- Check if label CSV files are correctly ordered

### Environment Variables

Create a `.env` file for configuration:

```env
SECRET_KEY=your-secret-key
MONGODB_URL=mongodb://localhost:27017
ENABLE_TRACING=false
```
