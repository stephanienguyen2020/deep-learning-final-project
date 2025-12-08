"""
Holistic Landmark Extraction for WLASL
--------------------------------------
Outputs:
    - dataset_index.csv
    - asl_landmarks.pkl  (dict: video_id → (T, D) landmarks)
"""

import os
import json
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle

# ============================================================
# CONFIG
# ============================================================
WLASL_ROOT = "wlasl-complete/"
JSON_PATH = WLASL_ROOT + "WLASL_v0.3.json"
VIDEO_DIR = WLASL_ROOT + "videos/"

NUM_FRAMES = 60
NUM_WORDS = 40
SAVE_DIR = "processed_wlasl_top40_holistic"
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_INDEX = f"{SAVE_DIR}/dataset_index.csv"
OUTPUT_PICKLE = f"{SAVE_DIR}/asl_landmarks.pkl"

# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================
mp_holistic = mp.solutions.holistic

HOLISTIC = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# LANDMARK HELPERS
# ============================================================


def safe_lm(lm_list, size):
    """Convert mediapipe landmark list → flattened array of size N*3."""
    if lm_list is None:
        return np.zeros(size * 3, dtype=np.float32)
    arr = np.zeros((size, 3), dtype=np.float32)
    for i, lm in enumerate(lm_list):
        if i >= size:
            break
        arr[i] = [lm.x, lm.y, lm.z]
    return arr.reshape(-1)


def normalize_landmarks(x):
    """Normalize coordinates by centering and scaling using torso size."""
    x = x.copy()
    torso_idx1, torso_idx2 = 11, 12  # shoulders

    # safety: if no pose exists, just return unchanged
    if x[torso_idx1*3] == 0 and x[torso_idx2*3] == 0:
        return x

    p1 = x[torso_idx1*3:torso_idx1*3+3]
    p2 = x[torso_idx2*3:torso_idx2*3+3]

    center = (p1 + p2) / 2
    scale = np.linalg.norm(p1 - p2) + 1e-6

    x = x.reshape(-1, 3)
    x = (x - center) / scale
    return x.reshape(-1)


# ============================================================
# EXTRACT SINGLE VIDEO
# ============================================================

def extract_video_landmarks(video_path, num_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = HOLISTIC.process(rgb)
        frames.append(result)

    cap.release()

    if len(frames) == 0:
        return None

    # Balanced temporal sampling
    idxs = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    seq = []

    for idx in idxs:
        r = frames[idx]

        # extract hands
        lh = safe_lm(
            r.left_hand_landmarks.landmark if r.left_hand_landmarks else None, 21)
        rh = safe_lm(
            r.right_hand_landmarks.landmark if r.right_hand_landmarks else None, 21)

        # pose (33 keypoints)
        pose = safe_lm(
            r.pose_landmarks.landmark if r.pose_landmarks else None, 33)

        # optional: face (use 50 for compactness)
        face = safe_lm(
            r.face_landmarks.landmark[:50] if r.face_landmarks else None, 50)

        # ~ (21+21+33+50)*3 = 375*3 = 1125 dims
        vec = np.concatenate([lh, rh, pose, face])

        vec = normalize_landmarks(vec)
        seq.append(vec)

    return np.array(seq, dtype=np.float32)


# ============================================================
# LOAD JSON
# ============================================================
print("Loading WLASL metadata...")
df = pd.read_json(JSON_PATH)

print("Scanning for available videos...")
word_counts = {}

for idx, row in df.iterrows():
    gloss = row["gloss"]
    valid = [ins["video_id"] for ins in row["instances"]
             if os.path.exists(f"{VIDEO_DIR}{ins['video_id']}.mp4")]
    if len(valid) > 0:
        word_counts[gloss] = len(valid)

sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
selected_words = [w for w, _ in sorted_words[:NUM_WORDS]]

print("Selected words:", selected_words)

# ============================================================
# EXTRACTION LOOP
# ============================================================
dataset = []
landmarks_dict = {}

for idx, row in df.iterrows():
    if row["gloss"] not in selected_words:
        continue

    gloss = row["gloss"]
    print(f"\nProcessing gloss: {gloss}")

    for ins in tqdm(row["instances"]):
        vid = ins["video_id"]
        path = f"{VIDEO_DIR}{vid}.mp4"

        if not os.path.exists(path):
            continue

        lm = extract_video_landmarks(path, NUM_FRAMES)
        if lm is None:
            continue

        landmarks_dict[str(vid)] = lm
        dataset.append({
            "id": vid,
            "path": path,
            "label": gloss,
            "num_frames": lm.shape[0]
        })

# ============================================================
# SAVE OUTPUTS
# ============================================================
df_out = pd.DataFrame(dataset)
df_out.to_csv(OUTPUT_INDEX, index=False)
with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(landmarks_dict, f)

print(f"Saved index → {OUTPUT_INDEX}")
print(f"Saved landmarks → {OUTPUT_PICKLE}")
