import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import psycopg
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(r"C:\КП\dataset")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]

MODEL_PATH = Path(r"C:\KP\pose_landmarker.task")

DB_CONFIG = {
    "dbname": "skate_analysis",
    "user": "postgres",
    "password": "qazwsx89",
    "host": "localhost",
    "port": 5432,
    "client_encoding": "UTF8",
}

# Индексы маркеров для подсчёта оборотов
with open(DATA_DIR / "markers.txt", encoding="utf-8") as f:
    MARKERS = f.read().splitlines()
IDX_L_HIP = MARKERS.index("L_Hip Joint")
IDX_R_HIP = MARKERS.index("R_Hip Joint")


class JumpLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)


model = JumpLSTM(input_dim=411)
model.load_state_dict(torch.load(DATA_DIR / "best_model.pt", map_location="cpu"))
model.eval()
print("Модель загружена")


# Соединения для скелета (33 точки MediaPipe)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        if start_point.visibility > 0.5 and end_point.visibility > 0.5:
            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
    for idx, landmark in enumerate(landmarks):
        if landmark.visibility > 0.5:
            x, y = int(landmark.x * w), int(landmark.y * h)
            color = (255, 0, 0) if idx in [23, 24] else (0, 255, 0)
            cv2.circle(frame, (x, y), 4, color, -1)


import os
import subprocess
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


def process_video_with_skeleton(input_path: Path, output_path: Path):
    """Накладывает скелет на видео и перекодирует в H.264."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {input_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создаём временный файл для промежуточного видео (mp4v)
    temp_output = str(output_path).replace('.mp4', '_temp.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # === НАЛОЖЕНИЕ СКЕЛЕТА ===
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.pose_landmarks:
                draw_skeleton(frame, result.pose_landmarks[0])
            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()

    # === ПЕРЕКОДИРОВАНИЕ В H.264 ===
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_output,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',   # для стриминга в браузере
            str(output_path)
        ], check=True, capture_output=True)

        # Удаляем временный файл
        if os.path.exists(temp_output):
            os.remove(temp_output)

        print(f"Видео готово: {output_path}")
        return True
    except FileNotFoundError:
        print("Ошибка: ffmpeg не найден. Установите FFmpeg и добавьте в PATH.")
        # Если ffmpeg нет — оставляем mp4v (не играет в браузере)
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка ffmpeg: {e.stderr.decode('utf-8', errors='ignore')}")
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)
        return True
    except Exception as e:
        print(f"Ошибка перекодирования: {e}")
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)
        return True


def count_rotations(pose_seq):
    l_hip = pose_seq[:, IDX_L_HIP, :]
    r_hip = pose_seq[:, IDX_R_HIP, :]
    vec = r_hip - l_hip
    angle = np.arctan2(vec[:, 2], vec[:, 0])
    angle_unwrapped = np.unwrap(angle)
    rotations = (angle_unwrapped[-1] - angle_unwrapped[0]) / (2 * np.pi)
    return float(abs(rotations))


# Отдаём видео как статику
app.mount("/videos", StaticFiles(directory=str(UPLOAD_DIR)), name="videos")


@app.get("/")
def root():
    return {"status": "ok", "message": "Skate Jump Classifier API"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Принимает видео, накладывает скелет, возвращает тип прыжка и обороты."""
    # Сохраняем исходное видео
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Накладываем скелет
    skeleton_filename = f"skeleton_{file.filename}"
    skeleton_path = UPLOAD_DIR / skeleton_filename
    try:
        process_video_with_skeleton(save_path, skeleton_path)
    except Exception as e:
        print(f"Ошибка наложения скелета: {e}")
        skeleton_filename = None

    # Анализ (заглушка: видео → 3D-позы → Спринт 3)
    X_test_raw = np.load(DATA_DIR / "X_test.npy")
    idx = np.random.randint(0, len(X_test_raw))
    pose_seq = X_test_raw[idx]

    rotations = count_rotations(pose_seq)

    X = pose_seq.reshape(1, 300, -1).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(probs.argmax())

    # Сохраняем в PostgreSQL
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO sessions (filename, jump_type, confidence, rotations)
        VALUES (%s, %s, %s, %s) RETURNING id""",
        (file.filename, CLASSES[pred], float(probs[pred]), rotations)
    )
    session_id = cur.fetchone()[0]

    for i, cls in enumerate(CLASSES):
        cur.execute(
            """INSERT INTO session_probs (session_id, class_name, probability)
            VALUES (%s, %s, %s)""",
            (session_id, cls, float(probs[i]))
        )

    conn.commit()
    cur.close()
    conn.close()

    response = {
        "session_id": session_id,
        "jump_type": CLASSES[pred],
        "confidence": float(probs[pred]),
        "rotations": rotations,
        "all_probs": {CLASSES[i]: float(probs[i]) for i in range(6)},
    }

    if skeleton_filename:
        response["skeleton_url"] = f"/videos/{skeleton_filename}"

    return response