import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import urllib.request

# Пути к видео
VIDEO_IN = Path(r"C:\КП\skater_A\cam_1\Axel_1.mp4")
VIDEO_OUT = Path(r"C:\КП\output_skeleton.mp4")
MODEL_PATH = Path(r"C:\КП\pose_landmarker.task")

# Скачиваем модель, если её нет
if not MODEL_PATH.exists():
    print("Скачиваем модель Pose Landmarker...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Модель скачана.")

# Соединения для скелета (индексы точек 0-32)
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
        start_idx = connection[0]
        end_idx = connection[1]
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

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Настройка Tasks API
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
            
            if frame_idx % 30 == 0:
                print(f"Обработано: {frame_idx}")

    cap.release()
    out.release()
    print(f"Сохранено: {output_path}")

if __name__ == "__main__":
    process_video(VIDEO_IN, VIDEO_OUT)