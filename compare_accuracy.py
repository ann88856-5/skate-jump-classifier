import json, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# 1. Ищем ЛЮБОЙ рабочий JSON
JSON = None
for p in Path("json").rglob("*.json"):
    try:
        with open(p, encoding="utf-8") as f:
            if "Markers" in json.load(f):
                JSON = p
                break
    except: continue

if JSON is None:
    print("Не найден подходящий JSON с маркерами!"); exit()

# 2. Ищем ЛЮБОЕ видео, на котором MediaPipe видит человека
VIDEO = None
frames_mp = []
options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path="pose_landmarker_full.task"), running_mode=vision.RunningMode.VIDEO)

for video_file in Path("videos").rglob("*.mp4"):
    print(f"Проверяю видео: {video_file.name}...")
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened(): continue
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        temp_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(idx * (1000.0/fps)))
            if res.pose_landmarks: 
                temp_frames.append([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks[0]])
            idx += 1
    cap.release()
    
    # Если нашли кадры с человеком, сохраняем и останавливаем поиск
    if len(temp_frames) > 10: 
        frames_mp = temp_frames
        VIDEO = video_file
        print(f"--> Подошло видео: {VIDEO.name}")
        break

if VIDEO is None:
    print("ОШИБКА: Ни на одном видео не найден человек."); exit()

# 3. Считаем ошибку
with open(JSON, encoding="utf-8") as f: data = json.load(f)
markers = [m for m in data["Markers"] if "LAB" not in m["Name"]]
min_len = min(len(m["Parts"][0]["Values"]) for m in markers)
ideal = np.stack([np.array(m["Parts"][0]["Values"], dtype=np.float32)[:min_len, :3] for m in markers[:33]], axis=1)
mp_coords = np.array(frames_mp)

# Нормализация
ideal = ideal - np.expand_dims((ideal[:, 23, :] + ideal[:, 24, :]) / 2.0, axis=1)
mp_coords = mp_coords - np.expand_dims((mp_coords[:, 23, :] + mp_coords[:, 24, :]) / 2.0, axis=1)

min_f = min(len(ideal), len(mp_coords))
error = np.linalg.norm(ideal[:min_f] - mp_coords[:min_f], axis=2)
print(f"\nСредняя ошибка MediaPipe: {np.mean(error):.4f} у.е.")