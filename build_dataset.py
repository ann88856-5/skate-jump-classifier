import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

# ПУТИ
DATA_ROOT = Path("videos")
OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ПУТЬ К ФАЙЛУ МОДЕЛИ 
MODEL_PATH = "pose_landmarker_full.task" 

CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
TARGET_FRAMES = 300
MIN_FRAMES = 100

def process_single_video(args):
    video_file, cls_idx = args
    cap = cv2.VideoCapture(str(video_file))
    frames_landmarks = []
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = landmarker.detect_for_video(mp_image, int(frame_idx * (1000.0 / fps)))
                frame_idx += 1
                
                if results.pose_landmarks:
                    frames_landmarks.append([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks[0]])
                else:
                    if frames_landmarks: frames_landmarks.append(frames_landmarks[-1])
                    else: frames_landmarks.append([[0.0, 0.0, 0.0]] * 33)
    except Exception as e:
        return None
    finally:
        cap.release()

    positions = np.array(frames_landmarks, dtype=np.float32)
    if positions.shape[0] < MIN_FRAMES:
        return None

    # Нормализация
    left_hip, right_hip = positions[:, 23, :], positions[:, 24, :]
    positions = positions - np.expand_dims((left_hip + right_hip) / 2.0, axis=1)

    # Фикс кадров
    n = positions.shape[0]
    if n > TARGET_FRAMES:
        positions = positions[(n - TARGET_FRAMES) // 2:(n - TARGET_FRAMES) // 2 + TARGET_FRAMES]
    elif n < TARGET_FRAMES:
        positions = np.concatenate([positions, np.repeat(positions[-1:], TARGET_FRAMES - n, axis=0)])

    return positions, cls_idx

def build_dataset_fast():
    tasks = []
    for class_dir in sorted(DATA_ROOT.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_IDX: continue
        for video_file in sorted(class_dir.glob("*.mp4")):
            tasks.append((video_file, CLASS_TO_IDX[class_dir.name]))
    
    print(f"Всего видео для обработки: {len(tasks)}")

    X_list, y_list = [], []
    
    # используем все ядра процессора чтоы быстрее было
    workers = os.cpu_count() 
    
    # Запускаем пул процессов
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # executor.map возвращает результаты по мере готовности
        for i, result in enumerate(executor.map(process_single_video, tasks)):
            if result is not None:
                positions, cls_idx = result
                X_list.append(positions)
                y_list.append(cls_idx)
            
            if (i + 1) % 50 == 0:
                print(f"Обработано: {i + 1}/{len(tasks)} видео...")

    print("\nСохранение датасета...")
    np.save(OUTPUT_DIR / "X.npy", np.stack(X_list))
    np.save(OUTPUT_DIR / "y.npy", np.array(y_list))
    print(f"ГОТОВО! X = {np.array(X_list).shape}")

if __name__ == "__main__":
    build_dataset_fast()