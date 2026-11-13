import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path

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

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames_landmarks = []
    
    # Инициализация НОВОГО API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 30.0  # Если FPS не читается, ставим 30
            
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Конвертация кадра для нового API
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Временная метка кадра
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            frame_idx += 1
            
            # Детекция
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                frame_data = []
                # В новом API это список списков (на случай если в кадре несколько людей)
                for lm in results.pose_landmarks[0]:
                    frame_data.append([lm.x, lm.y, lm.z])
                frames_landmarks.append(frame_data)
            else:
                # Если человека не видно, дублируем прошлый кадр
                if frames_landmarks:
                    frames_landmarks.append(frames_landmarks[-1])
                else:
                    frames_landmarks.append([[0.0, 0.0, 0.0]] * 33)
                    
    cap.release()
    return np.array(frames_landmarks, dtype=np.float32)

def normalize_landmarks(positions):
    # Центрируем скелет по тазу (23 и 24 точки)
    left_hip = positions[:, 23, :]
    right_hip = positions[:, 24, :]
    pelvis_center = (left_hip + right_hip) / 2.0
    pelvis_center = np.expand_dims(pelvis_center, axis=1) 
    return positions - pelvis_center

def fix_frames(arr, target=TARGET_FRAMES):
    n = arr.shape[0]
    if n == target:
        return arr
    if n > target:
        start = (n - target) // 2
        return arr[start:start + target]
    pad = np.repeat(arr[-1:], target - n, axis=0)
    return np.concatenate([arr, pad], axis=0)

def build_dataset():
    X_list, y_list = [], []
    
    for class_dir in sorted(DATA_ROOT.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_IDX:
            continue
            
        cls = class_dir.name
        print(f"Обработка класса: {cls}")
        
        # Ищем видеофайлы
        for video_file in sorted(list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))):
            print(f"  Читаю: {video_file.name}")
            positions = extract_landmarks_from_video(video_file)
            n_frames = positions.shape[0]
            
            if n_frames < MIN_FRAMES:
                print(f"  Пропуск (мало кадров {n_frames}): {video_file.name}")
                continue
                
            positions = normalize_landmarks(positions)
            positions = fix_frames(positions, TARGET_FRAMES)
            
            X_list.append(positions)
            y_list.append(CLASS_TO_IDX[cls])

    if len(X_list) == 0:
        print("\nОШИБКА: Не найдено ни одного видео! Убедись, что в папке C:\\КП\\json лежат папки с .mp4 файлами.")
        return

    X = np.stack(X_list)
    y = np.array(y_list)

    print(f"\nГотово: X = {X.shape}, y = {y.shape}")
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

if __name__ == "__main__":
    build_dataset()