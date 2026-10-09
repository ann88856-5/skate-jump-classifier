import argparse
import json
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Скачиваю модель {MODEL_PATH}... (это займет пару секунд)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Модель скачана!")


def fill_nans(data):
    data = data.copy()
    T, N, C = data.shape
    for j in range(N):
        for k in range(C):
            series = data[:, j, k]
            mask = np.isnan(series)
            if mask.any() and not mask.all():
                idx = np.arange(T)
                series[mask] = np.interp(idx[mask], idx[~mask], series[~mask])
    return data


def extract_pose_from_video(video_path):
    download_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in result.pose_landmarks[0]
            ]
            frames.append(landmarks)
        else:
            frames.append([[np.nan, np.nan, np.nan, 0.0] for _ in range(33)])
        
        frame_idx += 1

    cap.release()

    data = np.array(frames, dtype=np.float32)
    data = fill_nans(data)
    return data, fps


def main():
    parser = argparse.ArgumentParser(description="Видео -> 3D-позы (новый MediaPipe API)")
    parser.add_argument("--video", required=True, help="путь к видео")
    parser.add_argument("--out", default="json/pose.json", help="куда сохранить JSON")
    args = parser.parse_args()

    data, fps = extract_pose_from_video(args.video)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video": str(args.video),
        "fps": float(fps),
        "shape": list(data.shape),
        "landmarks": data.tolist(),
        "landmark_names": [f"lm_{i}" for i in range(33)]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Готово: {out_path}")
    print(f"Кадров: {data.shape[0]}, fps: {fps:.2f}")


if __name__ == "__main__":
    main()