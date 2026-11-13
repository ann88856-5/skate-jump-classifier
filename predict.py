import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn

# --- 1. НАСТРОЙКИ ---
VIDEO_PATH = "test_video.mp4" ### ТЕСТ ВИДЕО МЕНЯТЬ ЗДЕСЬ!!!
MODEL_PATH = "pose_landmarker_full.task"
LSTM_MODEL_PATH = "dataset/best_model.pt" 
CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]

# --- 2. АРХИТЕКТУРА ---
class JumpLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)

model = JumpLSTM(input_dim=99)
model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- 3. ИЗВЛЕЧЕНИЕ ДАННЫХ ---
print(f"Анализ видео: {VIDEO_PATH}...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ОШИБКА: Не удалось открыть видео {VIDEO_PATH}. Проверь, что файл существует и это реально видео.")
    exit()

frames_landmarks = []
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp_ms = int(frame_idx * (1000.0 / fps))
    frame_idx += 1
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    if results.pose_landmarks:
        frame_data = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks[0]]
        frames_landmarks.append(frame_data)
    else:
        if frames_landmarks: frames_landmarks.append(frames_landmarks[-1])
        else: frames_landmarks.append([[0.0, 0.0, 0.0]] * 33)

cap.release()

# --- ЗАЩИТА ОТ ПУСТОТЫ ---
if len(frames_landmarks) == 0 or np.array(frames_landmarks).shape[0] == 0:
    print("ОШИБКА: Видео пустое или MediaPipe не нашел ни одного человека на видео.")
    exit()

# Нормализация
positions = np.array(frames_landmarks, dtype=np.float32)
if positions.ndim != 3:
    print(f"ОШИБКА: Неправильная размерность массива {positions.shape}")
    exit()

left_hip, right_hip = positions[:, 23, :], positions[:, 24, :]
positions = positions - np.expand_dims((left_hip + right_hip) / 2.0, axis=1)

if positions.shape[0] > 300: 
    positions = positions[100:400]
elif positions.shape[0] < 300: 
    positions = np.concatenate([positions, np.repeat(positions[-1:], 300 - positions.shape[0], axis=0)])

# --- 4. ПРЕДСКАЗАНИЕ ---
X_test = torch.tensor(positions.reshape(1, 300, -1), dtype=torch.float32)

with torch.no_grad():
    logits = model(X_test)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()

print("\n" + "="*30)
print(f"РЕЗУЛЬТАТ: {CLASSES[pred_idx]}")
print(f"Уверенность: {probs[pred_idx].item()*100:.1f}%")
print("="*30)