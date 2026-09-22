import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\КП\dataset")

X = np.load(DATA_DIR / "X.npy")  # (204, 300, 137, 3)
y = np.load(DATA_DIR / "y.npy")

with open(DATA_DIR / "markers.txt", encoding="utf-8") as f:
    markers = f.read().splitlines()

# Индексы ключевых маркеров
idx_l_hip = markers.index("L_Hip Joint")
idx_r_hip = markers.index("R_Hip Joint")
idx_head = markers.index("HEAD")

print(f"L_Hip Joint: {idx_l_hip}, R_Hip Joint: {idx_r_hip}, HEAD: {idx_head}")

# 1. Центрирование по тазу
pelvis = (X[:, :, idx_l_hip, :] + X[:, :, idx_r_hip, :]) / 2  # (N, 300, 3)
X_centered = X - pelvis[:, :, None, :]  # broadcast

# 2. Масштабирование: расстояние таз → голова
scale = np.linalg.norm(X_centered[:, :, idx_head, :], axis=-1)  # (N, 300)
scale = scale.mean(axis=1, keepdims=True)  # (N, 1) — средний рост по кадрам
X_norm = X_centered / scale[:, :, None, None]

print(f"X_norm: {X_norm.shape}")
print(f"Средний рост (мм): {scale.mean():.1f}")
print(f"Диапазон X_norm: [{X_norm.min():.2f}, {X_norm.max():.2f}]")

np.save(DATA_DIR / "X_norm.npy", X_norm)
print(f"Сохранено: {DATA_DIR / 'X_norm.npy'}")