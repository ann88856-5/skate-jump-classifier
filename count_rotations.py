import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\КП\dataset")
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

with open(DATA_DIR / "markers.txt", encoding="utf-8") as f:
    markers = f.read().splitlines()

idx_l_hip = markers.index("L_Hip Joint")    # 9
idx_r_hip = markers.index("R_Hip Joint")    # 44
idx_l_sho = markers.index("L_Shoulder Joint")  # 84
idx_r_sho = markers.index("R_Shoulder Joint")  # 111

CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]

print(f"L_Hip: {idx_l_hip}, R_Hip: {idx_r_hip}")
print(f"L_Sho: {idx_l_sho}, R_Sho: {idx_r_sho}")


def count_rotations(pose_seq):
    """Считает обороты по углу линии бёдер (L_Hip → R_Hip) в плоскости XZ."""
    l_hip = pose_seq[:, idx_l_hip, :]
    r_hip = pose_seq[:, idx_r_hip, :]

    # Вектор от левого бедра к правому
    vec = r_hip - l_hip  # (frames, 3)

    # Угол в горизонтальной плоскости (XZ)
    angle = np.arctan2(vec[:, 2], vec[:, 0])

    # Разворачиваем
    angle_unwrapped = np.unwrap(angle)

    rotations = (angle_unwrapped[-1] - angle_unwrapped[0]) / (2 * np.pi)
    return abs(rotations)


def count_rotations_shoulders(pose_seq):
    """Считает обороты по углу линии плеч (L_Shoulder → R_Shoulder)."""
    l_sho = pose_seq[:, idx_l_sho, :]
    r_sho = pose_seq[:, idx_r_sho, :]
    vec = r_sho - l_sho
    angle = np.arctan2(vec[:, 2], vec[:, 0])
    angle_unwrapped = np.unwrap(angle)
    rotations = (angle_unwrapped[-1] - angle_unwrapped[0]) / (2 * np.pi)
    return abs(rotations)


print("\nОбороты по линии бёдер:")
for i in range(20):
    rot = count_rotations(X[i])
    print(f"Пример {i:2d}: класс={CLASSES[y[i]]:8s}, оборотов={rot:.2f}")

print("\nОбороты по линии плеч:")
for i in range(20):
    rot = count_rotations_shoulders(X[i])
    print(f"Пример {i:2d}: класс={CLASSES[y[i]]:8s}, оборотов={rot:.2f}")