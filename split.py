import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\КП\dataset")

X = np.load(DATA_DIR / "X_norm.npy")
y = np.load(DATA_DIR / "y.npy")

# Читаем meta
with open(DATA_DIR / "meta.txt", encoding="utf-8") as f:
    meta = [line.strip().split("\t") for line in f if line.strip()]

skaters = np.array([m[0] for m in meta])
print("Фигуристы:", np.unique(skaters, return_counts=True))

# Skater_A, B, C → train, Skater_D → test
train_mask = skaters != "Skater_D"
test_mask = skaters == "Skater_D"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"\nTrain: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# Проверка баланса
import collections
print(f"\nTrain классы: {collections.Counter(y_train.tolist())}")
print(f"Test классы:  {collections.Counter(y_test.tolist())}")

np.save(DATA_DIR / "X_train.npy", X_train)
np.save(DATA_DIR / "y_train.npy", y_train)
np.save(DATA_DIR / "X_test.npy", X_test)
np.save(DATA_DIR / "y_test.npy", y_test)
print(f"\nСохранено: X_train, y_train, X_test, y_test")