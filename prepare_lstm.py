import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\КП\dataset")

X_train = np.load(DATA_DIR / "X_train.npy")  # (N, 300, 137, 3)
X_test = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# (N, 300, 137, 3) → (N, 300, 411)
X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

print(f"X_train: {X_train_flat.shape}")
print(f"X_test:  {X_test_flat.shape}")

np.save(DATA_DIR / "X_train_lstm.npy", X_train_flat)
np.save(DATA_DIR / "X_test_lstm.npy", X_test_flat)

print("Готово!")