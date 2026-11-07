import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\KP\dataset")

# Загружаем данные
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

num_classes = 6
X_train, X_test, y_train, y_test = [], [], [], []

np.random.seed(42) # Чтобы результат был одинаковым при перезапуске

for c in range(num_classes):
    idx = np.where(y == c)[0]
    np.random.shuffle(idx)
    
    # Берем 80% в трейн, 20% в тест для каждого класса отдельно
    split_point = int(len(idx) * 0.8)
    train_idx = idx[:split_point]
    test_idx = idx[split_point:]
    
    X_train.append(X[train_idx])
    X_test.append(X[test_idx])
    y_train.append(y[train_idx])
    y_test.append(y[test_idx])

# Склеиваем обратно
X_train = np.concatenate(X_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_test = np.concatenate(y_test, axis=0)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Сохраняем
np.save(DATA_DIR / "X_train.npy", X_train)
np.save(DATA_DIR / "X_test.npy", X_test)
np.save(DATA_DIR / "y_train.npy", y_train)
np.save(DATA_DIR / "y_test.npy", y_test)

print("Готово!")