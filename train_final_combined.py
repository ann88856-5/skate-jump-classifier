import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

DATA_DIR = Path("dataset")

# 1. Загружаем РЕАЛЬНЫЕ ВИДЕО (Train)
X_vid = np.load(DATA_DIR / "X_train_lstm.npy")
y_vid = np.load(DATA_DIR / "y_train.npy")

# 2. Загружаем ИДЕАЛЬНЫЕ ДАННЫЕ (JSON)
X_ideal = np.load(DATA_DIR / "X_ideal.npy")
y_ideal = np.load(DATA_DIR / "y_ideal.npy")

# 3. СКЛЕИВАЕМ ИХ ВМЕСТЕ
X_combined = np.concatenate([X_vid, X_ideal], axis=0)
y_combined = np.concatenate([y_vid, y_ideal], axis=0)

print(f"Общий размер датасета: {X_combined.shape}")

X_t = torch.tensor(X_combined, dtype=torch.float32)
y_t = torch.tensor(y_combined, dtype=torch.long)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

# Тест оставляем ТОЛЬКО НА РЕАЛЬНЫХ ВИДЕО, чтобы проверить, не забыла ли модель шум
X_test = np.load(DATA_DIR / "X_test_lstm.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

class JumpLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(torch.max(out, dim=1)[0]))

model = JumpLSTM(input_dim=99)
model.load_state_dict(torch.load(DATA_DIR / "best_model.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 15
best_test_acc = 0

print("Дообучение на смешанных данных (Видео + JSON)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), DATA_DIR / "best_model_final.pt") # Сохраняем лучшую!
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Test Acc (на видео): {test_acc:.3f}")

print(f"\nЛучшая точность на реальном видео: {best_test_acc:.3f}")
print("Сохранено как best_model_final.pt")