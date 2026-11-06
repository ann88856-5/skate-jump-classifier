import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# --- 1. ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
X_train = torch.FloatTensor(np.load('dataset/X_train_lstm.npy'))
y_train = torch.LongTensor(np.load('dataset/y_train.npy'))
X_test = torch.FloatTensor(np.load('dataset/X_test_lstm.npy'))
y_test = torch.LongTensor(np.load('dataset/y_test.npy'))

print("Уникальные классы в трейне:", torch.unique(y_train))

AXEL_IDX = 0 

# --- 2. АРХИТЕКТУРА ---
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Берем последний временной шаг
        return out

input_size = X_train.shape[2]
hidden_size = 64

# --- 3. ОБУЧЕНИЕ 1-Й СТУПЕНИ (Аксель vs Остальные) ---
print("\n=== СТУПЕНЬ 1: Аксель или нет? ===")
y_train_s1 = (y_train == AXEL_IDX).long()
y_test_s1 = (y_test == AXEL_IDX).long()

model1 = BiLSTM(input_size, hidden_size, num_classes=2) # 2 класса
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.0005)

loader1 = DataLoader(TensorDataset(X_train, y_train_s1), batch_size=8, shuffle=True)

for epoch in range(80):
    model1.train()
    for bx, by in loader1:
        optimizer1.zero_grad()
        loss = criterion1(model1(bx), by)
        loss.backward()
        optimizer1.step()
print("Ступень 1 обучена.")

# --- 4. ОБУЧЕНИЕ 2-Й СТУПЕНИ (5 классов, без Акселя) ---
print("\n=== СТУПЕНЬ 2: Остальные 5 прыжков ===")

mask_train = (y_train != AXEL_IDX)
X_train_s2 = X_train[mask_train]
y_train_s2_raw = y_train[mask_train]

unique_classes = torch.unique(y_train_s2_raw)
class_map = {int(c): i for i, c in enumerate(unique_classes)}
y_train_s2 = torch.tensor([class_map[int(c)] for c in y_train_s2_raw]).long()

model2 = BiLSTM(input_size, hidden_size, num_classes=5) # 5 классов
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters(), lr=0.0005)

loader2 = DataLoader(TensorDataset(X_train_s2, y_train_s2), batch_size=8, shuffle=True)

for epoch in range(100):
    model2.train()
    for bx, by in loader2:
        optimizer2.zero_grad()
        loss = criterion2(model2(bx), by)
        loss.backward()
        optimizer2.step()
print("Ступень 2 обучена.")

# --- 5. СОХРАНЕНИЕ ---
torch.save(model1.state_dict(), 'stage1_axel.pt')
torch.save(model2.state_dict(), 'stage2_others.pt')
print("\nМодели сохранены: stage1_axel.pt и stage2_others.pt")

# --- 6. ТЕСТ (проверка точности) ---
model1.eval()
model2.eval()
with torch.no_grad():
    # Проверяем на тесте
    preds1 = model1(X_test).argmax(dim=1)
    correct1 = (preds1 == y_test_s1).sum().item()
    acc1 = correct1 / len(y_test_s1) * 100
    print(f"Точность Ступени 1 (Аксель/Не Аксель): {acc1:.1f}%")
    
    # Проверяем 2-ю ступень (только на тех, кто не Аксель)
    mask_test = (y_test != AXEL_IDX)
    X_test_s2 = X_test[mask_test]
    y_test_s2_raw = y_test[mask_test]
    y_test_s2 = torch.tensor([class_map[int(c)] for c in y_test_s2_raw]).long()
    
    preds2 = model2(X_test_s2).argmax(dim=1)
    correct2 = (preds2 == y_test_s2).sum().item()
    acc2 = correct2 / len(y_test_s2) * 100
    print(f"Точность Ступени 2 (5 классов): {acc2:.1f}%")