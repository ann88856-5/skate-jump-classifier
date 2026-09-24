import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path(r"C:\КП\dataset")

X_train = np.load(DATA_DIR / "X_train_lstm.npy")
X_test = np.load(DATA_DIR / "X_test_lstm.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)


class JumpLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)


model = JumpLSTM(input_dim=X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 100
best_test_acc = 0
best_epoch = 0
patience = 25
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train_t.to(device)).argmax(1).cpu() == y_train_t).float().mean().item()
        test_acc = (model(X_test_t.to(device)).argmax(1).cpu() == y_test_t).float().mean().item()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), DATA_DIR / "best_model.pt")
        no_improve = 0
        print(f"  → Сохранена (Test: {test_acc:.3f})")
    else:
        no_improve += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | Best: {best_test_acc:.3f}")

    if no_improve >= patience:
        print(f"\nEarly stopping на эпохе {epoch+1}")
        break

print(f"\nЛучший Test Acc: {best_test_acc:.3f} (эпоха {best_epoch})")