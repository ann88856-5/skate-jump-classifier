import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

DATA_DIR = Path("dataset")
X = np.load(DATA_DIR / "X_ideal.npy")
y = np.load(DATA_DIR / "y_ideal.npy")

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

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
# ЗАГРУЖАЕМ ВЕСА (Fine-tuning)
model.load_state_dict(torch.load(DATA_DIR / "best_model.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
criterion = nn.CrossEntropyLoss()

EPOCHS = 30 
best_loss = float('inf')
patience = 5
no_improve = 0

print("Дообучение на идеальных данных (Fine-tuning)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
    
    # защита от переобучения
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        torch.save(model.state_dict(), DATA_DIR / "best_model_ideal.pt") # Сохраняем лучшую
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1} (Loss перестал падать).")
            break

print("Сохранено как best_model_ideal.pt")