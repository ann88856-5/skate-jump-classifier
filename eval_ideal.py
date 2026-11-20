import numpy as np, torch, torch.nn as nn
from pathlib import Path

DATA_DIR = Path("dataset")
X = np.load(DATA_DIR / "X_ideal.npy")
y = np.load(DATA_DIR / "y_ideal.npy")

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

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
model.load_state_dict(torch.load(DATA_DIR / "best_model_ideal.pt"))
model.eval()

with torch.no_grad():
    logits = model(X_t)
    preds = logits.argmax(1)
    acc = (preds == y_t).float().mean().item()

print(f"=== ТЕСТ НА ИДЕАЛЬНЫХ ДАННЫХ (JSON) ===")
print(f"Точность (Accuracy): {acc * 100:.2f}%")