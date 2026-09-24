import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = Path(r"C:\КП\dataset")
CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]


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


X_test = np.load(DATA_DIR / "X_test_lstm.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

device = torch.device("cpu")
model = JumpLSTM(input_dim=X_test.shape[2]).to(device)
model.load_state_dict(torch.load(DATA_DIR / "best_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred = logits.argmax(1).numpy()

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=CLASSES))