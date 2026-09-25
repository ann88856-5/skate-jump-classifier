import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil

app = FastAPI()

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


# Загружаем модель ОДИН РАЗ при старте
model = JumpLSTM(input_dim=411)
model.load_state_dict(torch.load(DATA_DIR / "best_model.pt", map_location="cpu"))
model.eval()
print("Модель загружена")


@app.get("/")
def root():
    return {"status": "ok", "message": "Skate Jump Classifier API"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Принимает видео, возвращает тип прыжка.
    
    Пока — заглушка: возвращает случайный класс.
    Следующий шаг — предобработка видео в X и вызов модели.
    """
    # Сохраняем файл
    save_path = DATA_DIR / "uploads" / file.filename
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # TODO: предобработка видео → X (300, 137, 3) → нормализация
    # Пока — заглушка
    X = np.random.randn(1, 300, 411).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(X))
        pred = logits.argmax(1).item()
        probs = torch.softmax(logits, dim=1).numpy()[0]

    return {
        "jump_type": CLASSES[pred],
        "confidence": float(probs[pred]),
        "all_probs": {CLASSES[i]: float(probs[i]) for i in range(6)}
    }