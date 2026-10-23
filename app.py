import numpy as np
from pdf_utils import create_analysis_pdf
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
import shutil
import psycopg
import io

app = FastAPI()

DATA_DIR = Path(r"C:\КП\dataset")
CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]

DB_CONFIG = {
    "dbname": "skate_analysis",
    "user": "postgres",
    "password": "qazwsx89",
    "host": "localhost",
    "port": 5432,
    "client_encoding": "UTF8",
}

# Индексы маркеров для подсчёта оборотов
with open(DATA_DIR / "markers.txt", encoding="utf-8") as f:
    MARKERS = f.read().splitlines()
IDX_L_HIP = MARKERS.index("L_Hip Joint")
IDX_R_HIP = MARKERS.index("R_Hip Joint")


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


# Загружаем модель один раз
model = JumpLSTM(input_dim=411)
model.load_state_dict(torch.load(DATA_DIR / "best_model.pt", map_location="cpu"))
model.eval()
print("Модель загружена")


def count_rotations(pose_seq):
    """Считает обороты по углу линии бёдер в плоскости XZ.
    
    pose_seq: (300, 137, 3)
    Возвращает число оборотов.
    """
    l_hip = pose_seq[:, IDX_L_HIP, :]
    r_hip = pose_seq[:, IDX_R_HIP, :]
    vec = r_hip - l_hip  # (300, 3)
    angle = np.arctan2(vec[:, 2], vec[:, 0])
    angle_unwrapped = np.unwrap(angle)
    rotations = (angle_unwrapped[-1] - angle_unwrapped[0]) / (2 * np.pi)
    return float(abs(rotations))


@app.get("/")
def root():
    return {"status": "ok", "message": "Skate Jump Classifier API"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Принимает видео, возвращает тип прыжка и число оборотов."""
    # Сохраняем файл
    save_path = DATA_DIR / "uploads" / file.filename
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Заглушка: видео → 3D-позы (Спринт 3)
    # Пока берём случайный пример из тестового набора
    X_test_raw = np.load(DATA_DIR / "X_test.npy")  # (N, 300, 137, 3)
    idx = np.random.randint(0, len(X_test_raw))
    pose_seq = X_test_raw[idx]  # (300, 137, 3)

    # Подсчёт оборотов
    rotations = count_rotations(pose_seq)

    # Сплющиваем для модели
    X = pose_seq.reshape(1, 300, -1).astype(np.float32)  # (1, 300, 411)

    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(probs.argmax())

    # Сохраняем в PostgreSQL
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO sessions (filename, jump_type, confidence, rotations)
        VALUES (%s, %s, %s, %s) RETURNING id""",
        (file.filename, CLASSES[pred], float(probs[pred]), rotations)
    )
    session_id = cur.fetchone()[0]

    for i, cls in enumerate(CLASSES):
        cur.execute(
            """INSERT INTO session_probs (session_id, class_name, probability)
            VALUES (%s, %s, %s)""",
            (session_id, cls, float(probs[i]))
        )

    conn.commit()
    cur.close()
    conn.close()

    return {
        "session_id": session_id,
        "jump_type": CLASSES[pred],
        "confidence": float(probs[pred]),
        "rotations": rotations,
        "all_probs": {CLASSES[i]: float(probs[i]) for i in range(6)},
    }

@app.get("/report/{session_id}")
async def download_report(session_id: int):
    """Генерирует и возвращает PDF-отчёт по ID сессии из PostgreSQL."""
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # 1. Получаем основную информацию о сессии
        cur.execute(
            """SELECT filename, jump_type, confidence, rotations
               FROM sessions WHERE id = %s""",
            (session_id,)
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        filename, jump_type, confidence, rotations = row

        # 2. Получаем вероятности по всем классам
        cur.execute(
            """SELECT class_name, probability
               FROM session_probs WHERE session_id = %s
               ORDER BY probability DESC""",
            (session_id,)
        )
        prob_rows = cur.fetchall()
        all_probs = {cls: float(prob) for cls, prob in prob_rows}

    finally:
        cur.close()
        conn.close()

    # 3. Генерируем PDF
    pdf_buffer = create_analysis_pdf(
        session_id=session_id,
        filename=filename,
        jump_type=jump_type,
        confidence=float(confidence),
        rotations=float(rotations),
        all_probs=all_probs,
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"}
    )