import json
import numpy as np
from pathlib import Path

DATA_ROOT = Path(r"json")
OUTPUT_DIR = Path(r"dataset")
CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SKIP = {"LAB_ORIGIN", "LAB_X", "LAB_Y", "LAB_Z", "Lab orientation"}
TARGET_FRAMES = 300

def build():
    X_list, y_list = [], []
    for class_dir in DATA_ROOT.rglob("*"):
        if class_dir.is_dir() and class_dir.name in CLASS_TO_IDX:
            for json_file in class_dir.glob("*.json"):
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                markers = [m for m in data["Markers"] if m["Name"] not in SKIP]
                if len(markers) < 33: continue
                
                min_len = min(len(m["Parts"][0]["Values"]) for m in markers)
                pos = np.stack([np.array(m["Parts"][0]["Values"], dtype=np.float32)[:min_len, :3] for m in markers[:33]], axis=1)
                if pos.shape[0] < 100: continue
                
                # Нормализация по тазу
                pos = pos - np.expand_dims((pos[:, 23, :] + pos[:, 24, :]) / 2.0, axis=1)
                
                # Фикс кадров до 300
                if pos.shape[0] > TARGET_FRAMES:
                    pos = pos[(pos.shape[0]-TARGET_FRAMES)//2 : (pos.shape[0]-TARGET_FRAMES)//2 + TARGET_FRAMES]
                else:
                    pos = np.concatenate([pos, np.repeat(pos[-1:], TARGET_FRAMES - pos.shape[0], axis=0)], axis=0)
                
                X_list.append(pos)
                y_list.append(CLASS_TO_IDX[class_dir.name])
                
    X = np.stack(X_list).reshape(len(X_list), 300, -1)
    y = np.array(y_list)
    print(f"Идеальный датасет: X = {X.shape}, y = {y.shape}")
    np.save(OUTPUT_DIR / "X_ideal.npy", X)
    np.save(OUTPUT_DIR / "y_ideal.npy", y)

if __name__ == "__main__":
    build()