import json
import numpy as np
from pathlib import Path
from collections import Counter

DATA_ROOT = Path(r"C:\КП\json")
OUTPUT_DIR = Path(r"C:\КП\dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SKIP_MARKERS = {"LAB_ORIGIN", "LAB_X", "LAB_Y", "LAB_Z", "Lab orientation"}

TARGET_FRAMES = 300
MIN_FRAMES = 100


def load_markers(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    markers = [m for m in data["Markers"] if m["Name"] not in SKIP_MARKERS]
    names = [m["Name"] for m in markers]

    # Минимальная длина среди маркеров внутри файла
    lengths = [len(m["Parts"][0]["Values"]) for m in markers]
    min_len = min(lengths)

    positions = []
    for m in markers:
        values = m["Parts"][0]["Values"]
        arr = np.array(values, dtype=np.float32)
        arr = arr[:min_len, :3]  # обрезаем до min_len
        positions.append(arr)

    positions = np.stack(positions, axis=1)  # (min_len, n_markers, 3)
    return names, positions


def fix_frames(arr, target=TARGET_FRAMES):
    n = arr.shape[0]
    if n == target:
        return arr
    if n > target:
        start = (n - target) // 2
        return arr[start:start + target]
    pad = np.repeat(arr[-1:], target - n, axis=0)
    return np.concatenate([arr, pad], axis=0)


def build_dataset():
    X_list, y_list, meta = [], [], []
    skipped_short, skipped_bad, skipped_comb = 0, 0, 0
    ref_names = None

    for skater_dir in sorted(DATA_ROOT.iterdir()):
        if not skater_dir.is_dir():
            continue
        for class_dir in sorted(skater_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cls = class_dir.name
            if cls == "Comb":
                skipped_comb += 1
                continue
            if cls not in CLASS_TO_IDX:
                continue

            for json_file in sorted(class_dir.glob("*.json")):
                try:
                    names, positions = load_markers(json_file)
                except Exception as e:
                    print(f"  Ошибка {json_file.name}: {e}")
                    skipped_bad += 1
                    continue

                n_frames = positions.shape[0]
                if n_frames < MIN_FRAMES:
                    print(f"  Пропуск (мало кадров {n_frames}): {json_file.name}")
                    skipped_short += 1
                    continue

                if ref_names is None:
                    ref_names = names
                    print(f"Эталон: {len(ref_names)} маркеров")
                elif names != ref_names:
                    name_to_idx = {n: i for i, n in enumerate(names)}
                    if not all(n in name_to_idx for n in ref_names):
                        print(f"  Пропуск (не хватает маркеров): {json_file.name}")
                        skipped_bad += 1
                        continue
                    idx = [name_to_idx[n] for n in ref_names]
                    positions = positions[:, idx, :]

                positions = fix_frames(positions, TARGET_FRAMES)

                X_list.append(positions)
                y_list.append(CLASS_TO_IDX[cls])
                meta.append((skater_dir.name, cls, json_file.name))

    X = np.stack(X_list)
    y = np.array(y_list)

    print(f"\nГотово: X = {X.shape}, y = {y.shape}")
    print(f"Пропущено (Comb): {skipped_comb}")
    print(f"Пропущено (мало кадров): {skipped_short}")
    print(f"Пропущено (битые): {skipped_bad}")
    print(f"Распределение классов: {Counter(y.tolist())}")

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    with open(OUTPUT_DIR / "markers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ref_names))
    with open(OUTPUT_DIR / "meta.txt", "w", encoding="utf-8") as f:
        for skater, cls, fname in meta:
            f.write(f"{skater}\t{cls}\t{fname}\n")

    print(f"Сохранено в {OUTPUT_DIR}")


if __name__ == "__main__":
    build_dataset()