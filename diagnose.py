import json
from pathlib import Path
from collections import Counter

DATA_ROOT = Path(r"C:\КП\json")
SKIP_MARKERS = {"LAB_ORIGIN", "LAB_X", "LAB_Y", "LAB_Z", "Lab orientation"}

shapes = []
frame_counts = []
marker_counts = []

for skater_dir in sorted(DATA_ROOT.iterdir()):
    if not skater_dir.is_dir():
        continue
    for class_dir in sorted(skater_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for json_file in sorted(class_dir.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            markers = [m for m in data["Markers"] if m["Name"] not in SKIP_MARKERS]
            n_markers = len(markers)
            n_frames = len(markers[0]["Parts"][0]["Values"]) if markers else 0

            marker_counts.append(n_markers)
            frame_counts.append(n_frames)
            shapes.append((n_markers, n_frames))

print("Распределение по числу маркеров:", Counter(marker_counts))
print("Распределение по числу кадров:", Counter(frame_counts))
print()
print("Топ-5 самых частых форм (markers, frames):", Counter(shapes).most_common(5))
print()
print("Файлы с нестандартной формой:")
common = Counter(shapes).most_common(1)[0][0]
for skater_dir in sorted(DATA_ROOT.iterdir()):
    if not skater_dir.is_dir():
        continue
    for class_dir in sorted(skater_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for json_file in sorted(class_dir.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            markers = [m for m in data["Markers"] if m["Name"] not in SKIP_MARKERS]
            n_markers = len(markers)
            n_frames = len(markers[0]["Parts"][0]["Values"]) if markers else 0
            if (n_markers, n_frames) != common:
                print(f"  {json_file.relative_to(DATA_ROOT)}: ({n_markers}, {n_frames})")