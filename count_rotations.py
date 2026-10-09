import argparse
import json
import numpy as np


def moving_average(x, w=7):
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="same")[pad:-pad]


def load_pose_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "landmarks" in data:
        arr = np.array(data["landmarks"], dtype=np.float32)
        fps = data.get("fps", 30.0)
    else:
        arr = np.array(data, dtype=np.float32)
        fps = 30.0
    return arr, fps


def body_angle(pose, w=7):
    """Средний угол поворота корпуса в плоскости XZ (по бёдрам и плечам)."""
    hip = pose[:, 24, [0, 2]] - pose[:, 23, [0, 2]]
    sh = pose[:, 12, [0, 2]] - pose[:, 11, [0, 2]]

    a_hip = np.arctan2(hip[:, 1], hip[:, 0])
    a_sh = np.arctan2(sh[:, 1], sh[:, 0])

    a_hip = moving_average(np.unwrap(a_hip), w)
    a_sh = moving_average(np.unwrap(a_sh), w)

    # Медиана двух сигналов устойчивее к выбросам
    return a_sh


def find_air_phase(pose, fps, height_threshold=0.03, min_duration=0.25, expand=2):
    y_hip = (pose[:, 23, 1] + pose[:, 24, 1]) / 2.0
    y_hip = moving_average(y_hip, 7)

    baseline = np.median(y_hip)
    air = y_hip < (baseline - height_threshold)

    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(air):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            if i - start >= int(fps * min_duration):
                segments.append((start, i))
            in_seg = False
    if in_seg and len(air) - start >= int(fps * min_duration):
        segments.append((start, len(air)))

    # Расширяем окно на expand кадров в каждую сторону
    T = len(pose)
    expanded = []
    for s, e in segments:
        s2 = max(0, s - expand)
        e2 = min(T, e + expand)
        expanded.append((s2, e2))
    return expanded, baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", required=True)
    p.add_argument("--out", default="json/rotations.json")
    args = p.parse_args()

    pose, fps = load_pose_json(args.pose)
    angle = body_angle(pose)
    segments, baseline = find_air_phase(pose, fps)

    results = []
    for s, e in segments:
        delta = angle[e - 1] - angle[s]
        results.append({
            "start_sec": round(s / fps, 2),
            "end_sec": round(e / fps, 2),
            "rotations": round(abs(delta) / (2 * np.pi), 2),
        })

    total = round(sum(r["rotations"] for r in results), 2)
    out = {
        "fps": fps,
        "air_segments": [{"start": s, "end": e} for s, e in segments],
        "jumps": results,
        "total_rotations": total,
        "hint": "ближайшее целое = число оборотов (1=одинарный, 2=двойной, 3=тройной, 4=четверной)",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()