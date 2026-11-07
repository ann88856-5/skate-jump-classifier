import shutil
from pathlib import Path

# ПУТЬ К ПАПКЕ С ЗАГРУЗКАМИ
SOURCE_DIR = Path(r"-") 

TARGET_DIR = Path(r"C:\KP\videos")

CLASSES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]

def organize_videos():
    count = 0
    for cls in CLASSES:
        (TARGET_DIR / cls).mkdir(parents=True, exist_ok=True)

    for video_path in SOURCE_DIR.rglob("*.mp4"):
        filename = video_path.name.lower()

        if "comb" in filename:
            continue

        # Ищем название прыжка в имени файла
        jump_type = None
        for cls in CLASSES:
            if cls.lower() in filename:
                jump_type = cls
                break

        if not jump_type:
            continue

        count += 1
        new_name = f"{jump_type}_{count:04d}.mp4"
        target_path = TARGET_DIR / jump_type / new_name


        shutil.copy2(video_path, target_path)
        print(f"Скопировано: {jump_type}/{new_name}")

    print(f"\nПОБЕДА Всего скопировано видео: {count}")

if __name__ == "__main__":
    organize_videos()