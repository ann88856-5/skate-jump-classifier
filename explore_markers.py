import json

path = r"C:\КП\json\Skater_A\Axel\Axel_1.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

# TraceRange и Timebase
print("TraceRange:", data["TraceRange"])
print("Timebase:", data["Timebase"])
print()

# Первый маркер
marker = data["Markers"][0]
print("Тип маркера:", type(marker))
print("Ключи маркера:", list(marker.keys()) if isinstance(marker, dict) else "—")
print()

# Полностью первый маркер (без больших массивов)
if isinstance(marker, dict):
    for k, v in marker.items():
        if isinstance(v, list):
            print(f"  {k}: list, длина = {len(v)}, пример = {v[:3]}")
        elif isinstance(v, dict):
            print(f"  {k}: dict, ключи = {list(v.keys())}")
        else:
            print(f"  {k}: {v}")

print()
print("Всего маркеров:", len(data["Markers"]))

# Имена всех маркеров (первые 30)
names = [m.get("Name", "?") for m in data["Markers"][:30]]
print("Первые 30 имён:", names)