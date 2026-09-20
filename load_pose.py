import json
import numpy as np

path = r"C:\КП\json\Skater_A\Axel\Axel_1.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("Тип данных:", type(data))
print("Ключи:", list(data.keys()) if isinstance(data, dict) else "—")

# Если это словарь — смотрим, что внутри
if isinstance(data, dict):
    for key in data:
        value = data[key]
        if isinstance(value, (list, dict)):
            print(f"  {key}: {type(value).__name__}, длина = {len(value)}")
        else:
            print(f"  {key}: {value}")