# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями (если его нет, создайте)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код
COPY . .

# Открываем порт, который слушает ваше приложение (обычно 5000 или 8000)
EXPOSE 5000

# Команда для запуска (предполагая, что app.py — это Flask/FastAPI приложение)
CMD ["python", "app.py"]