# Используем официальный образ Python
FROM python:3.14-slim

# Устанавливаем рабочую директорию
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgthread-2.0-0 \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями (если его нет, создайте)
COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код
COPY . .

# Открываем порт, который слушает ваше приложение (обычно 5000 или 8000)
EXPOSE 8000

# Команда для запуска (предполагая, что app.py — это Flask/FastAPI приложение)
CMD ["python", "app.py"]