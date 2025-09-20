# Базовый образ с Python
FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости и ставим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Копируем модель внутрь образа
COPY model_checkpoint /app/model_checkpoint

# Переменные окружения
ENV PYTHONPATH=/app

# Открываем порт
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
