FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema (corrigido para Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libpq5 \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY repo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY repo/ repo/

EXPOSE 5000

CMD ["python", "repo/app.py"]
