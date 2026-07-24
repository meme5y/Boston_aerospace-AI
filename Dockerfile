FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    libsndfile1 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY . .

RUN mkdir -p \
    Modelos \
    Uploads \
    Logs \
    knowledge \
    Data/Raw \
    Data/Processed \
    Data/Synthetic

# Render detecta a porta através da variável PORT
EXPOSE 10000

CMD ["python", "app.py"]
