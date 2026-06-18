FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt de dentro da pasta repo/
COPY repo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o conteúdo da pasta repo/ para /app/repo/
COPY repo/ repo/

# Entrar na pasta repo/ para executar o app.py
WORKDIR /app/repo

EXPOSE 5000

CMD ["python", "app.py"]
