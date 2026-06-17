FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq5 \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências directamente (sem precisar de requirements.txt)
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    waitress==3.0.0 \
    bcrypt==4.1.3 \
    psycopg2-binary==2.9.9 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.5 \
    joblib==1.4.2 \
    shap==0.45.1 \
    langchain==0.2.6 \
    langchain-community==0.2.6 \
    langchain-ollama==0.1.1 \
    langchain-text-splitters==0.2.2 \
    chromadb==0.5.3 \
    pypdf==4.2.0 \
    reportlab==4.2.2 \
    opencv-python-headless==4.10.0.84 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    scipy==1.13.1 \
    requests==2.32.3

COPY . .

RUN mkdir -p Modelos Uploads Logs knowledge Data/Raw Data/Processed Data/Synthetic

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/ping')"

CMD ["python", "app.py"]
