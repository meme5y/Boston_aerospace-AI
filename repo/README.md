# Boston Aerospace AI

> Predictive maintenance platform for aeronautical engines using NASA CMAPSS dataset.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)

## Features

- **RUL Prediction** — Ensemble of 6 ML models (XGBoost, LightGBM, CatBoost, RF, GBR, ET)
- **SHAP Explainability** — Per-prediction sensor importance analysis
- **RAG Chat** — Ollama + ChromaDB + LangChain for technical document Q&A
- **Computer Vision** — Crack detection, thermal analysis, audio vibration anomaly
- **PDF Reports** — Exportable maintenance reports with full prediction breakdown
- **Batch Prediction** — Fleet-level CSV processing

## Quick Start

```bash
# 1. Clone
git clone https://github.com/meme5y/Boston_aerospace-AI
cd Boston_aerospace-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull models
ollama serve
ollama pull tinyllama
ollama pull nomic-embed-text

# 4. Run
python app.py
# Open http://localhost:5000
```

## Docker

```bash
cp .env.example .env        # edit CLOUDFLARE_TUNNEL_TOKEN
docker-compose up -d
docker exec boston_ollama ollama pull tinyllama
docker exec boston_ollama ollama pull nomic-embed-text
```

## Repository Structure

```
Boston_aerospace-AI/
├── app.py                  # Flask entrypoint
├── Api/                    # Route blueprints
│   ├── Admin_Routes.py     # Auth, ping, audit
│   ├── Predict_Routes.py   # RUL, chat, batch
│   └── Upload_Routes.py    # Files, KB, CV, PDF
├── Config/Settings.py      # All configuration
├── Core/
│   ├── Predictor.py        # ML inference + SHAP
│   ├── Feature_Engineering.py
│   ├── Preprocess.py       # Data loading
│   ├── RAG.py              # Ollama + ChromaDB
│   ├── CV.py               # OpenCV + Librosa
│   └── Database.py         # SQLite helpers
├── Training/
│   ├── Train_model.py      # Model training
│   ├── Evaluate.py         # Metrics
│   └── Synthetic_data_gen.py
├── Untils/                 # Logger, validators, file helpers
├── Tests/                  # pytest test suite
├── Frontend/Templates/     # Dashboard HTML
└── Data/Raw/               # Place NASA CMAPSS files here
```

## NASA CMAPSS Data

Place `train_FD001.txt` and `train_FD002.txt` in `Data/Raw/`.  
Download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

Without the files, synthetic data is generated automatically.

## License

MIT — Fernando Artur Augusto (Arthur Wach), 2026
