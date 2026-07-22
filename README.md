# Boston Aerospace AI

> Predictive maintenance platform for aeronautical engines using NASA CMAPSS dataset.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)

## Features

- **RUL Prediction** — Ensemble of 6 ML models (XGBoost, LightGBM, CatBoost, RF, GBR, ET)
- **SHAP Explainability** — Per-prediction sensor importance analysis
- **AI Maintenance Assistant** — OpenAI (GPT-5.6) orchestrator combining ML/SHAP results, an optional internal RAG index (ChromaDB + LangChain) of technical manuals, and live Web Search, returning a structured, source-attributed answer for engineers. Works with or without an indexed knowledge base.
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

# 3. Configure the AI assistant (OpenAI)
cp .env.example .env
# edit .env: set OPENAI_API_KEY and OPENAI_MODEL (e.g. gpt-5.6-terra)

# 4. Run
python app.py
# Open http://localhost:5000
```

> Optional local/offline mode (Ollama) is documented in `Core/RAG.py` and
> `.env.example` for future use with clients who need fully offline operation,
> but it isn't wired into the current routes — the app runs on OpenAI only.

## Docker

```bash
cp .env.example .env        # edit OPENAI_API_KEY, OPENAI_MODEL, CLOUDFLARE_TUNNEL_TOKEN
docker-compose up -d
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
│   ├── RAG.py              # [legacy] Ollama local mode, not wired to routes
│   ├── AI_Orchestrator.py  # OpenAI: ML/SHAP + internal RAG + Web Search, structured answers
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
