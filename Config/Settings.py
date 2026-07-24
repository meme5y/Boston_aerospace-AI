import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.5"
).strip()

OPENAI_EMBED_MODEL = os.getenv(
    "OPENAI_EMBED_MODEL",
    "text-embedding-3-small"
).strip()

# ============================================================
# APPLICATION
# ============================================================

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key-in-production"
)

MAX_CONTENT_LEN = int(
    os.getenv(
        "MAX_CONTENT_LEN",
        str(50 * 1024 * 1024)
    )
)

# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR = BASE_DIR / "Modelos"

UPLOAD_DIR = BASE_DIR / "Uploads"

LOG_DIR = BASE_DIR / "Logs"

KNOWLEDGE_DIR = BASE_DIR / "knowledge"

DATA_DIR = BASE_DIR / "Data"

RAW_DATA_DIR = DATA_DIR / "Raw"

PROCESSED_DATA_DIR = DATA_DIR / "Processed"

SYNTHETIC_DATA_DIR = DATA_DIR / "Synthetic"

# ============================================================
# MODEL FILES
# ============================================================

MODEL_NAMES = [
    "xgb",
    "lgb",
    "cb",
]

META_FILE = MODEL_DIR / "meta.json"

SCALER_FILE = MODEL_DIR / "scaler.pkl"

FCOLS_FILE = MODEL_DIR / "feature_columns.pkl"

# ============================================================
# RAG
# ============================================================

CHROMA_DIR = KNOWLEDGE_DIR / "chroma"

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "boston_aerospace_knowledge"
)

# ============================================================
# MODEL STATUS
# ============================================================

def openai_configured() -> bool:
    return bool(
        OPENAI_API_KEY
        and OPENAI_API_KEY.startswith("sk-")
)
