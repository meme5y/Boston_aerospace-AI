"""Config/Settings.py — Configuração global"""
import os

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE, "Modelos");    os.makedirs(MODEL_DIR,  exist_ok=True)
UPLOAD_DIR = os.path.join(BASE, "Uploads");    os.makedirs(UPLOAD_DIR, exist_ok=True)
DATA_DIR   = os.path.join(BASE, "Data", "Raw"); os.makedirs(DATA_DIR,  exist_ok=True)
CHROMA_DIR = os.path.join(BASE, "knowledge");  os.makedirs(CHROMA_DIR, exist_ok=True)
LOG_DIR    = os.path.join(BASE, "Logs");        os.makedirs(LOG_DIR,   exist_ok=True)

DB_PATH         = os.path.join(BASE, "boston.db")
LOG_FILE        = os.path.join(LOG_DIR, "App.log")
SECRET_KEY      = os.environ.get("SECRET_KEY", "boston_aerospace_v3_2026")
MAX_CONTENT_LEN = 200 * 1024 * 1024

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "nomic-embed-text")

RUL_MAX     = 125
SENSOR_KEYS = ["s1","s2","s3","s4","s5","s6","s7","s8","s9",
               "s11","s12","s13","s14","s17","s18","s19","s20","s21"]
N_SENSORS   = len(SENSOR_KEYS)
CMAPSS_COLS = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
MODEL_NAMES = ["xgb","lgb","cb","rf","gbr","et"]

META_FILE   = os.path.join(MODEL_DIR, "meta.json")
SCALER_FILE = os.path.join(MODEL_DIR, "Scaler.pkl")
FCOLS_FILE  = os.path.join(MODEL_DIR, "Feature_cols.pkl")

SYSTEM_PROMPT = (
    "Voce e o assistente tecnico do Boston Aerospace AI. "
    "Especialidades: manutencao preditiva de motores aeronauticos, NASA CMAPSS, "
    "RUL (Remaining Useful Life), ensemble de 6 modelos ML, protocolos PHM. "
    "Responda SEMPRE em portugues. Seja tecnico e conciso."
)
