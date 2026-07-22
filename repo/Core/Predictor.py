"""Core/Predictor.py — Motor de predição RUL e SHAP"""
import os, json
import numpy as np
import joblib
from Config.Settings import MODEL_DIR, MODEL_NAMES, META_FILE, SCALER_FILE, FCOLS_FILE
from Core.Exceptions import ModelNotLoadedError


MODELS: dict  = {}
SCALER        = None
FEAT_COLS: list = []
META: dict    = {}
SHAP_EXP      = None


def load_models() -> dict:
    """Carrega modelos, scaler e feature cols do disco."""
    global MODELS, SCALER, FEAT_COLS, META, SHAP_EXP

    required = [os.path.join(MODEL_DIR, f"{n}.pkl") for n in MODEL_NAMES]
    if not all(os.path.exists(p) for p in required + [SCALER_FILE, FCOLS_FILE, META_FILE]):
        from Training.Train_model import run_training
        MODELS, SCALER, FEAT_COLS, META = run_training()
    else:
        MODELS   = {n: joblib.load(os.path.join(MODEL_DIR, f"{n}.pkl")) for n in MODEL_NAMES}
        SCALER   = joblib.load(SCALER_FILE)
        FEAT_COLS = joblib.load(FCOLS_FILE)
        META     = json.load(open(META_FILE))
        print(f"[OK] MAE={META.get('mae')} [{META.get('src')}]")

    try:
        import shap
        SHAP_EXP = shap.TreeExplainer(MODELS["xgb"])
        print("[SHAP] OK")
    except Exception as e:
        print(f"[SHAP] nao disponivel: {e}")

    return MODELS


def predict_rul(sensors: list, cycle: float) -> dict:
    """Executa predição RUL e devolve resultado completo."""
    if not MODELS:
        raise ModelNotLoadedError("Modelos não carregados. Chame load_models() primeiro.")

    from Core.Feature_Engineering import build_inference_row

    df  = build_inference_row(sensors, cycle)
    vec = np.array([df.iloc[-1][c] if c in df.columns else 0.0 for c in FEAT_COLS], dtype=float)
    x   = SCALER.transform(vec.reshape(1, -1))

    raw  = {n: float(m.predict(x)[0]) for n, m in MODELS.items()}
    mean = float(np.mean(list(raw.values())))
    std  = float(np.std(list(raw.values())))
    rul  = max(0, int(round(mean)))
    conf = max(0, min(100, int(100 - (std / max(mean, 1)) * 50)))

    st, color, rec = get_status(rul)

    shap_top = []
    if SHAP_EXP:
        try:
            vals = SHAP_EXP.shap_values(x)[0]
            top  = sorted(zip(FEAT_COLS, map(abs, vals)), key=lambda t: t[1], reverse=True)[:10]
            shap_top = [{"f": f, "v": round(float(v), 4)} for f, v in top]
        except Exception:
            pass

    return {
        "rul": rul,
        "lower": max(0, int(round(mean - 1.96 * std))),
        "upper": int(round(mean + 1.96 * std)),
        "confidence": conf,
        "std": round(std, 1),
        "status": st,
        "color": color,
        "recommendation": rec,
        "details": raw,
        "shap": shap_top,
    }


def get_status(rul: int) -> tuple:
    if   rul <= 15: return "CRÍTICO", "#ff2244", "Manutenção imediata"
    elif rul <= 40: return "ALERTA",  "#ff9500", "Agendar manutenção em 48h"
    elif rul <= 80: return "ATENÇÃO", "#ffdd00", "Monitorar de perto"
    else:           return "NORMAL",  "#00ff88", "Motor em boas condições"
