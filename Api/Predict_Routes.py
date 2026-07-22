"""Api/Predict_Routes.py — Rotas de predição RUL, Chat RAG e batch"""
import json
from flask import Blueprint, request, jsonify, session
from Core.Predictor import predict_rul
from Core.AI_Orchestrator import init_ai, ask as ai_ask
from Core.Database import get_db
from Untils import log_action, validate_sensors
from Config.Settings import N_SENSORS

predict_bp = Blueprint("predict", __name__)

def auth_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get("user_id"):
            return jsonify({"error": "Nao autenticado"}), 401
        return f(*a, **kw)
    return wrapper


@predict_bp.route("/api/predict", methods=["POST"])
@auth_required
def predict():
    d       = request.json or {}
    sensors = d.get("sensors")
    cycle   = float(d.get("cycle", 150))

    ok, msg = validate_sensors(sensors)
    if not ok:
        return jsonify({"error": msg}), 400

    try:
        result = predict_rul(sensors, cycle)
        with get_db() as conn:
            conn.execute(
                "INSERT INTO predictions(user_id,rul,status) VALUES(?,?,?)",
                (session["user_id"], result["rul"], result["status"])
            )
        log_action("predict", {"rul": result["rul"], "status": result["status"]})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@predict_bp.route("/api/chat", methods=["POST"])
@auth_required
def chat():
    d              = request.json or {}
    message        = d.get("message", "").strip()
    history        = d.get("history", [])
    ml_context     = d.get("prediction_context")   # opcional: ultimo resultado de predict_rul()
    use_web_search = d.get("web_search", True)      # ligado por padrao
    if not message:
        return jsonify({"error": "Mensagem vazia"}), 400
    if not init_ai():
        return jsonify({"error": "OpenAI nao configurado. Verifique OPENAI_API_KEY e OPENAI_MODEL no .env.",
                         "offline": True}), 503
    try:
        result = ai_ask(message, history=history, ml_context=ml_context, use_web_search=use_web_search)
        log_action("chat", {"preview": message[:60]})
        return jsonify(result)
    except Exception as e:
        err = str(e)
        if "Connection" in err or "refused" in err.lower() or "api_key" in err.lower():
            return jsonify({"error": "OpenAI indisponivel ou mal configurado.", "offline": True}), 503
        return jsonify({"error": err}), 500


@predict_bp.route("/api/batch", methods=["POST"])
@auth_required
def batch():
    import pandas as pd
    if "file" not in request.files:
        return jsonify({"error": "Sem ficheiro"}), 400
    try:
        df = pd.read_csv(request.files["file"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    from Config.Settings import SENSOR_KEYS
    sc = [c for c in df.columns if c.lower() in [s.lower() for s in SENSOR_KEYS]]
    if len(sc) < 5:
        return jsonify({"error": f"Colunas: {list(df.columns)[:8]}"}), 400

    cc = next((c for c in df.columns if "cycle" in c.lower()), None)
    ic = next((c for c in df.columns if c.lower() in ["unit","engine","id","motor"]), None)
    results = []
    for _, row in df.iterrows():
        try:
            sensors = [float(row.get(s, 0)) for s in SENSOR_KEYS]
            cycle   = float(row.get(cc, 150)) if cc else 150
            eid     = str(row.get(ic, "N/A")) if ic else "N/A"
            r       = predict_rul(sensors, cycle)
            results.append({"eid": eid, "cycle": cycle, **r})
        except Exception as e:
            results.append({"eid": "ERR", "error": str(e)})

    log_action("batch", {"rows": len(results)})
    return jsonify({
        "results":  results,
        "total":    len(results),
        "critical": sum(1 for r in results if r.get("status") == "CRITICO"),
        "alert":    sum(1 for r in results if r.get("status") == "ALERTA"),
    })
