"""
Api/Predict_Routes.py

Rotas:
- Predição RUL
- Chat Boston Aerospace AI
- Batch prediction
"""

from flask import (
    Blueprint,
    request,
    jsonify,
    session,
)

from Core.Predictor import predict_rul

from Core.AI_Orchestrator import (
    init_ai,
    ask as ai_ask,
)

from Core.Database import get_db

from Untils import (
    log_action,
    validate_sensors,
)

predict_bp = Blueprint(
    "predict",
    __name__
)


# ============================================================
# AUTENTICAÇÃO
# ============================================================

def auth_required(f):

    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):

        if not session.get(
            "user_id"
        ):

            return jsonify(
                {
                    "error": "Nao autenticado"
                }
            ), 401

        return f(
            *args,
            **kwargs
        )

    return wrapper


# ============================================================
# RUL PREDICTION
# ============================================================

@predict_bp.route(
    "/api/predict",
    methods=["POST"]
)
@auth_required
def predict():

    data = request.json or {}

    sensors = data.get(
        "sensors"
    )

    cycle = float(
        data.get(
            "cycle",
            150
        )
    )

    ok, message = validate_sensors(
        sensors
    )

    if not ok:

        return jsonify(
            {
                "error": message
            }
        ), 400

    try:

        result = predict_rul(
            sensors,
            cycle
        )

        with get_db() as conn:

            conn.execute(
                """
                INSERT INTO predictions
                (user_id, rul, status)
                VALUES (?, ?, ?)
                """,

                (
                    session["user_id"],

                    result["rul"],

                    result["status"],
                )
            )

        log_action(
            "predict",

            {
                "rul": result["rul"],

                "status": result["status"],
            }
        )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 500


# ============================================================
# CHAT OPENAI + RAG + WEB SEARCH
# ============================================================

@predict_bp.route(
    "/api/chat",
    methods=["POST"]
)
@auth_required
def chat():

    data = request.json or {}

    message = (
        data.get(
            "message",
            ""
        )
        .strip()
    )

    history = data.get(
        "history",
        []
    )

    ml_context = data.get(
        "prediction_context"
    )

    use_web_search = data.get(
        "web_search",
        True
    )

    if not message:

        return jsonify(
            {
                "error": "Mensagem vazia"
            }
        ), 400

    # Verificar OpenAI
    if not init_ai():

        return jsonify(
            {
                "error": (
                    "OpenAI não configurado. "
                    "Verifique OPENAI_API_KEY "
                    "e OPENAI_MODEL no Render."
                ),

                "offline": True,
            }
        ), 503

    try:

        result = ai_ask(

            question=message,

            history=history,

            ml_context=ml_context,

            use_web_search=use_web_search,
        )

        log_action(

            "chat",

            {
                "preview": message[:60]
            }
        )

        return jsonify(
            result
        )

    except Exception as error:

        error_text = str(
            error
        )

        print(
            f"[CHAT] Erro OpenAI: {error_text}"
        )

        if (

            "api_key"

            in error_text.lower()

            or "authentication"

            in error_text.lower()

            or "connection"

            in error_text.lower()

            or "401"

            in error_text

        ):

            return jsonify(
                {
                    "error": (
                        "OpenAI indisponível "
                        "ou chave API inválida."
                    ),

                    "offline": True,
                }
            ), 503

        return jsonify(
            {
                "error": error_text
            }
        ), 500


# ============================================================
# BATCH PREDICTION
# ============================================================

@predict_bp.route(
    "/api/batch",
    methods=["POST"]
)
@auth_required
def batch():

    import pandas as pd

    if "file" not in request.files:

        return jsonify(
            {
                "error": "Sem ficheiro"
            }
        ), 400

    try:

        dataframe = pd.read_csv(

            request.files["file"]
        )

    except Exception as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 400

    from Config.Settings import SENSOR_KEYS

    sensor_columns = [

        column

        for column in dataframe.columns

        if column.lower()

        in [

            sensor.lower()

            for sensor in SENSOR_KEYS

        ]

    ]

    if len(sensor_columns) < 5:

        return jsonify(
            {
                "error": (
                    "Poucas colunas de sensores. "
                    f"Colunas encontradas: "
                    f"{list(dataframe.columns)[:8]}"
                )
            }
        ), 400

    cycle_column = next(

        (

            column

            for column in dataframe.columns

            if "cycle"

            in column.lower()

        ),

        None
    )

    engine_column = next(

        (

            column

            for column in dataframe.columns

            if column.lower()

            in [

                "unit",

                "engine",

                "id",

                "motor",

            ]

        ),

        None
    )

    results = []

    for _, row in dataframe.iterrows():

        try:

            sensors = [

                float(

                    row.get(

                        sensor,

                        0
                    )
                )

                for sensor in SENSOR_KEYS

            ]

            cycle = (

                float(

                    row.get(

                        cycle_column,

                        150
                    )
                )

                if cycle_column

                else 150

            )

            engine_id = (

                str(

                    row.get(

                        engine_column,

                        "N/A"
                    )
                )

                if engine_column

                else "N/A"

            )

            result = predict_rul(

                sensors,

                cycle
            )

            results.append(

                {

                    "eid": engine_id,

                    "cycle": cycle,

                    **result,

                }

            )

        except Exception as error:

            results.append(

                {

                    "eid": "ERR",

                    "error": str(
                        error
                    ),

                }

            )

    log_action(

        "batch",

        {

            "rows": len(
                results
            )

        }

    )

    return jsonify(

        {

            "results": results,

            "total": len(
                results
            ),

            "critical": sum(

                1

                for result in results

                if result.get(
                    "status"
                )

                in [

                    "CRÍTICO",

                    "CRITICO",

                ]

            ),

            "alert": sum(

                1

                for result in results

                if result.get(
                    "status"
                )

                == "ALERTA"

            ),

        }

            )
