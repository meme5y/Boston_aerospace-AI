"""Core/Predictor.py — Motor de predição RUL e SHAP"""

import os
import json
import threading

import numpy as np
import joblib

from Config.Settings import (
    MODEL_DIR,
    MODEL_NAMES,
    META_FILE,
    SCALER_FILE,
    FCOLS_FILE,
)

from Core.Exceptions import ModelNotLoadedError


# Estado global dos modelos
MODELS: dict = {}
SCALER = None
FEAT_COLS: list = []
META: dict = {}
SHAP_EXP = None

# Controle do treinamento
TRAINING_IN_PROGRESS = False
TRAINING_LOCK = threading.Lock()


def models_exist() -> bool:
    """Verifica se todos os artefatos necessários existem."""

    required = [
        os.path.join(MODEL_DIR, f"{name}.pkl")
        for name in MODEL_NAMES
    ]

    required += [
        SCALER_FILE,
        FCOLS_FILE,
        META_FILE,
    ]

    return all(os.path.exists(path) for path in required)


def load_models() -> dict:
    """
    Carrega modelos já treinados do disco.

    IMPORTANTE:
    Esta função nunca inicia treinamento.
    Isso permite que o servidor Render abra a porta imediatamente.
    """

    global MODELS
    global SCALER
    global FEAT_COLS
    global META
    global SHAP_EXP

    if not models_exist():

        print(
            "[MODEL] Modelos ainda não existem. "
            "O treinamento será iniciado em background."
        )

        return MODELS

    try:

        MODELS = {
            name: joblib.load(
                os.path.join(MODEL_DIR, f"{name}.pkl")
            )
            for name in MODEL_NAMES
        }

        SCALER = joblib.load(SCALER_FILE)
        FEAT_COLS = joblib.load(FCOLS_FILE)

        with open(META_FILE, "r", encoding="utf-8") as f:
            META = json.load(f)

        print(
            f"[MODEL] Modelos carregados. "
            f"MAE={META.get('mae')} | "
            f"R2={META.get('r2')} | "
            f"Fonte={META.get('src')}"
        )

        # Inicializa SHAP
        try:

            import shap

            SHAP_EXP = shap.TreeExplainer(
                MODELS["xgb"]
            )

            print("[SHAP] Inicializado com sucesso")

        except Exception as e:

            print(
                f"[SHAP] Indisponível: {e}"
            )

    except Exception as e:

        print(
            f"[MODEL] Erro ao carregar modelos: {e}"
        )

        MODELS = {}

    return MODELS


def train_models_background():
    """
    Executa o treinamento em background.

    Quando termina, atualiza os modelos disponíveis
    sem precisar reiniciar o servidor.
    """

    global MODELS
    global SCALER
    global FEAT_COLS
    global META
    global SHAP_EXP
    global TRAINING_IN_PROGRESS

    with TRAINING_LOCK:

        if TRAINING_IN_PROGRESS:
            print(
                "[TRAIN] Treinamento já está em andamento."
            )

            return

        # Se os modelos já existem, não treinar novamente
        if models_exist():

            print(
                "[TRAIN] Modelos já existem. "
                "Treinamento não necessário."
            )

            load_models()

            return

        TRAINING_IN_PROGRESS = True

    try:

        print(
            "[TRAIN] Iniciando treinamento em background..."
        )

        from Training.Train_model import run_training

        models, scaler, feat_cols, meta = run_training()

        MODELS = models
        SCALER = scaler
        FEAT_COLS = feat_cols
        META = meta

        # Inicializa SHAP após o treinamento
        try:

            import shap

            SHAP_EXP = shap.TreeExplainer(
                MODELS["xgb"]
            )

            print(
                "[SHAP] Inicializado após treinamento"
            )

        except Exception as e:

            print(
                f"[SHAP] Não disponível após treinamento: {e}"
            )

        print(
            "[TRAIN] Treinamento concluído com sucesso."
        )

    except Exception as e:

        print(
            f"[TRAIN] Falha no treinamento: {e}"
        )

    finally:

        TRAINING_IN_PROGRESS = False


def predict_rul(
    sensors: list,
    cycle: float
) -> dict:

    """
    Executa a predição RUL.
    """

    if not MODELS:

        if TRAINING_IN_PROGRESS:

            raise ModelNotLoadedError(
                "Os modelos ainda estão sendo treinados. "
                "Tente novamente em alguns instantes."
            )

        raise ModelNotLoadedError(
            "Modelos não carregados."
        )

    from Core.Feature_Engineering import (
        build_inference_row
    )

    df = build_inference_row(
        sensors,
        cycle
    )

    vec = np.array(
        [
            df.iloc[-1][column]
            if column in df.columns
            else 0.0
            for column in FEAT_COLS
        ],
        dtype=float
    )

    x = SCALER.transform(
        vec.reshape(1, -1)
    )

    raw = {
        name: float(
            model.predict(x)[0]
        )
        for name, model in MODELS.items()
    }

    mean = float(
        np.mean(
            list(raw.values())
        )
    )

    std = float(
        np.std(
            list(raw.values())
        )
    )

    rul = max(
        0,
        int(
            round(mean)
        )
    )

    conf = max(
        0,
        min(
            100,
            int(
                100 -
                (
                    std /
                    max(mean, 1)
                ) *
                50
            )
        )
    )

    status, color, recommendation = get_status(rul)

    shap_top = []

    if SHAP_EXP:

        try:

            values = SHAP_EXP.shap_values(x)[0]

            top = sorted(
                zip(
                    FEAT_COLS,
                    map(abs, values)
                ),
                key=lambda item: item[1],
                reverse=True
            )[:10]

            shap_top = [
                {
                    "f": feature,
                    "v": round(
                        float(value),
                        4
                    )
                }
                for feature, value in top
            ]

        except Exception:

            pass

    return {

        "rul": rul,

        "lower": max(
            0,
            int(
                round(
                    mean -
                    1.96 *
                    std
                )
            )
        ),

        "upper": int(
            round(
                mean +
                1.96 *
                std
            )
        ),

        "confidence": conf,

        "std": round(
            std,
            1
        ),

        "status": status,

        "color": color,

        "recommendation": recommendation,

        "details": raw,

        "shap": shap_top,

    }


def get_status(
    rul: int
) -> tuple:

    if rul <= 15:

        return (
            "CRÍTICO",
            "#ff2244",
            "Manutenção imediata"
        )

    elif rul <= 40:

        return (
            "ALERTA",
            "#ff9500",
            "Agendar manutenção em 48h"
        )

    elif rul <= 80:

        return (
            "ATENÇÃO",
            "#ffdd00",
            "Monitorar de perto"
        )

    else:

        return (
            "NORMAL",
            "#00ff88",
            "Motor em boas condições"
            )
