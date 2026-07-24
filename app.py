#!/usr/bin/env python3
"""app.py — Entrypoint do Boston Aerospace AI"""

import os
import threading
from flask import Flask, render_template
from flask_cors import CORS

from Config.Settings import SECRET_KEY, MAX_CONTENT_LEN
from Core.Database import init_db
from Core.Predictor import load_models, train_models_background
from Api import register_routes


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="Frontend/Templates",
        static_folder="Frontend/Statics"
    )

    app.secret_key = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LEN

    CORS(app, supports_credentials=True)

    # Inicializa banco de dados rapidamente
    init_db()

    # Carrega modelos existentes sem bloquear o servidor
    load_models()

    # Registra as rotas
    register_routes(app)

    @app.route("/")
    def index():
        return render_template("Index.html")

    return app


# Cria a aplicação imediatamente
app = create_app()


def start_background_training():
    """
    Treina os modelos em segundo plano caso ainda não existam.
    O servidor web permanece disponível durante o treinamento.
    """
    try:
        train_models_background()
    except Exception as e:
        print(f"[TRAIN] Erro no treinamento em background: {e}")


# Se executado diretamente
if __name__ == "__main__":

    # Se os modelos ainda não existirem, treina em background
    training_thread = threading.Thread(
        target=start_background_training,
        daemon=True,
        name="model-training"
    )

    training_thread.start()

    # Render fornece a porta através da variável PORT
    port = int(os.environ.get("PORT", 5000))

    print(f"[SERVER] Boston Aerospace AI iniciando em 0.0.0.0:{port}")

    try:
        from waitress import serve

        serve(
            app,
            host="0.0.0.0",
            port=port,
            threads=4
        )

    except ImportError:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False
    )
