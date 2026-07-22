#!/usr/bin/env python3
"""app.py — Entrypoint do Boston Aerospace AI"""
import os
import sys
from flask import Flask, render_template, session
from flask_cors import CORS
from Config.Settings import SECRET_KEY, MAX_CONTENT_LEN
from Core.Database import init_db
from Core.Predictor import load_models
from Api import register_routes

def create_app() -> Flask:
    app = Flask(__name__, template_folder="Frontend/Templates",
                static_folder="Frontend/Statics")
    app.secret_key = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LEN
    CORS(app, supports_credentials=True)
    
    # Inicializar banco de dados e modelos
    init_db()
    load_models()
    register_routes(app)
    
    @app.route("/")
    def index():
        return render_template("Index.html")
    
    @app.route("/health")
    def health():
        return {"status": "healthy", "version": "1.0"}, 200
    
    return app

if __name__ == "__main__":
    app = create_app()
    
    # Usar a porta definida pelo Render (ou 5000 como fallback)
    port = int(os.environ.get("PORT", 5000))
    
    try:
        from waitress import serve
        print(f"Boston Aerospace AI — http://0.0.0.0:{port}")
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        print(f"Waitress não disponível. Usando Flask development server na porta {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
