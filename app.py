#!/usr/bin/env python3
"""app.py — Entrypoint do Boston Aerospace AI"""
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
    init_db()
    load_models()
    register_routes(app)
    @app.route("/")
    def index():
        return render_template("Index.html")
    return app

if __name__ == "__main__":
    app = create_app()
    try:
        from waitress import serve
        print("Boston Aerospace AI — http://localhost:5000")
        serve(app, host="0.0.0.0", port=5000, threads=4)
    except ImportError:
        app.run(host="0.0.0.0", port=5000, debug=False)
