"""Api/Routes.py — Registo de todas as blueprints"""
from flask import Blueprint
from .Predict_Routes import predict_bp
from .Upload_Routes  import upload_bp
from .Admin_Routes   import admin_bp

def register_routes(app):
    app.register_blueprint(predict_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(admin_bp)
