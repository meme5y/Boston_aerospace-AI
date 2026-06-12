"""Untils/Logger.py — Sistema de logging"""
import logging, os
from Config.Settings import LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BostonAI")

def log_action(action: str, details: dict = None):
    from flask import session, request
    import json, sqlite3
    from Config.Settings import DB_PATH
    try:
        uid   = session.get("user_id", 0)
        ip    = request.remote_addr or ""
        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "INSERT INTO audit_log(user_id,action,details,ip) VALUES(?,?,?,?)",
                (uid, action, json.dumps(details or {}), ip)
            )
    except Exception as e:
        logger.warning(f"log_action failed: {e}")
