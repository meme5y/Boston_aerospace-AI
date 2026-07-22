"""Api/Admin_Routes.py — Autenticacao, ping, me, audit"""
import json
import bcrypt
from flask import Blueprint, request, jsonify, session
from Core.Database import get_db, init_db
from Untils import log_action

admin_bp = Blueprint("admin", __name__)


def hash_pw(pw):  return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
def check_pw(pw, h): return bcrypt.checkpw(pw.encode(), h.encode() if isinstance(h,str) else h)


@admin_bp.route("/api/ping")
def ping():
    from Core.Predictor import META
    return jsonify({"ok": True, "mae": META.get("mae"), "r2": META.get("r2"), "src": META.get("src")})


@admin_bp.route("/api/llm-status")
@admin_bp.route("/api/ollama")  # mantido por compatibilidade com o frontend existente
def ollama_status():
    """Status do assistente IA. Para o lancamento do contest, so OpenAI e usado."""
    from Config.Settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBED_MODEL
    online = bool(OPENAI_API_KEY and OPENAI_MODEL)
    return jsonify({
        "provider": "openai",
        "online": online,
        "llm": OPENAI_MODEL,
        "embed": OPENAI_EMBED_MODEL,
        "has_llm": online,
        "has_embed": online,
    })


@admin_bp.route("/api/register", methods=["POST"])
def register():
    d     = request.json or {}
    email = d.get("email","").lower().strip()
    pw    = d.get("password","")
    name  = d.get("name","")
    if not email or not pw:
        return jsonify({"error": "Email e senha obrigatorios"}), 400
    if len(pw) < 4:
        return jsonify({"error": "Senha minima 4 caracteres"}), 400
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO users(email,pw_hash,name) VALUES(?,?,?)",
                         (email, hash_pw(pw), name))
        return jsonify({"ok": True}), 201
    except Exception:
        return jsonify({"error": "Email ja cadastrado"}), 409


@admin_bp.route("/api/login", methods=["POST"])
def login():
    d     = request.json or {}
    email = d.get("email","").lower().strip()
    pw    = d.get("password","")
    with get_db() as conn:
        row = conn.execute("SELECT id,pw_hash,name FROM users WHERE email=?", (email,)).fetchone()
    if row and check_pw(pw, row[1]):
        session["user_id"] = row[0]
        session["uname"]   = row[2] or email.split("@")[0]
        log_action("login")
        return jsonify({"ok": True, "name": session["uname"]})
    return jsonify({"error": "Email ou senha incorretos"}), 401


@admin_bp.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@admin_bp.route("/api/me")
def me():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"in": False})
    with get_db() as conn:
        preds = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (uid,)).fetchone()[0]
        docs  = conn.execute("SELECT COUNT(*) FROM kb_docs     WHERE user_id=?", (uid,)).fetchone()[0]
    return jsonify({"in": True, "name": session.get("uname",""), "preds": preds, "docs": docs})


@admin_bp.route("/api/audit")
def audit():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Nao autenticado"}), 401
    with get_db() as conn:
        rows = conn.execute(
            "SELECT action,details,ip,created_at FROM audit_log "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT 50", (uid,)
        ).fetchall()
    return jsonify({"log": [{"action":r[0],"details":json.loads(r[1] or "{}"),"ip":r[2],"time":str(r[3])} for r in rows]})
