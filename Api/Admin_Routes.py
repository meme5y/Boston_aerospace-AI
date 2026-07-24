"""
Api/Admin_Routes.py

Autenticação, status do sistema, utilizador e auditoria.
"""

import json
import bcrypt

from flask import (
    Blueprint,
    request,
    jsonify,
    session,
)

from Core.Database import (
    get_db,
)

from Untils import (
    log_action,
)


admin_bp = Blueprint(
    "admin",
    __name__
)


# ============================================================
# PASSWORD
# ============================================================

def hash_pw(
    password
):

    return bcrypt.hashpw(

        password.encode(),

        bcrypt.gensalt()

    ).decode()


def check_pw(
    password,

    hashed
):

    return bcrypt.checkpw(

        password.encode(),

        (

            hashed.encode()

            if isinstance(
                hashed,
                str
            )

            else hashed

        )

    )


# ============================================================
# PING
# ============================================================

@admin_bp.route(
    "/api/ping"
)
def ping():

    from Core.Predictor import META

    return jsonify(

        {

            "ok": True,

            "mae": META.get(
                "mae"
            ),

            "r2": META.get(
                "r2"
            ),

            "src": META.get(
                "src"
            ),

        }

    )


# ============================================================
# OPENAI STATUS
# ============================================================

@admin_bp.route(
    "/api/llm-status"
)
def llm_status():

    from Core.AI_Orchestrator import (
        get_status
    )

    return jsonify(
        get_status()
    )


# ============================================================
# REGISTER
# ============================================================

@admin_bp.route(
    "/api/register",
    methods=["POST"]
)
def register():

    data = request.json or {}

    email = (

        data.get(
            "email",
            ""
        )

        .lower()

        .strip()

    )

    password = data.get(
        "password",
        ""
    )

    name = data.get(
        "name",
        ""
    )

    if not email or not password:

        return jsonify(

            {

                "error": (
                    "Email e senha obrigatórios"
                )

            }

        ), 400

    if len(password) < 4:

        return jsonify(

            {

                "error": (
                    "Senha mínima de 4 caracteres"
                )

            }

        ), 400

    try:

        with get_db() as conn:

            conn.execute(

                """

                INSERT INTO users

                (email, pw_hash, name)

                VALUES (?, ?, ?)

                """,

                (

                    email,

                    hash_pw(
                        password
                    ),

                    name,

                )

            )

        return jsonify(

            {

                "ok": True

            }

        ), 201

    except Exception:

        return jsonify(

            {

                "error": (
                    "Email já cadastrado"
                )

            }

        ), 409


# ============================================================
# LOGIN
# ============================================================

@admin_bp.route(
    "/api/login",
    methods=["POST"]
)
def login():

    data = request.json or {}

    email = (

        data.get(
            "email",
            ""
        )

        .lower()

        .strip()

    )

    password = data.get(
        "password",
        ""
    )

    with get_db() as conn:

        row = conn.execute(

            """

            SELECT id,
                   pw_hash,
                   name

            FROM users

            WHERE email=?

            """,

            (

                email,

            )

        ).fetchone()

    if row and check_pw(

        password,

        row[1]

    ):

        session[
            "user_id"
        ] = row[0]

        session[
            "uname"
        ] = (

            row[2]

            or email.split(
                "@"
            )[0]

        )

        log_action(
            "login"
        )

        return jsonify(

            {

                "ok": True,

                "name": session[
                    "uname"
                ],

            }

        )

    return jsonify(

        {

            "error": (
                "Email ou senha incorretos"
            )

        }

    ), 401


# ============================================================
# LOGOUT
# ============================================================

@admin_bp.route(
    "/api/logout",
    methods=["POST"]
)
def logout():

    session.clear()

    return jsonify(

        {

            "ok": True

        }

    )


# ============================================================
# CURRENT USER
# ============================================================

@admin_bp.route(
    "/api/me"
)
def me():

    user_id = session.get(
        "user_id"
    )

    if not user_id:

        return jsonify(

            {

                "in": False

            }

        )

    with get_db() as conn:

        predictions = conn.execute(

            """

            SELECT COUNT(*)

            FROM predictions

            WHERE user_id=?

            """,

            (

                user_id,

            )

        ).fetchone()[0]

        documents = conn.execute(

            """

            SELECT COUNT(*)

            FROM kb_docs

            WHERE user_id=?

            """,

            (

                user_id,

            )

        ).fetchone()[0]

    return jsonify(

        {

            "in": True,

            "name": session.get(
                "uname",
                ""
            ),

            "preds": predictions,

            "docs": documents,

        }

    )


# ============================================================
# AUDIT
# ============================================================

@admin_bp.route(
    "/api/audit"
)
def audit():

    user_id = session.get(
        "user_id"
    )

    if not user_id:

        return jsonify(

            {

                "error": (
                    "Nao autenticado"
                )

            }

        ), 401

    with get_db() as conn:

        rows = conn.execute(

            """

            SELECT action,
                   details,
                   ip,
                   created_at

            FROM audit_log

            WHERE user_id=?

            ORDER BY created_at DESC

            LIMIT 50

            """,

            (

                user_id,

            )

        ).fetchall()

    return jsonify(

        {

            "log": [

                {

                    "action": row[0],

                    "details": json.loads(

                        row[1]

                        or "{}"

                    ),

                    "ip": row[2],

                    "time": str(
                        row[3]
                    ),

                }

                for row in rows

            ]

        }

                )
