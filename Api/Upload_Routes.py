"""
Api/Upload_Routes.py

Upload de:
- Arquivos gerais
- Documentos da Knowledge Base
- Imagens para detecção de trincas
- Imagens térmicas
- Áudio
- Exportação PDF
"""

import io
import os

from flask import (
    Blueprint,
    request,
    jsonify,
    session,
    send_file,
)

from Core.Database import get_db

from Core.AI_Orchestrator import (
    init_ai,
    add_document,
)

from Untils import (
    log_action,
    save_upload,
)


upload_bp = Blueprint(
    "upload",
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
# UPLOAD GERAL
# ============================================================

@upload_bp.route(
    "/api/upload",
    methods=["POST"]
)
@auth_required
def upload():

    if "file" not in request.files:

        return jsonify(
            {
                "error": "Sem ficheiro"
            }
        ), 400

    file = request.files[
        "file"
    ]

    save_upload(
        file
    )

    log_action(
        "upload",

        {
            "file": file.filename
        }
    )

    return jsonify(

        {

            "ok": True,

            "msg": (
                f'"{file.filename}" '
                "enviado."
            ),

        }

    )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

@upload_bp.route(
    "/api/kb-upload",
    methods=["POST"]
)
@auth_required
def kb_upload():

    if "file" not in request.files:

        return jsonify(
            {
                "error": "Sem ficheiro"
            }
        ), 400

    file = request.files[
        "file"
    ]

    filename = file.filename

    extension = (

        filename.rsplit(
            ".",
            1
        )[-1]

        .lower()

        if "."

        in filename

        else "txt"

    )

    allowed = [

        "pdf",

        "txt",

        "md",

        "csv",

    ]

    if extension not in allowed:

        return jsonify(

            {

                "error": (
                    "Use PDF, TXT, MD ou CSV"
                )

            }

        ), 400

    # OpenAI deve estar configurado
    if not init_ai():

        return jsonify(

            {

                "error": (
                    "OpenAI não configurado. "
                    "Adicione OPENAI_API_KEY "
                    "no Render."
                ),

                "offline": True,

            }

        ), 503

    try:

        path = save_upload(
            file
        )

        result = add_document(

            path,

            filename

        )

        if result.get(
            "success"
        ):

            with get_db() as conn:

                conn.execute(

                    """

                    INSERT INTO kb_docs

                    (user_id, filename, chunks)

                    VALUES (?, ?, ?)

                    """,

                    (

                        session[
                            "user_id"
                        ],

                        filename,

                        result.get(
                            "chunks",
                            0
                        ),

                    )

                )

            log_action(

                "kb_upload",

                {

                    "file": filename

                }

            )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify(

            {

                "error": str(
                    error
                )

            }

        ), 500


# ============================================================
# LISTA DE DOCUMENTOS
# ============================================================

@upload_bp.route(
    "/api/kb-list"
)
@auth_required
def kb_list():

    with get_db() as conn:

        rows = conn.execute(

            """

            SELECT filename,
                   chunks,
                   created_at

            FROM kb_docs

            WHERE user_id=?

            ORDER BY created_at DESC

            """,

            (

                session[
                    "user_id"
                ],

            )

        ).fetchall()

    return jsonify(

        {

            "docs": [

                {

                    "name": row[0],

                    "chunks": row[1],

                    "date": str(
                        row[2]
                    )[:10],

                }

                for row in rows

            ]

        }

    )


# ============================================================
# DETECÇÃO DE TRINCAS
# ============================================================

@upload_bp.route(
    "/api/crack",
    methods=["POST"]
)
@auth_required
def crack():

    from Core.CV import (
        detect_cracks
    )

    if "image" not in request.files:

        return jsonify(

            {

                "error": (
                    "Imagem não enviada"
                )

            }

        ), 400

    try:

        file = request.files[
            "image"
        ]

        result = detect_cracks(
            file
        )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify(

            {

                "error": str(
                    error
                )

            }

        ), 500


# ============================================================
# ANÁLISE TÉRMICA
# ============================================================

@upload_bp.route(
    "/api/thermal",
    methods=["POST"]
)
@auth_required
def thermal():

    from Core.CV import (
        analyze_thermal
    )

    if "image" not in request.files:

        return jsonify(

            {

                "error": (
                    "Imagem não enviada"
                )

            }

        ), 400

    try:

        file = request.files[
            "image"
        ]

        result = analyze_thermal(
            file
        )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify(

            {

                "error": str(
                    error
                )

            }

        ), 500


# ============================================================
# ÁUDIO
# ============================================================

@upload_bp.route(
    "/api/audio",
    methods=["POST"]
)
@auth_required
def audio():

    from Core.Audio import (
        analyze_audio
    )

    if "audio" not in request.files:

        return jsonify(

            {

                "error": (
                    "Áudio não enviado"
                )

            }

        ), 400

    try:

        file = request.files[
            "audio"
        ]

        result = analyze_audio(
            file
        )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify(

            {

                "error": str(
                    error
                )

            }

        ), 500


# ============================================================
# EXPORTAÇÃO PDF
# ============================================================

@upload_bp.route(
    "/api/export-pdf",
    methods=["POST"]
)
@auth_required
def export_pdf():

    from Core.PDF import (
        create_report
    )

    data = request.json or {}

    try:

        pdf_bytes = create_report(
            data
        )

        return send_file(

            io.BytesIO(
                pdf_bytes
            ),

            mimetype="application/pdf",

            as_attachment=True,

            download_name=(
                "boston_aerospace_report.pdf"
            ),

        )

    except Exception as error:

        return jsonify(

            {

                "error": str(
                    error
                )

            }

        ), 500
