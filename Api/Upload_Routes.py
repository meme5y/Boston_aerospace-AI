"""Api/Upload_Routes.py — Upload geral, knowledge base e PDF export"""
import io, os
from flask import Blueprint, request, jsonify, session, send_file
from Core.Database import get_db
from Core.AI_Orchestrator import init_ai, add_document
from Untils import log_action, save_upload

upload_bp = Blueprint("upload", __name__)

def auth_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get("user_id"):
            return jsonify({"error": "Nao autenticado"}), 401
        return f(*a, **kw)
    return wrapper


@upload_bp.route("/api/upload", methods=["POST"])
@auth_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Sem ficheiro"}), 400
    f     = request.files["file"]
    path  = save_upload(f)
    return jsonify({"ok": True, "msg": f'"{f.filename}" enviado.'})


@upload_bp.route("/api/kb-upload", methods=["POST"])
@auth_required
def kb_upload():
    if "file" not in request.files:
        return jsonify({"error": "Sem ficheiro"}), 400
    f     = request.files["file"]
    fname = f.filename
    ext   = fname.rsplit(".", 1)[-1].lower() if "." in fname else "txt"
    if ext not in ["pdf", "txt", "md", "csv"]:
        return jsonify({"error": "Use PDF, TXT, MD ou CSV"}), 400
    if not init_ai():
        return jsonify({"error": "OpenAI nao configurado. Verifique OPENAI_API_KEY e OPENAI_MODEL no .env.",
                         "offline": True}), 503
    path = save_upload(f)
    res  = add_document(path, fname)
    if res.get("success"):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO kb_docs(user_id,filename,chunks) VALUES(?,?,?)",
                (session["user_id"], fname, res["chunks"])
            )
        log_action("kb_upload", {"file": fname})
    return jsonify(res)


@upload_bp.route("/api/kb-list")
@auth_required
def kb_list():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT filename,chunks,created_at FROM kb_docs WHERE user_id=? ORDER BY created_at DESC",
            (session["user_id"],)
        ).fetchall()
    return jsonify({"docs": [{"name": r[0], "chunks": r[1], "date": str(r[2])[:10]} for r in rows]})


@upload_bp.route("/api/crack", methods=["POST"])
@auth_required
def crack():
    from Core.CV import detect_cracks
    if "image" not in request.files:
        return jsonify({"error": "Sem imagem"}), 400
    path = save_upload(request.files["image"])
    return jsonify(detect_cracks(path))


@upload_bp.route("/api/thermal", methods=["POST"])
@auth_required
def thermal():
    from Core.CV import analyze_thermal
    if "image" not in request.files:
        return jsonify({"error": "Sem imagem"}), 400
    path = save_upload(request.files["image"])
    return jsonify(analyze_thermal(path))


@upload_bp.route("/api/audio", methods=["POST"])
@auth_required
def audio():
    from Core.CV import analyze_audio
    if "audio" not in request.files:
        return jsonify({"error": "Sem audio"}), 400
    path = save_upload(request.files["audio"])
    return jsonify(analyze_audio(path))


@upload_bp.route("/api/export-pdf", methods=["POST"])
@auth_required
def export_pdf():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    except ImportError:
        return jsonify({"error": "pip install reportlab"}), 503

    from datetime import datetime
    d      = request.json or {}
    rul    = d.get("rul", 0)
    lo     = d.get("lower", 0)
    hi     = d.get("hi",    0)
    conf   = d.get("confidence", 0)
    st     = d.get("status", "N/A")
    rec    = d.get("recommendation", "")
    color  = d.get("color", "#888888")
    eid    = d.get("eid", "N/A")
    cycle  = d.get("cycle", 0)
    details= d.get("details", {})

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
    TS = ParagraphStyle("t", parent=styles["Title"], fontSize=18,
                        textColor=colors.HexColor("#00d8ff"), alignment=TA_CENTER, spaceAfter=4)
    SS = ParagraphStyle("s", parent=styles["Normal"], fontSize=9,
                        textColor=colors.HexColor("#3a506b"), alignment=TA_CENTER, spaceAfter=2)
    story += [
        Paragraph("RELATORIO DE ANALISE PREDITIVA", TS),
        Paragraph("Boston Aerospace AI — NASA CMAPSS", SS),
        Paragraph(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por {session.get('uname','N/A')}", SS),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#00d8ff"), spaceAfter=12),
        Paragraph(f"<b>Motor:</b> {eid}   <b>Ciclos:</b> {cycle}",
                  ParagraphStyle("i", parent=styles["Normal"], fontSize=10, spaceAfter=4)),
        Spacer(1, 0.3*cm),
    ]
    sc = colors.HexColor(color)
    td = [["RUL Estimado","Intervalo 95%","Confianca","Status"],
          [f"{rul} ciclos", f"[{lo} - {hi}]", f"{conf}%", st]]
    t  = Table(td, colWidths=[4*cm,5*cm,3.5*cm,4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0b1020")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#00d8ff")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),9),
        ("FONTSIZE",(0,1),(-1,1),13),("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
        ("TEXTCOLOR",(3,1),(3,1),sc),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor("#00d8ff")),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    story += [t, Spacer(1, 0.4*cm),
              Paragraph(f"<b>Recomendacao:</b> {rec}",
                        ParagraphStyle("r", parent=styles["Normal"], fontSize=11, textColor=sc))]
    doc.build(story)
    buf.seek(0)
    log_action("export_pdf", {"eid": eid, "rul": rul})
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"relatorio_{eid}.pdf")
