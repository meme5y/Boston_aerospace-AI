"""Untils/File_helpers.py — Utilitários de ficheiros"""
import os, time, uuid
from Config.Settings import UPLOAD_DIR

ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_AUDIO  = {".wav", ".mp3", ".flac", ".ogg"}
ALLOWED_KB     = {".pdf", ".txt", ".md", ".csv"}

def safe_filename(original: str) -> str:
    ext = os.path.splitext(original)[1].lower() if "." in original else ".bin"
    return f"{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"

def save_upload(file_obj, subdir: str = "") -> str:
    dest = os.path.join(UPLOAD_DIR, subdir) if subdir else UPLOAD_DIR
    os.makedirs(dest, exist_ok=True)
    fname = safe_filename(file_obj.filename)
    path  = os.path.join(dest, fname)
    file_obj.save(path)
    return path

def allowed_extension(filename: str, allowed: set) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed
