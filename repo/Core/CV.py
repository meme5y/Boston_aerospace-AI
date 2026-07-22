"""Core/CV.py — Visão computacional: trincas, térmica, áudio"""
import base64
import numpy as np

try:
    import cv2
    CV_OK = True
except ImportError:
    CV_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


def detect_cracks(path: str) -> dict:
    if not CV_OK:
        return {"error": "pip install opencv-python-headless"}
    img = cv2.imread(path)
    if img is None:
        return {"error": "Imagem inválida"}
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.equalizeHist(gray), (5,5), 0), 40, 120)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cks = [c for c in cnts if 30 < cv2.contourArea(c) < 10000]
    n   = len(cks)
    sev = "CRÍTICO" if n>8 else "ALTO" if n>4 else "MÉDIO" if n>1 else "BAIXO"
    clr = {"CRÍTICO":(0,0,255),"ALTO":(0,100,255),"MÉDIO":(0,220,255),"BAIXO":(0,255,100)}[sev]
    ann = img.copy()
    cv2.drawContours(ann, cks, -1, clr, 2)
    cv2.putText(ann, f"{n} | {sev}", (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, clr, 2)
    _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"sev":sev,"n":n,"detected":n>1,
            "rec":{"CRÍTICO":"PARAR","ALTO":"24H","MÉDIO":"MONITORAR","BAIXO":"OK"}[sev],
            "img": base64.b64encode(buf).decode()}


def analyze_thermal(path: str) -> dict:
    if not CV_OK:
        return {"error": "pip install opencv-python-headless"}
    img  = cv2.imread(path)
    if img is None:
        return {"error": "Imagem inválida"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hot = [c for c in cnts if cv2.contourArea(c) > 80]
    mx  = int(gray.max())
    sev = "CRÍTICO" if mx>230 else "ALTO" if mx>180 else "NORMAL"
    cv2.putText(heat, f"{len(hot)} | {sev}", (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    _, buf = cv2.imencode(".jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"sev":sev,"hot":len(hot),"max":mx,
            "rec":{"CRÍTICO":"DESLIGAR","ALTO":"REDUZIR","NORMAL":"OK"}[sev],
            "img": base64.b64encode(buf).decode()}


def analyze_audio(path: str) -> dict:
    if not LIBROSA_OK:
        return {"error": "pip install librosa"}
    y, sr = librosa.load(path, sr=None, duration=30, mono=True)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms   = float(np.mean(librosa.feature.rms(y=y)))
    cent  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    fft   = np.abs(np.fft.rfft(y))
    peak  = float(np.fft.rfftfreq(len(y), 1/sr)[np.argmax(fft)])
    anom  = rms > 0.15 or cent > 4000
    return {"anom":anom,"rms":round(rms,4),"cent":round(cent,1),"peak":round(peak,1),
            "sr":sr,"dur":round(len(y)/sr,1),
            "mfcc":[round(float(np.mean(mfcc[i])),2) for i in range(13)],
            "rec":"VIBRAÇÃO ANÔMALA" if anom else "SINAL NORMAL"}
