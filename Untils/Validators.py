"""Untils/Validators.py — Validação de inputs"""
from Config.Settings import N_SENSORS

def validate_sensors(sensors) -> tuple[bool, str]:
    if not sensors:
        return False, "Lista de sensores vazia"
    if len(sensors) != N_SENSORS:
        return False, f"Esperados {N_SENSORS} sensores, recebidos {len(sensors)}"
    try:
        [float(s) for s in sensors]
    except (TypeError, ValueError):
        return False, "Valores de sensores inválidos"
    return True, ""

def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]

def validate_password(pw: str) -> tuple[bool, str]:
    if len(pw) < 4:
        return False, "Senha mínima de 4 caracteres"
    return True, ""
