"""Tests/Test_predictor.py — Testes do motor de predição"""
import pytest

def test_get_status_normal():
    from Core.Predictor import get_status
    st, color, rec = get_status(100)
    assert st == "NORMAL"

def test_get_status_critico():
    from Core.Predictor import get_status
    st, _, _ = get_status(10)
    assert st == "CRÍTICO"

def test_feature_engineering():
    import pandas as pd
    from Core.Feature_Engineering import build_inference_row
    df = build_inference_row([490]*18, 150)
    assert len(df) == 1

def test_validate_sensors():
    from Untils.Validators import validate_sensors
    ok, _ = validate_sensors([1.0]*18)
    assert ok is True
    ok, msg = validate_sensors([1.0]*5)
    assert ok is False
