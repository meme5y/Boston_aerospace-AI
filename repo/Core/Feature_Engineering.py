"""Core/Feature_Engineering.py — Feature engineering para sensores CMAPSS"""
import pandas as pd
import numpy as np
from Config.Settings import SENSOR_KEYS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona rolling mean (10 ciclos) e diff por motor."""
    df = df.copy().sort_values(["unit", "cycle"])
    for s in SENSOR_KEYS:
        if s not in df.columns:
            continue
        g = df.groupby("unit")[s]
        df[f"{s}_m"] = g.transform(lambda x: x.rolling(10, min_periods=1).mean())
        df[f"{s}_d"] = g.diff().fillna(0)
    return df


def build_feature_cols(df: pd.DataFrame) -> list:
    """Devolve lista ordenada de colunas de features disponíveis."""
    base  = SENSOR_KEYS + ["op1", "op2", "op3", "cycle"]
    extra = [c for c in df.columns if any(c.startswith(f"{s}_") for s in SENSOR_KEYS)]
    return [c for c in base + extra if c in df.columns]


def build_inference_row(sensors: list, cycle: float) -> pd.DataFrame:
    """Constrói DataFrame de 1 linha para inferência."""
    from Config.Settings import N_SENSORS
    row = dict(zip(SENSOR_KEYS, sensors[:N_SENSORS]))
    row.update({"cycle": cycle, "op1": 0, "op2": 0, "op3": 100, "unit": "TEST"})
    return add_temporal_features(pd.DataFrame([row]))
