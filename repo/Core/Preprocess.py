"""Core/Preprocess.py — Carregamento e pré-processamento dos dados"""
import os
import numpy as np
import pandas as pd
from Config.Settings import DATA_DIR, CMAPSS_COLS, RUL_MAX


def load_nasa_data() -> tuple[pd.DataFrame, bool]:
    """Carrega dados NASA CMAPSS ou gera dados sintéticos."""
    dfs = []
    for fn in ["train_FD001.txt", "train_FD002.txt"]:
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLS)
            df["unit"] = fn[:5] + "_" + df["unit"].astype(str)
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True), True
    return generate_synthetic(), False


def generate_synthetic() -> pd.DataFrame:
    """Gera dados sintéticos de degradação de motor."""
    np.random.seed(42)
    rows = []
    for e in range(1, 201):
        life = np.random.randint(150, 420)
        for c in range(1, life + 1):
            d = (c / life) ** 1.4
            row = [f"E{e}", c, 0, 0, 100]
            for s in range(21):
                row.append(np.clip(300 + s * 12 + d * 120 + np.random.normal(0, 7), 50, 950))
            rows.append(row)
    return pd.DataFrame(rows, columns=["unit", "cycle", "op1", "op2", "op3"] +
                        [f"s{i}" for i in range(1, 22)])


def attach_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula e anexa coluna RUL ao DataFrame."""
    df = df.copy()
    df["rul"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).clip(upper=RUL_MAX)
    return df
