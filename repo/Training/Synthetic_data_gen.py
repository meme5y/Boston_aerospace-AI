"""Training/Synthetic_data_gen.py — Geração de dados sintéticos"""
import numpy as np
import pandas as pd
import os
from Config.Settings import DATA_DIR

def generate_and_save(n_engines: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    for e in range(1, n_engines + 1):
        life = np.random.randint(150, 420)
        for c in range(1, life + 1):
            d   = (c / life) ** 1.4
            row = [f"SYN_{e}", c, 0, 0, 100]
            for s in range(21):
                row.append(np.clip(300 + s * 12 + d * 120 + np.random.normal(0, 7), 50, 950))
            rows.append(row)
    df = pd.DataFrame(rows, columns=["unit","cycle","op1","op2","op3"] +
                      [f"s{i}" for i in range(1, 22)])
    out = os.path.join(DATA_DIR, "..", "Synthetic", "synthetic_cmapss.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[GEN] {len(df)} linhas geradas -> {out}")
    return df

if __name__ == "__main__":
    generate_and_save()
