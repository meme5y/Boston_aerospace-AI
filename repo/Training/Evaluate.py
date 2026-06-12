"""Training/Evaluate.py — Avaliação e métricas do modelo"""
import json
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from Config.Settings import MODEL_DIR, MODEL_NAMES, SCALER_FILE, FCOLS_FILE
from Core.Preprocess import load_nasa_data, attach_rul
from Core.Feature_Engineering import add_temporal_features, build_feature_cols
from sklearn.model_selection import train_test_split


def evaluate_all() -> dict:
    raw, _ = load_nasa_data()
    raw = attach_rul(raw)
    df  = add_temporal_features(raw)
    fc  = joblib.load(FCOLS_FILE)
    sc  = joblib.load(SCALER_FILE)

    X, y = df[fc].fillna(0), df["rul"]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xte_s = sc.transform(Xte)

    results = {}
    preds_all = []
    for name in MODEL_NAMES:
        import os
        m   = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
        p   = m.predict(Xte_s)
        preds_all.append(p)
        results[name] = {
            "mae":  round(float(mean_absolute_error(yte, p)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(yte, p))), 2),
            "r2":   round(float(r2_score(yte, p)), 4),
        }

    ens = np.mean(preds_all, axis=0)
    results["ensemble"] = {
        "mae":  round(float(mean_absolute_error(yte, ens)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, ens))), 2),
        "r2":   round(float(r2_score(yte, ens)), 4),
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    evaluate_all()
