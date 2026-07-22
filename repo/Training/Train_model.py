"""Training/Train_model.py — Treino do ensemble de 6 modelos ML"""
import os, json, joblib
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from Config.Settings import MODEL_DIR, MODEL_NAMES, META_FILE, SCALER_FILE, FCOLS_FILE, RUL_MAX
from Core.Preprocess import load_nasa_data, attach_rul
from Core.Feature_Engineering import add_temporal_features, build_feature_cols


def run_training() -> tuple:
    print("[TRAIN] A iniciar treino...")
    raw, is_nasa = load_nasa_data()
    raw = attach_rul(raw)
    df  = add_temporal_features(raw)
    fc  = build_feature_cols(df)

    X, y = df[fc].fillna(0), df["rul"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = RobustScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

    specs = {
        "xgb": xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbosity=0),
        "lgb": lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1),
        "cb":  cb.CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, random_seed=42, verbose=0),
        "rf":  RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "gbr": GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42),
        "et":  ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    }

    models = {}
    for name, m in specs.items():
        print(f"  [{name}]...", end=" ", flush=True)
        m.fit(Xtr_s, ytr)
        joblib.dump(m, os.path.join(MODEL_DIR, f"{name}.pkl"))
        models[name] = m
        print("ok")

    ens = np.mean([m.predict(Xte_s) for m in models.values()], axis=0)
    mae = round(float(mean_absolute_error(yte, ens)), 2)
    r2  = round(float(r2_score(yte, ens)), 4)
    meta = {"mae": mae, "r2": r2, "src": "NASA CMAPSS" if is_nasa else "Sintetico",
            "trained_at": datetime.now().isoformat()}

    joblib.dump(sc, SCALER_FILE)
    joblib.dump(fc, FCOLS_FILE)
    with open(META_FILE, "w") as f:
        json.dump(meta, f)

    print(f"[TRAIN] MAE={mae} | R2={r2} | {meta['src']}")
    return models, sc, fc, meta


if __name__ == "__main__":
    run_training()
