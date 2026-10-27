from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "traffic_accident_data.csv"


def load_bundle(name: str) -> dict:
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run `python src/train_models.py` first.")
    return joblib.load(path)


def prepare_single_record(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    date = pd.to_datetime(df["date"], errors="coerce")
    time = pd.to_datetime(df["time"].astype(str), format="%H:%M", errors="coerce")
    if pd.isna(time.iloc[0]):
        time = pd.to_datetime(df["time"].astype(str), errors="coerce")
    if pd.isna(date.iloc[0]) or pd.isna(time.iloc[0]):
        raise ValueError("Invalid date or time.")
    df["hour"] = time.dt.hour
    df["day_of_week"] = date.dt.dayofweek
    df["month"] = date.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20, 21]).astype(int)
    return df.drop(columns=["date", "time"])


def predict(bundle: dict, payload: dict) -> dict:
    model = bundle["model"]
    frame = prepare_single_record(payload)
    frame = frame.reindex(columns=bundle["feature_columns"], fill_value=None)
    prediction_id = int(model.predict(frame)[0])
    class_names = bundle.get("class_names", ["Low", "Medium", "High"])
    prediction = class_names[prediction_id] if 0 <= prediction_id < len(class_names) else str(prediction_id)
    probs = model.predict_proba(frame)[0] if hasattr(model, "predict_proba") else None
    result = {"prediction": prediction, "probability": None, "probabilities": {}}
    if probs is not None:
        result["probabilities"] = {name: float(p) for name, p in zip(class_names, probs)}
        result["probability"] = float(max(probs))
    return result


def load_metrics() -> dict:
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
