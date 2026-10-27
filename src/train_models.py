from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "traffic_accident_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC = [
    "latitude", "longitude", "traffic_volume", "vehicle_count", "average_speed",
    "queue_length", "lane_occupancy", "temperature", "rainfall", "visibility",
    "road_width", "number_of_lanes", "accident_count",
]
CATEGORICAL = [
    "location", "traffic_density", "weather", "road_condition", "road_type",
    "intersection_type", "construction_zone", "accident_history", "traffic_violations",
    "pedestrian_density", "accident_severity",
]
BASE_FEATURES = NUMERIC + CATEGORICAL + ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour"]


def normalize_target(s: pd.Series) -> pd.Series:
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = sorted(s[out.isna()].astype(str).unique().tolist())
        raise ValueError(f"Target contains unsupported labels: {bad}. Use Low/Medium/High.")
    return out


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = set(NUMERIC + CATEGORICAL + ["date", "time", "traffic_level", "accident_risk"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().all():
        raise ValueError("The date column could not be parsed.")
    time_parsed = pd.to_datetime(out["time"].astype(str), format="%H:%M", errors="coerce")
    if time_parsed.isna().any():
        # Accept HH:MM:SS as well.
        time_parsed = pd.to_datetime(out["time"].astype(str), errors="coerce")
    if time_parsed.isna().all():
        raise ValueError("The time column could not be parsed. Use HH:MM or HH:MM:SS.")
    out["hour"] = time_parsed.dt.hour
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_peak_hour"] = out["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20, 21]).astype(int)
    out = out.drop(columns=["date", "time"])
    return out


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in NUMERIC + ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=350, max_depth=18, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            result["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted", labels=[0, 1, 2])
            )
        except Exception:
            result["roc_auc_ovr_weighted"] = None
    else:
        result["roc_auc_ovr_weighted"] = None
    return result


def train_one(df: pd.DataFrame, target: str, filename: str) -> dict:
    # The UI uses Low/Medium/High strings, while XGBoost requires numeric classes.
    # Encode targets once for every model and map them back when the app displays results.
    y_text = normalize_target(df[target])
    class_names = ["Low", "Medium", "High"]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    y = y_text.map(class_to_id).astype(int)

    X = prepare_dataframe(df.copy())
    X = X.drop(columns=["traffic_level", "accident_risk"], errors="ignore")

    # Prevent obvious leakage: target labels are never model features.
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    leaderboard = {}
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", make_preprocessor(train_x)), ("model", estimator)])
        cv_scores = cross_val_score(pipe, train_x, train_y, cv=cv, scoring="f1_macro", n_jobs=None)
        pipe.fit(train_x, train_y)
        fitted[name] = pipe
        leaderboard[name] = {
            "cv_f1_macro_mean": float(np.mean(cv_scores)),
            "cv_f1_macro_std": float(np.std(cv_scores)),
        }

    best_name = max(leaderboard, key=lambda n: leaderboard[n]["cv_f1_macro_mean"])
    best_model = fitted[best_name]
    metrics = evaluate(best_model, test_x, test_y)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": list(train_x.columns),
        "class_names": class_names,
        "metrics": metrics,
        "cv_leaderboard": leaderboard,
    }
    joblib.dump(bundle, MODEL_DIR / filename)
    return {
        "model_name": best_name,
        "metrics": metrics,
        "cv_leaderboard": leaderboard,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put your CSV there, or run: python src/generate_demo_data.py"
        )
    raw = pd.read_csv(DATA_PATH)
    if len(raw) < 100:
        raise ValueError("Use at least 100 rows for this baseline. More representative data is strongly preferred.")

    traffic = train_one(raw, "traffic_level", "traffic_model.pkl")
    accident = train_one(raw, "accident_risk", "accident_model.pkl")

    metrics = {
        "dataset_rows": int(len(raw)),
        "traffic": traffic,
        "accident": accident,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training complete.")
    print(f"Traffic model: {traffic['model_name']} | test macro-F1: {traffic['metrics']['f1_macro']:.4f}")
    print(f"Accident model: {accident['model_name']} | test macro-F1: {accident['metrics']['f1_macro']:.4f}")
    print("Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()
