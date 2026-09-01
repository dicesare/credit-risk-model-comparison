from __future__ import annotations

import pandas as pd


def split_features_target(data: pd.DataFrame, target: str = "default"):
    if target not in data:
        raise ValueError(f"Missing target column: {target}")
    labels = data[target]
    if not set(labels.dropna().unique()) <= {0, 1}:
        raise ValueError("Target must be binary (0/1)")
    return data.drop(columns=[target]), labels.astype(int)


def build_model_pipeline(data: pd.DataFrame, model: str = "logistic", random_state: int = 42):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = data.select_dtypes(include="number").columns.tolist()
    categorical = data.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    estimators = {
        "logistic": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1),
    }
    if model not in estimators:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline([("preprocessing", preprocessing), ("model", estimators[model])])
