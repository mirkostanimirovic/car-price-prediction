"""Feature selection and preprocessing pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "priceUSD"

NUMERIC_FEATURES = [
    "year",
    "mileage(kilometers)",
    "volume(cm3)",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters",
    "is_newer_car",
    "is_high_mileage",
    "log_mileage",
    "log_engine_volume",
    "age_squared",
    "mileage_age_interaction",
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
    "brand_model",
]


def split_features_target(df: pd.DataFrame):
    """Split dataframe into X and y using the project's feature definition."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor():
    """Build a leakage-safe preprocessing transformer."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
