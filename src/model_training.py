"""Train and save the final selected model."""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target, build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_final_model():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=20,
            max_depth=30,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def train_final_model(df: pd.DataFrame):
    data = add_features(clean_data(df))
    X, y = split_features_target(data)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_final_model()
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "cars.csv")
    model = train_final_model(df)
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(model, root / "models" / "car_price_model.joblib")
    print("Saved models/car_price_model.joblib")
