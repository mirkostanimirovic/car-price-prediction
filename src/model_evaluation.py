"""Evaluate the saved final model on the same reproducible hold-out split."""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target

RANDOM_STATE = 42
TEST_SIZE = 0.20


def evaluate_final_model(df: pd.DataFrame, model_path):
    data = add_features(clean_data(df))
    X, y = split_features_target(data)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = joblib.load(model_path)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, pred),
    }

    examples = pd.DataFrame({
        "actual_price": y_test.values,
        "predicted_price": pred,
    })
    examples["absolute_error"] = (
        examples["actual_price"] - examples["predicted_price"]
    ).abs()

    return metrics, examples


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    metrics, examples = evaluate_final_model(
        pd.read_csv(root / "data" / "cars.csv"),
        root / "models" / "car_price_model.joblib",
    )

    (root / "reports").mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(root / "reports" / "final_metrics.csv", index=False)
    examples.head(20).to_csv(root / "reports" / "example_predictions.csv", index=False)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
