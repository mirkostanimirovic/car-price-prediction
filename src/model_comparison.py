"""Train and compare multiple regression algorithms.

The preprocessing transformer is fitted only on the training split and then
reused for every model. This keeps the comparison fair and avoids data leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target, build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.20


def evaluate_models(df: pd.DataFrame):
    data = add_features(clean_data(df))
    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit preprocessing once, using training data only.
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=20, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=20, max_depth=30, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    rows = []
    predictions = {}

    for name, estimator in models.items():
        estimator.fit(X_train_t, y_train)
        pred = estimator.predict(X_test_t)
        predictions[name] = pred

        mse = mean_squared_error(y_test, pred)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, pred),
        })

    result = pd.DataFrame(rows).sort_values(
        ["MAE", "RMSE"], ascending=True
    ).reset_index(drop=True)

    return result, (X_train, X_test, y_train, y_test), predictions


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "cars.csv")
    result, _, _ = evaluate_models(df)
    (root / "reports").mkdir(exist_ok=True)
    result.to_csv(root / "reports" / "model_comparison.csv", index=False)
    print(result.to_string(index=False))
