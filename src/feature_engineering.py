"""Feature engineering utilities."""

import pandas as pd
import numpy as np

REFERENCE_YEAR = 2019


def add_features(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    data = df.copy()

    data["car_age"] = (reference_year - data["year"]).clip(lower=0)
    data["mileage_per_year"] = (
        data["mileage(kilometers)"] / data["car_age"].replace(0, 1)
    )
    data["engine_volume_liters"] = data["volume(cm3)"] / 1000.0
    data["is_newer_car"] = (data["year"] >= 2015).astype(int)
    data["is_high_mileage"] = (data["mileage(kilometers)"] >= 300_000).astype(int)
    data["brand_model"] = (
        data["make"].fillna("missing").astype(str) + "_" +
        data["model"].fillna("missing").astype(str)
    )

    # Additional nonlinear features useful for tree models and diagnostics.
    data["log_mileage"] = np.log1p(data["mileage(kilometers)"])
    data["log_engine_volume"] = np.log1p(data["volume(cm3)"])
    data["age_squared"] = data["car_age"] ** 2
    data["mileage_age_interaction"] = (
        data["mileage(kilometers)"] * data["car_age"]
    )

    return data


if __name__ == "__main__":
    from pathlib import Path
    from data_cleaning import clean_data

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "cars.csv")
    features = add_features(clean_data(df))
    features.to_csv(root / "data" / "cars_features.csv", index=False)
    print(f"Saved {len(features):,} rows to data/cars_features.csv")
