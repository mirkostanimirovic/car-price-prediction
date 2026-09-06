"""Data cleaning utilities for the car price prediction project."""

from pathlib import Path
import pandas as pd

VALID_YEAR_MIN = 1950
VALID_YEAR_MAX = 2019
MIN_PRICE_USD = 100
MAX_MILEAGE = 1_500_000
MAX_ENGINE_VOLUME = 10_000


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw cars dataframe without imputing missing values.

    Missing values are intentionally left for the preprocessing pipeline.
    """
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    required = {
        "make", "model", "priceUSD", "year", "condition",
        "mileage(kilometers)", "fuel_type", "volume(cm3)",
        "color", "transmission", "drive_unit", "segment"
    }
    missing_required = required.difference(data.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    # Normalize categorical text.
    categorical = data.select_dtypes(include="object").columns
    for col in categorical:
        data[col] = data[col].astype("string").str.strip().str.lower()
        data[col] = data[col].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA}).astype(object)
        data[col] = data[col].where(pd.notna(data[col]), None)

    # Ensure numerical columns are numeric.
    numeric = ["priceUSD", "year", "mileage(kilometers)", "volume(cm3)"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    before = len(data)
    data = data.drop_duplicates().copy()

    # Remove rows that cannot represent a valid target/vehicle record.
    data = data[data["priceUSD"].notna() & (data["priceUSD"] >= MIN_PRICE_USD)]
    data = data[data["year"].notna() & data["year"].between(VALID_YEAR_MIN, VALID_YEAR_MAX)]
    data = data[data["mileage(kilometers)"].notna() & data["mileage(kilometers)"].between(0, MAX_MILEAGE)]

    # Engine volume may be missing, but an observed value must be plausible.
    valid_volume = data["volume(cm3)"].isna() | data["volume(cm3)"].between(1, MAX_ENGINE_VOLUME)
    data = data[valid_volume]

    data = data.reset_index(drop=True)
    data.attrs["rows_before_cleaning"] = before
    data.attrs["rows_after_cleaning"] = len(data)
    return data


def cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return a compact before/after cleaning report."""
    return pd.DataFrame([
        {"check": "rows", "raw": len(raw), "cleaned": len(cleaned)},
        {"check": "duplicates", "raw": int(raw.duplicated().sum()),
         "cleaned": int(cleaned.duplicated().sum())},
        {"check": "missing_volume", "raw": int(raw["volume(cm3)"].isna().sum()),
         "cleaned": int(cleaned["volume(cm3)"].isna().sum())},
        {"check": "missing_drive_unit", "raw": int(raw["drive_unit"].isna().sum()),
         "cleaned": int(cleaned["drive_unit"].isna().sum())},
        {"check": "missing_segment", "raw": int(raw["segment"].isna().sum()),
         "cleaned": int(cleaned["segment"].isna().sum())},
    ])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(root / "data" / "cars.csv")
    cleaned = clean_data(raw)
    cleaned.to_csv(root / "data" / "cars_clean.csv", index=False)
    print(f"Saved {len(cleaned):,} rows to data/cars_clean.csv")
