import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_FILE = Path("data/processed/model_splits/test.csv")


def main():
    df = pd.read_csv(TEST_FILE)

    required = [
        "target_water_level",
        "lag_1_water_level",
    ]

    before = len(df)

    evaluation = df.dropna(subset=required).copy()

    after = len(evaluation)

    actual = evaluation["target_water_level"]
    predicted = evaluation["lag_1_water_level"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print("========== PERSISTENCE BASELINE ==========")
    print("Method: Tomorrow = today's water level")
    print(f"Test rows: {before:,}")
    print(f"Evaluated rows: {after:,}")
    print(f"Excluded rows: {before - after:,}")

    print(f"\nMAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


if __name__ == "__main__":
    main()
