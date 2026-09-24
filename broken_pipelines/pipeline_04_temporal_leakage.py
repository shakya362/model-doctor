"""BROKEN PIPELINE 4 — future information in a time series.

Planted bugs:

1. `rolling_7d_avg_sales` is a CENTRED rolling mean, so each row's feature
   includes days that had not happened yet at prediction time.
2. Time-ordered rows are split randomly rather than chronologically.
3. The same stores appear in both training and test data, so per-store levels
   are memorised rather than learned.
4. Category labels are inconsistent ('North' / 'north ' / 'NORTH') and missing
   weather is stored as the string 'unknown'.

Expected Model Doctor findings:
    leakage.temporal_split  /  code.random_split_on_time_series
    leakage.target_proxy_feature (the centred rolling mean)
    contamination.group_leakage
    quality.inconsistent_category_labels
    quality.disguised_missing_values
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from datasets import store_sales

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "pipeline_04"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = store_sales()
    df.to_csv(OUT / "data.csv", index=False)

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"]).astype("int64") // 10**9
    frame["store_id"] = frame["store_id"].astype("category").cat.codes
    frame["region"] = frame["region"].astype("category").cat.codes
    frame["weather"] = frame["weather"].astype("category").cat.codes

    y = frame["sales"]
    X = frame.drop(columns=["sales"])

    # BUG 2: a random shuffle of time-ordered rows
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # also wrong for time series

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    print(f"[pipeline 04] test R^2: {r2_score(y_test, model.predict(X_test)):.4f}  <- too good to be true")

    joblib.dump(model, OUT / "model.joblib")


if __name__ == "__main__":
    main()
