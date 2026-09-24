"""BROKEN PIPELINE 1 — leaky preprocessing + a target-derived feature.

Two planted bugs, both extremely common in real client code:

1. `StandardScaler` and the label encoding are fitted on the FULL dataset,
   before the train/test split, so test rows influence training.
2. `collections_contact_flag` is written by the collections team *after* a
   customer defaults, so it is a proxy for the answer.

Expected Model Doctor findings:
    leakage.target_proxy_feature
    code.preprocessing_fit_before_split
    quality.constant_columns
    quality.missing_values
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from datasets import credit_default

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "pipeline_01"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = credit_default()
    df.to_csv(OUT / "data.csv", index=False)

    y = df["defaulted"]
    X = df.drop(columns=["defaulted", "customer_id"])
    X["annual_income"] = X["annual_income"].fillna(X["annual_income"].mean())

    # BUG 1a: the encoder sees every row, including the ones held out later
    encoder = LabelEncoder()
    X["region"] = encoder.fit_transform(X["region"])
    X["extract_version"] = encoder.fit_transform(X["extract_version"])

    # BUG 1b: scaler statistics are computed over the test rows too
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print(f"[pipeline 01] test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

    joblib.dump(model, OUT / "model.joblib")


if __name__ == "__main__":
    main()
