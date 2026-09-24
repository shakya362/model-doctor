"""BROKEN PIPELINE 3 — a 98% accurate fraud model that catches no fraud.

Planted bugs:

1. Accuracy is the only metric reported, on a target that is ~2% positive.
2. Nothing compensates for the imbalance, so the model learns to answer "not
   fraud" to everything and is congratulated for it.
3. The split is not stratified, so the rare class lands where chance puts it.

Expected Model Doctor findings:
    metrics.accuracy_on_imbalanced_classes
    imbalance.model_ignores_minority_class
    imbalance.skewed_target
    code.split_without_stratify
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from datasets import card_fraud

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "pipeline_03"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = card_fraud()
    df.to_csv(OUT / "data.csv", index=False)

    X = pd.get_dummies(df.drop(columns=["is_fraud"]), columns=["merchant_category"])
    y = df["is_fraud"]

    # BUG 3: no stratify on a 2% positive rate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # BUG 2: no class weighting on a severely imbalanced target
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, y_train)

    # BUG 1: accuracy is the only number anyone sees
    print(f"[pipeline 03] accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}  <- looks great")

    joblib.dump(model, OUT / "model.joblib")


if __name__ == "__main__":
    main()
