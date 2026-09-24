"""BROKEN PIPELINE 2 — oversampling before the split, plus an unconstrained tree.

Planted bugs:

1. Minority-class rows are duplicated to "balance the data" BEFORE the
   train/test split, so identical rows sit on both sides of it.
2. The decision tree is grown without any depth limit, so it memorises them.

Expected Model Doctor findings:
    contamination.duplicate_rows
    code.resample_before_split
    overfitting.train_test_gap
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from datasets import telecom_churn

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "pipeline_02"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = telecom_churn()
    df = pd.get_dummies(df, columns=["contract_type"], drop_first=False)

    # BUG 1: oversample the minority class before splitting
    churners = df[df["churned"] == 1]
    df = pd.concat([df, churners, churners], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(OUT / "data.csv", index=False)

    y = df["churned"]
    X = df.drop(columns=["churned"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # BUG 2: no depth limit — the tree can memorise every duplicated row
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(f"[pipeline 02] train accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"[pipeline 02] test  accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

    joblib.dump(model, OUT / "model.joblib")


if __name__ == "__main__":
    main()
