"""CONTROL PIPELINE — deliberately correct, used to measure false positives.

An auditor that flags everything is useless. This pipeline does the same job as
pipeline 02 but properly: no duplication before the split, preprocessing inside
a Pipeline, a stratified split, a regularised model, and metrics beyond
accuracy. Model Doctor should report no critical or high findings here.
"""

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datasets import telecom_churn

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "pipeline_05"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = telecom_churn().drop_duplicates()
    df.to_csv(OUT / "data.csv", index=False)

    y = df["churned"]
    X = df.drop(columns=["churned"])

    # split FIRST, then fit every transformation inside the pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric = ["tenure_months", "monthly_charges", "support_tickets", "has_fibre"]
    categorical = ["contract_type"]
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=8, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    print(f"[pipeline 05] ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(classification_report(y_test, model.predict(X_test), digits=3))

    joblib.dump(model, OUT / "model.joblib")


if __name__ == "__main__":
    main()
