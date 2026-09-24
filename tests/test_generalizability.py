"""The auditor must not be tied to one model type or one dataset shape."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_doctor import audit

MODELS = [
    ("logistic_regression", LogisticRegression(max_iter=500)),
    ("random_forest", RandomForestClassifier(n_estimators=40, random_state=0)),
    ("sklearn_pipeline", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=500))])),
]
try:
    from xgboost import XGBClassifier

    MODELS.append(("xgboost", XGBClassifier(n_estimators=40, max_depth=3, verbosity=0)))
except ImportError:  # pragma: no cover
    pass


@pytest.mark.parametrize("name,model", MODELS, ids=[m[0] for m in MODELS])
def test_runs_on_any_model_type(name, model, leaky_classification):
    df = leaky_classification.copy()
    df = pd.get_dummies(df, columns=["x3"])
    fitted = model.fit(df.drop(columns=["y"]), df["y"])
    result = audit(df, target="y", model=fitted, autofix=False)
    assert result.errors == [], result.errors
    assert result.has("leakage.target_proxy_feature")
    assert result.verdict in {"Healthy", "Needs attention", "Not safe to deploy"}


def test_at_least_three_model_families_supported():
    assert len(MODELS) >= 3


def test_runs_on_regression(clean_regression):
    result = audit(clean_regression, target="y", autofix=False)
    assert result.metrics["task"] == "regression"
    assert "r2" in result.metrics["holdout"]["test"]


def test_runs_on_multiclass():
    rng = np.random.default_rng(6)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = np.select([x1 + x2 > 1, x1 + x2 < -1], ["high", "low"], default="mid")
    result = audit(pd.DataFrame({"x1": x1, "x2": x2, "y": y}), target="y", autofix=False)
    assert result.errors == []


def test_runs_on_wide_and_tiny_frames():
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(rng.normal(size=(120, 60)), columns=[f"f{i}" for i in range(60)])
    wide["y"] = rng.binomial(1, 0.5, 120)
    assert audit(wide, target="y", autofix=False).errors == []

    tiny = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": list("xyzxyz"), "y": [0, 1, 0, 1, 0, 1]})
    assert audit(tiny, target="y", autofix=False) is not None


def test_handles_missing_target_rows(clean_classification):
    df = clean_classification.copy()
    df.loc[:40, "y"] = np.nan
    result = audit(df, target="y", task="classification", autofix=False)
    assert result.has("quality.missing_target_values")


def test_unknown_target_raises_clear_error(clean_classification):
    with pytest.raises(ValueError, match="not found"):
        audit(clean_classification, target="does_not_exist")


def test_a_single_broken_check_cannot_kill_the_audit(monkeypatch, clean_classification):
    from model_doctor.checks import REGISTRY, load_all

    load_all()
    monkeypatch.setitem(REGISTRY, "boom", lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    result = audit(clean_classification, target="y", autofix=False)
    assert any("boom" in e for e in result.errors)
    assert result.verdict  # the audit still produced a verdict
