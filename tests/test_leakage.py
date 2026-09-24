"""Does the auditor catch leakage — and stay quiet when there is none?"""
from model_doctor import audit


def test_detects_target_proxy_feature(leaky_classification):
    result = audit(leaky_classification, target="y", autofix=False, only=["leakage."])
    assert result.has("leakage.target_proxy_feature")
    finding = next(f for f in result.findings if f.check_id == "leakage.target_proxy_feature")
    assert finding.affected_columns == ["settlement_flag"]
    assert finding.confidence >= 0.6
    assert finding.auto_fixable


def test_no_leakage_flagged_on_clean_data(clean_classification):
    result = audit(clean_classification, target="y", autofix=False, only=["leakage."])
    assert not result.has("leakage.target_proxy_feature")


def test_detects_identifier_column(clean_classification):
    df = clean_classification.copy()
    df.insert(0, "customer_id", [f"C{i}" for i in range(len(df))])
    result = audit(df, target="y", autofix=False, only=["leakage."])
    assert result.has("leakage.identifier_column")


def test_low_cardinality_id_named_column_is_not_an_identifier(clean_classification):
    """store_id over 40 stores is a category, not a record identifier."""
    df = clean_classification.copy()
    df["store_id"] = [f"S{i % 40}" for i in range(len(df))]
    result = audit(df, target="y", autofix=False, only=["leakage.identifier"])
    assert not result.has("leakage.identifier_column")


def test_detects_future_window_feature():
    """A centred rolling mean is caught; a properly lagged one is not."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    frames = []
    for store in range(12):
        level = rng.normal(1000, 200)
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        sales = level * (1 + 0.3 * np.sin(np.arange(120) / 3)) * rng.normal(1, 0.1, 120)
        f = pd.DataFrame({"date": dates, "store": f"S{store}", "sales": sales})
        f["centred_avg"] = f["sales"].rolling(7, center=True, min_periods=1).mean()
        f["lagged_avg"] = f["sales"].shift(1).rolling(7, min_periods=1).mean().bfill()
        frames.append(f)
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "store"]).reset_index(drop=True)

    result = audit(df, target="sales", autofix=False, only=["leakage.future_window"])
    flagged = {c for f in result.findings for c in f.affected_columns}
    assert "centred_avg" in flagged, "centred rolling mean should be flagged"
    assert "lagged_avg" not in flagged, "a properly lagged feature must not be flagged"
