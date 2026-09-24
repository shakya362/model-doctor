from model_doctor import audit


def test_detects_duplicate_rows(duplicated_classification):
    result = audit(duplicated_classification, target="y", autofix=False, only=["contamination."])
    assert result.has("contamination.duplicate_rows")
    f = next(x for x in result.findings if x.check_id == "contamination.duplicate_rows")
    assert f.evidence["duplicate_rows"] > 100
    assert f.severity.rank >= 3


def test_no_duplicates_flagged_on_clean_data(clean_classification):
    result = audit(clean_classification, target="y", autofix=False, only=["contamination."])
    assert not result.has("contamination.duplicate_rows")


def test_detects_group_leakage():
    """Repeated entities with their own level: random CV beats grouped CV."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2)
    rows = []
    for patient in range(60):
        offset = rng.normal(0, 4)
        for _ in range(14):
            x = rng.normal(0, 1)
            rows.append({
                "patient_id": f"P{patient}",
                # an alias of the entity id left in the feature set — the usual
                # way per-entity effects get memorised across a random split
                "patient_code": 9000 + patient,
                "x": x,
                "y": 2 * x + offset + rng.normal(0, 0.4),
            })
    df = pd.DataFrame(rows)
    result = audit(df, target="y", autofix=False, only=["contamination.group"])
    assert result.has("contamination.group_leakage")
