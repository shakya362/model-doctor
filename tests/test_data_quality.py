from model_doctor import audit


def test_detects_quality_problems(messy_quality):
    result = audit(messy_quality, target="y", autofix=False, only=["quality."])
    assert result.has("quality.missing_values")
    assert result.has("quality.constant_columns")
    assert result.has("quality.inconsistent_category_labels")
    assert result.has("quality.disguised_missing_values")


def test_clean_data_raises_no_quality_flags(clean_classification):
    result = audit(clean_classification, target="y", autofix=False, only=["quality."])
    assert result.findings == []


def test_single_class_target(clean_classification):
    df = clean_classification.copy()
    df["y"] = 1
    result = audit(df, target="y", task="classification", autofix=False, only=["quality."])
    assert result.has("quality.single_class_target")
