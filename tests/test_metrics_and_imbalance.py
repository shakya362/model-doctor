from model_doctor import audit


def test_flags_accuracy_on_imbalanced_target(imbalanced_classification):
    result = audit(imbalanced_classification, target="y", autofix=False,
                   only=["metrics.", "imbalance."])
    assert result.has("metrics.accuracy_on_imbalanced_classes")
    assert result.has("imbalance.skewed_target")
    f = next(x for x in result.findings if x.check_id == "metrics.accuracy_on_imbalanced_classes")
    assert f.evidence["majority_baseline_accuracy"] > 0.9


def test_flags_model_that_never_predicts_the_minority(imbalanced_classification):
    result = audit(imbalanced_classification, target="y", autofix=False, only=["imbalance."])
    assert result.has("imbalance.model_ignores_minority_class")


def test_balanced_target_is_not_flagged(clean_classification):
    result = audit(clean_classification, target="y", autofix=False,
                   only=["metrics.", "imbalance."])
    assert not result.has("metrics.accuracy_on_imbalanced_classes")
    assert not result.has("imbalance.skewed_target")


def test_regression_no_better_than_mean():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(9)
    n = 500
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
                       "y": rng.normal(0, 1, n)})  # target is pure noise
    result = audit(df, target="y", autofix=False, only=["metrics."])
    assert result.has("metrics.regression_no_better_than_mean")
