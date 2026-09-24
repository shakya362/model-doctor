from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from model_doctor import audit


def test_detects_train_test_gap(clean_classification):
    """An unconstrained tree on noisy data memorises the training set."""
    result = audit(clean_classification, target="y",
                   model=DecisionTreeClassifier(random_state=0),
                   autofix=False, only=["overfitting."])
    assert result.has("overfitting.train_test_gap")
    f = next(x for x in result.findings if x.check_id == "overfitting.train_test_gap")
    assert f.evidence["gap"] > 0.1


def test_detects_suspiciously_perfect_score(leaky_classification):
    df = leaky_classification.copy()
    df["settlement_flag"] = df["y"]        # an exact copy of the target
    result = audit(df, target="y", autofix=False, only=["overfitting."])
    assert result.has("overfitting.suspiciously_perfect_score")


def test_regularised_model_on_clean_data_is_not_flagged(clean_classification):
    result = audit(clean_classification, target="y",
                   model=LogisticRegression(max_iter=500),
                   autofix=False, only=["overfitting."])
    assert not result.has("overfitting.train_test_gap")
