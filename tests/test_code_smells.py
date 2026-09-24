"""Static analysis of training scripts — no code is executed."""
from model_doctor import audit

LEAKY_SCRIPT = '''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")
X, y = df.drop(columns=["y"]), df["y"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LogisticRegression().fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
'''

CORRECT_SCRIPT = '''
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

df = pd.read_csv("data.csv")
X, y = df.drop(columns=["y"]), df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
model.fit(X_train, y_train)
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), f1_score(y_test, model.predict(X_test)))
'''

RESAMPLE_SCRIPT = '''
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X, y = df.drop(columns=["y"]), df["y"]
X, y = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
RandomForestClassifier().fit(X_train, y_train)
'''


def _write(tmp_path, text, name="train.py"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_detects_preprocessing_before_split(tmp_path, clean_classification):
    result = audit(clean_classification, target="y", code=_write(tmp_path, LEAKY_SCRIPT),
                   autofix=False, only=["code."])
    assert result.has("code.preprocessing_fit_before_split")
    f = next(x for x in result.findings if x.check_id == "code.preprocessing_fit_before_split")
    assert f.evidence["offending_calls"][0]["line"] < f.evidence["split_line"]


def test_correct_script_is_not_flagged(tmp_path, clean_classification):
    result = audit(clean_classification, target="y", code=_write(tmp_path, CORRECT_SCRIPT),
                   autofix=False, only=["code."])
    assert not result.has("code.preprocessing_fit_before_split")
    assert not result.has("code.preprocessing_outside_pipeline")


def test_detects_resampling_before_split(tmp_path, clean_classification):
    result = audit(clean_classification, target="y", code=_write(tmp_path, RESAMPLE_SCRIPT),
                   autofix=False, only=["code."])
    assert result.has("code.resample_before_split")


def test_detects_missing_stratify_on_imbalanced_target(tmp_path, imbalanced_classification):
    result = audit(imbalanced_classification, target="y", code=_write(tmp_path, RESAMPLE_SCRIPT),
                   autofix=False, only=["code."])
    assert result.has("code.split_without_stratify")


def test_reads_notebooks(tmp_path, clean_classification):
    import json
    nb = {"cells": [{"cell_type": "markdown", "source": ["# title"]},
                    {"cell_type": "code", "source": LEAKY_SCRIPT.splitlines(keepends=True)}]}
    p = tmp_path / "train.ipynb"
    p.write_text(json.dumps(nb))
    result = audit(clean_classification, target="y", code=p, autofix=False, only=["code."])
    assert result.has("code.preprocessing_fit_before_split")


def test_malformed_script_does_not_crash_the_audit(tmp_path, clean_classification):
    result = audit(clean_classification, target="y",
                   code=_write(tmp_path, "def broken(:\n  this is not python"),
                   autofix=False, only=["code."])
    assert result.errors == [] and result.findings == []
