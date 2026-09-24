"""Audit context — everything the checks share.

The context is deliberately model-agnostic: it never imports a specific
estimator library. Any object exposing ``fit``/``predict`` works, so logistic
regression, random forests, XGBoost, LightGBM and sklearn Pipelines all travel
through the same code path.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 20260919
DATE_HINT = re.compile(r"(date|datetime|timestamp|_dt$|^dt_|^time$|_time$|period_index)", re.I)
ID_HINT = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|^key$|_key$|_no$|number$)", re.I)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_table(path: str | Path) -> pd.DataFrame:
    """Load an arbitrary tabular file (csv/tsv/parquet/xlsx/json)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported data file type: {suffix}")


def load_model(path: str | Path) -> Any:
    """Load a .pkl / .joblib model."""
    import joblib

    p = Path(path)
    if p.suffix.lower() in {".joblib", ".pkl", ".pickle", ".sav"}:
        return joblib.load(p)
    raise ValueError(f"Unsupported model file type: {p.suffix}")


# --------------------------------------------------------------------------- #
# task inference & encoding
# --------------------------------------------------------------------------- #
def infer_task(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    y = y.dropna()
    if y.empty:
        raise ValueError("Target column is entirely missing.")
    if y.dtype == bool or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        return "classification"
    nun = y.nunique()
    if pd.api.types.is_integer_dtype(y) and nun <= max(20, int(0.01 * len(y))):
        return "classification"
    if nun <= 2:
        return "classification"
    return "regression"


def is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            return False
        ok = sum(
            bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", v) or re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", v))
            for v in sample
        )
        return ok / len(sample) > 0.8
    return False


def encode_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model-free numeric encoding used for internal probing only.

    Numerics are kept, datetimes become epoch seconds, everything else becomes
    ordinal codes. Missing values get an out-of-range sentinel so that
    'missingness' itself stays visible to the probe models.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[col] = pd.to_datetime(s, errors="coerce").astype("int64") / 1e9
        elif is_datetime_like(s):
            out[col] = pd.to_datetime(s, errors="coerce").astype("int64") / 1e9
        else:
            out[col] = pd.factorize(s.astype(str), use_na_sentinel=True)[0].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(-999_999.0)


def encode_target(y: pd.Series, task: str) -> np.ndarray:
    if task == "classification":
        return pd.factorize(y.astype(str))[0]
    return pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score_predictions(y_true, y_pred, y_proba=None, task: str = "classification") -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task == "regression":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2))),
        }
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    classes = np.unique(y_true)
    if len(classes) == 2:
        pos = classes[-1]
        out["precision"] = float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0))
        if y_proba is not None:
            try:
                p = np.asarray(y_proba, dtype=float)
                p = p[:, -1] if p.ndim == 2 else p
                out["roc_auc"] = float(roc_auc_score((y_true == pos).astype(int), p))
            except Exception:
                pass
    return out


def pick_headline(scores: Dict[str, float], task: str) -> Tuple[str, float]:
    if task == "regression":
        return "r2", scores.get("r2", float("nan"))
    for key in ("roc_auc", "f1_macro", "balanced_accuracy", "accuracy"):
        if key in scores:
            return key, scores[key]
    return "accuracy", scores.get("accuracy", float("nan"))


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def model_task(model: Any) -> Optional[str]:
    try:
        if is_classifier(model):
            return "classification"
        if is_regressor(model):
            return "regression"
    except Exception:
        pass
    name = type(model).__name__.lower()
    if "classifier" in name:
        return "classification"
    if "regressor" in name:
        return "regression"
    return None


def describe_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {"type": None}
    info: Dict[str, Any] = {"type": type(model).__name__, "task": model_task(model)}
    steps = getattr(model, "steps", None)
    if steps:
        info["pipeline_steps"] = [f"{n}:{type(s).__name__}" for n, s in steps]
    try:
        params = model.get_params(deep=False)
        info["params"] = {k: str(v) for k, v in list(params.items())[:25]}
    except Exception:
        pass
    for attr in ("n_features_in_", "classes_"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            info[attr] = val.tolist() if hasattr(val, "tolist") else val
    return info


def fresh_like(model: Any, task: str) -> Any:
    """An unfitted clone of the client's estimator, or a sensible default."""
    if model is not None:
        try:
            return clone(model)
        except Exception:
            try:
                return type(model)(**model.get_params())
            except Exception:
                pass
    if task == "classification":
        return RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, n_jobs=-1)
    return RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE, n_jobs=-1)


def probe_estimator(task: str, depth: Optional[int] = None):
    if task == "classification":
        return DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    return DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)


def _candidate_inputs(model: Any, X_raw: pd.DataFrame, X_enc: pd.DataFrame):
    """Frames to try, best guess first, when scoring a model someone handed us."""
    out = []
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        cols = [str(c) for c in names]
        if all(c in X_raw.columns for c in cols):
            out.append((X_raw[cols], "matched the model's recorded feature names"))
            out.append((X_enc[cols], "matched feature names, encoded numerically"))
    n_expected = getattr(model, "n_features_in_", None)
    out.append((X_raw, "the dataset as supplied"))
    out.append((X_enc, "the dataset, encoded numerically"))
    if n_expected is not None and X_enc.shape[1] != n_expected:
        # the commonest cause: an id column dropped during training
        trimmed = X_enc.drop(columns=[c for c in X_enc.columns if ID_HINT.search(c)], errors="ignore")
        if trimmed.shape[1] == n_expected:
            out.append((trimmed, "the dataset with identifier columns removed"))
    return out


def safe_predict(model: Any, X_raw: pd.DataFrame, X_enc: pd.DataFrame):
    """Predict with a model that may expect raw frames, encoded arrays or a subset.

    Returns ``(predictions, probabilities, how)`` where ``how`` records which
    input shape the model accepted — useful evidence when the saved artifact
    does not match the dataset it is supposed to score.
    """
    last_error = None
    for candidate, how in _candidate_inputs(model, X_raw, X_enc):
        try:
            pred = model.predict(candidate)
        except Exception as exc:
            last_error = exc
            continue
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(candidate)
            except Exception:
                proba = None
        return np.asarray(pred), proba, how
    raise RuntimeError(f"Model could not predict on the supplied dataset: {last_error}")


def fit_score(estimator, X_tr, y_tr, X_te, y_te, task: str) -> Dict[str, float]:
    estimator.fit(X_tr, y_tr)
    proba = None
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X_te)
        except Exception:
            proba = None
    return score_predictions(y_te, estimator.predict(X_te), proba, task)


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
@dataclass
class AuditContext:
    df: pd.DataFrame
    target: str
    task: str
    model: Any = None
    code_text: Optional[str] = None
    code_path: Optional[str] = None
    goal: Optional[str] = None
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    X: pd.DataFrame = field(init=False)
    y: pd.Series = field(init=False)
    X_enc: pd.DataFrame = field(init=False)
    y_enc: np.ndarray = field(init=False)
    features: List[str] = field(init=False)
    dropped_target_na: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.target not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found. Available: {list(self.df.columns)[:25]}"
            )
        self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()
        mask = self.df[self.target].notna()
        self.dropped_target_na = int((~mask).sum())
        frame = self.df.loc[mask].reset_index(drop=True)
        self.y = frame[self.target]
        self.X = frame.drop(columns=[self.target])
        self.features = list(self.X.columns)
        self.X_enc = encode_matrix(self.X)
        self.y_enc = encode_target(self.y, self.task)
        if self.time_column is None:
            self.time_column = self._guess_time_column()

    def _guess_time_column(self) -> Optional[str]:
        candidates = [c for c in self.X.columns if is_datetime_like(self.X[c])]
        if not candidates:
            # numeric fallback: the name must look like a date AND the values must
            # behave like an index (mostly non-decreasing down the file).
            candidates = []
            for c in self.X.columns:
                if not (DATE_HINT.search(c) and pd.api.types.is_numeric_dtype(self.X[c])):
                    continue
                vals = pd.to_numeric(self.X[c], errors="coerce").dropna()
                if vals.nunique() <= 5 or vals.empty:
                    continue
                diffs = vals.diff().dropna()
                if not diffs.empty and float((diffs >= 0).mean()) >= 0.9:
                    candidates.append(c)
        if not candidates:
            return None
        best, best_score = None, -1.0
        for c in candidates:
            try:
                vals = pd.to_datetime(self.X[c], errors="coerce")
                if vals.isna().mean() > 0.5:
                    vals = pd.to_numeric(self.X[c], errors="coerce")
                order = vals.dropna()
                if order.empty:
                    continue
                diffs = order.diff().dropna()
                if diffs.empty:
                    continue
                score = float((diffs >= (pd.Timedelta(0) if pd.api.types.is_timedelta64_dtype(diffs) else 0)).mean())
            except Exception:
                continue
            if score > best_score:
                best, best_score = c, score
        if best is None:
            return candidates[0]
        return best

    def class_counts(self) -> pd.Series:
        return self.y.astype(str).value_counts()

    def imbalance_ratio(self) -> float:
        if self.task != "classification":
            return 1.0
        counts = self.class_counts()
        if len(counts) < 2:
            return float("inf")
        return float(counts.iloc[0] / counts.iloc[-1])

    def minority_share(self) -> float:
        counts = self.class_counts()
        return float(counts.iloc[-1] / counts.sum()) if len(counts) > 1 else 1.0

    def split(self, test_size: float = 0.25, stratify: bool = True):
        strat = None
        if stratify and self.task == "classification":
            counts = pd.Series(self.y_enc).value_counts()
            if counts.min() >= 2:
                strat = self.y_enc
        return train_test_split(
            self.X_enc, self.y_enc, test_size=test_size,
            random_state=RANDOM_STATE, stratify=strat,
        )

    def time_split(self, test_size: float = 0.25):
        if self.time_column is None or self.time_column not in self.X.columns:
            return None
        order = pd.to_datetime(self.X[self.time_column], errors="coerce")
        if order.isna().mean() > 0.5:
            order = pd.to_numeric(self.X[self.time_column], errors="coerce")
        if order.isna().all():
            return None
        idx = np.argsort(order.fillna(order.min()).to_numpy(), kind="stable")
        cut = int(len(idx) * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
        if len(tr) < 20 or len(te) < 10:
            return None
        if self.task == "classification" and len(np.unique(self.y_enc[tr])) < 2:
            return None
        return self.X_enc.iloc[tr], self.X_enc.iloc[te], self.y_enc[tr], self.y_enc[te]

    def summary(self) -> Dict[str, Any]:
        counts = self.class_counts() if self.task == "classification" else None
        return {
            "rows": int(len(self.df)),
            "columns": int(self.df.shape[1]),
            "features": len(self.features),
            "target": self.target,
            "task": self.task,
            "goal": self.goal,
            "time_column": self.time_column,
            "group_column": self.group_column,
            "target_missing_rows_dropped": int(self.dropped_target_na),
            "class_balance": {str(k): int(v) for k, v in counts.items()} if counts is not None else None,
            "code_analysed": bool(self.code_text),
        }
