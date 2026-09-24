"""Shared, cached evaluations.

Several checks need the same three facts, and each is expensive, so they are
computed once per audit and memoised on the context object:

* ``holdout_evaluation`` — refit a clone of the client's estimator on a proper
  split and score it on both train and test (this is what exposes overfitting).
* ``provided_model_scores`` — score the already-trained model the client handed
  us, on the data they handed us.
* ``majority_baseline`` — what a model that always predicts the majority class
  would score: the yardstick for "is this accuracy actually impressive?".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from .context import AuditContext, fresh_like, safe_predict, score_predictions


def _cache(ctx: AuditContext) -> Dict[str, Any]:
    if not hasattr(ctx, "_eval_cache"):
        ctx._eval_cache = {}  # type: ignore[attr-defined]
    return ctx._eval_cache  # type: ignore[attr-defined]


def holdout_evaluation(ctx: AuditContext) -> Optional[Dict[str, Any]]:
    cache = _cache(ctx)
    if "holdout" in cache:
        return cache["holdout"]
    result = None
    try:
        X_tr, X_te, y_tr, y_te = ctx.split()
        est = fresh_like(ctx.model, ctx.task)
        est.fit(X_tr, y_tr)

        def _sc(X, y):
            proba = None
            if hasattr(est, "predict_proba"):
                try:
                    proba = est.predict_proba(X)
                except Exception:
                    proba = None
            return score_predictions(y, est.predict(X), proba, ctx.task)

        result = {
            "train": _sc(X_tr, y_tr),
            "test": _sc(X_te, y_te),
            "estimator": type(est).__name__,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "test_predictions": est.predict(X_te),
            "y_test": y_te,
        }
    except Exception as exc:  # pragma: no cover - defensive
        ctx.notes.append(f"holdout evaluation failed: {exc}")
    cache["holdout"] = result
    return result


def provided_model_scores(ctx: AuditContext) -> Optional[Dict[str, Any]]:
    """Score the client's already-fitted model, mapping label spaces if needed."""
    cache = _cache(ctx)
    if "provided" in cache:
        return cache["provided"]
    result = None
    if ctx.model is not None:
        try:
            pred, proba, how = safe_predict(ctx.model, ctx.X, ctx.X_enc)
            if ctx.task == "classification":
                y_true_s = ctx.y.astype(str).to_numpy()
                pred_s = pd.Series(pred).astype(str).to_numpy()
                known = set(y_true_s)
                if not set(pred_s) & known:
                    # model emits encoded codes rather than original labels
                    _, uniques = pd.factorize(ctx.y.astype(str))
                    try:
                        pred_s = np.asarray([str(uniques[int(float(p))]) for p in pred])
                    except Exception:
                        pass
                result = {
                    "scores": score_predictions(y_true_s, pred_s, proba, ctx.task),
                    "predictions": pred_s,
                    "y_true": y_true_s,
                    "input_used": how,
                }
            else:
                y_true = pd.to_numeric(ctx.y, errors="coerce").to_numpy(dtype=float)
                result = {
                    "scores": score_predictions(y_true, np.asarray(pred, dtype=float), proba, ctx.task),
                    "predictions": np.asarray(pred, dtype=float),
                    "y_true": y_true,
                    "input_used": how,
                }
        except Exception as exc:
            ctx.notes.append(f"supplied model could not be scored: {exc}")
            _cache(ctx)["provided_error"] = str(exc)
    cache["provided"] = result
    return result


def majority_baseline(ctx: AuditContext) -> Dict[str, float]:
    cache = _cache(ctx)
    if "baseline" in cache:
        return cache["baseline"]
    try:
        X_tr, X_te, y_tr, y_te = ctx.split()
        dummy = DummyClassifier(strategy="most_frequent") if ctx.task == "classification" else DummyRegressor(strategy="mean")
        dummy.fit(X_tr, y_tr)
        scores = score_predictions(y_te, dummy.predict(X_te), None, ctx.task)
    except Exception:
        scores = {}
    cache["baseline"] = scores
    return scores
