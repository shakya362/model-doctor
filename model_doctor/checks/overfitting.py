"""Overfitting signals: train-vs-test gap and suspiciously perfect scores."""

from __future__ import annotations

from typing import List

import numpy as np

from ..context import AuditContext, pick_headline
from ..evaluation import holdout_evaluation, provided_model_scores
from ..findings import Category, Finding, Severity
from . import register


@register("overfitting.train_test_gap")
def train_test_gap(ctx: AuditContext) -> List[Finding]:
    hold = holdout_evaluation(ctx)
    if not hold:
        return []
    key, train_val = pick_headline(hold["train"], ctx.task)
    test_val = hold["test"].get(key, float("nan"))
    if test_val != test_val or train_val != train_val:
        return []
    gap = float(train_val - test_val)

    # When no model was supplied we are measuring a standard model of our own
    # choosing, not the client's. That is still worth reporting — it says this
    # data invites memorisation — but it is attributed honestly and graded
    # softly, and it takes a much larger gap to be worth raising at all.
    surrogate = ctx.model is None
    if gap < (0.3 if surrogate else 0.1):
        return []

    if surrogate:
        severity, confidence = Severity.MEDIUM, 0.5
    elif gap >= 0.35:
        severity, confidence = Severity.CRITICAL, 0.9
    elif gap >= 0.2:
        severity, confidence = Severity.HIGH, 0.82
    else:
        severity, confidence = Severity.MEDIUM, 0.65

    params = []
    try:
        p = ctx.model.get_params() if ctx.model is not None else {}
        for k in ("max_depth", "min_samples_leaf", "n_estimators", "C", "alpha", "reg_lambda"):
            if k in p:
                params.append(f"{k}={p[k]}")
    except Exception:
        pass

    return [
        Finding(
            check_id="overfitting.train_test_gap",
            title="The model performs far better on data it trained on",
            category=Category.OVERFITTING,
            severity=severity,
            confidence=confidence,
            plain_english=(
                (
                    f"No trained model was supplied, so we fitted a standard random forest to see how this "
                    f"data behaves. It scores {train_val:.3f} on the rows it trained on and {test_val:.3f} "
                    f"on rows it has never seen ({key}) — a gap of {gap:.3f}. Data with this shape invites "
                    "a model to memorise rather than generalise, so whatever model you use needs to be "
                    "constrained and checked for the same gap."
                ) if surrogate else (
                    f"Refitting the same model type on a clean split gives {train_val:.3f} on the training "
                    f"rows but {test_val:.3f} on rows it has never seen ({key}) — a gap of {gap:.3f}. The "
                    "model has memorised the training file rather than learned a general rule."
                )
            ),
            why_it_matters=(
                "Whatever number was quoted from the training set is the number the model will never "
                "reproduce again. Only the held-out figure is a forecast of live performance."
            ),
            suggested_fix=(
                "Constrain the model (lower max_depth, raise min_samples_leaf, add regularisation), "
                "reduce the feature count, or get more rows. Tune against cross-validated scores, "
                "never against the training score."
                + (f" Current settings worth revisiting: {', '.join(params)}." if params else "")
            ),
            evidence={
                "metric": key,
                "train_score": round(float(train_val), 4),
                "test_score": round(float(test_val), 4),
                "gap": round(gap, 4),
                "estimator": hold["estimator"],
                "model_supplied_by_client": not surrogate,
                "n_train": hold["n_train"],
                "n_test": hold["n_test"],
            },
            auto_fixable=True,
        )
    ]


@register("overfitting.perfect_score")
def perfect_score(ctx: AuditContext) -> List[Finding]:
    """A perfect score on real-world data is a bug report, not an achievement."""
    prov = provided_model_scores(ctx)
    hold = holdout_evaluation(ctx)
    candidates = []
    if prov:
        key, val = pick_headline(prov["scores"], ctx.task)
        candidates.append(("the supplied trained model scored on the supplied dataset", key, val))
    if hold:
        key, val = pick_headline(hold["test"], ctx.task)
        candidates.append(("a fresh model of the same type on a held-out split", key, val))

    out: List[Finding] = []
    for label, key, val in candidates:
        if val != val or val < 0.995:
            continue
        out.append(
            Finding(
                check_id="overfitting.suspiciously_perfect_score",
                title=f"Suspiciously perfect score ({key} = {val:.4f})",
                category=Category.OVERFITTING,
                severity=Severity.CRITICAL,
                confidence=0.85,
                plain_english=(
                    f"Evaluating {label} produces {key} of {val:.4f}. Real business problems are noisy; "
                    "a near-perfect score almost always means the answer is reachable from the inputs — "
                    "leakage, duplicated rows, or the target accidentally left in the feature set."
                ),
                why_it_matters=(
                    "This is the strongest single warning sign in a model audit. Treat the number as "
                    "evidence of a pipeline bug until the cause has been found and explained."
                ),
                suggested_fix=(
                    "Work through the leakage findings in this report, remove the offending columns, "
                    "de-duplicate, then re-measure. A believable score is a successful audit outcome."
                ),
                evidence={"metric": key, "score": round(float(val), 5), "evaluated_on": label},
                auto_fixable=False,
            )
        )
        break
    return out
