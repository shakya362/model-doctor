"""Class imbalance blindness — the model that never says 'yes'."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from ..context import AuditContext, pick_headline
from ..evaluation import holdout_evaluation, provided_model_scores
from ..findings import Category, Finding, Severity
from . import register


@register("imbalance.skewed_target")
def skewed_target(ctx: AuditContext) -> List[Finding]:
    if ctx.task != "classification":
        return []
    ratio = ctx.imbalance_ratio()
    if ratio < 5:
        return []
    counts = ctx.class_counts()
    minority, n_min = counts.index[-1], int(counts.iloc[-1])
    severity = Severity.MEDIUM if ratio < 20 else Severity.HIGH
    return [
        Finding(
            check_id="imbalance.skewed_target",
            title=f"The target is heavily skewed ({ratio:.0f}:1)",
            category=Category.IMBALANCE,
            severity=severity,
            confidence=0.9,
            plain_english=(
                f"Only {n_min:,} of {len(ctx.y):,} rows ({ctx.minority_share():.1%}) belong to the "
                f"'{minority}' class. Left untreated, standard training pushes the model towards the "
                "majority because that is the cheapest way to look accurate."
            ),
            why_it_matters="The rare class is usually the one that matters commercially; a model tuned for overall accuracy quietly optimises against it.",
            suggested_fix=(
                "Use class_weight='balanced' (or scale_pos_weight for XGBoost), resample *inside* the "
                "training fold only, stratify every split, and choose the decision threshold from a "
                "precision/recall curve rather than leaving it at 0.5."
            ),
            evidence={
                "imbalance_ratio": round(ratio, 2),
                "minority_class": str(minority),
                "minority_rows": n_min,
                "class_counts": {str(k): int(v) for k, v in counts.items()},
            },
            auto_fixable=True,
        )
    ]


@register("imbalance.majority_predictor")
def majority_predictor(ctx: AuditContext) -> List[Finding]:
    """Does the model ever actually predict the minority class?"""
    if ctx.task != "classification":
        return []

    sources = []
    prov = provided_model_scores(ctx)
    hold = holdout_evaluation(ctx)

    # If the delivered artifact scores far below a straight re-fit it is not
    # reproducing its own preprocessing, so its predictions say nothing about
    # whether the *modelling* ignores the minority class. Judge the re-fit.
    trust_supplied = prov is not None
    if prov is not None and hold is not None:
        key, refit = pick_headline(hold["test"], ctx.task)
        supplied = prov["scores"].get(key)
        if supplied is not None and refit == refit and (float(refit) - float(supplied)) >= 0.12:
            trust_supplied = False

    if trust_supplied and prov is not None:
        sources.append(("the supplied trained model", np.asarray(prov["y_true"]), np.asarray(prov["predictions"])))
    elif hold:
        sources.append(("a freshly fitted model of the same type", np.asarray(hold["y_test"]), np.asarray(hold["test_predictions"])))
    if not sources:
        return []

    label, y_true, y_pred = sources[0]
    actual_classes = pd.Series(y_true).astype(str).value_counts()
    pred_classes = pd.Series(y_pred).astype(str).value_counts()
    if len(actual_classes) < 2:
        return []

    missing = [c for c in actual_classes.index if c not in set(pred_classes.index)]
    minority = actual_classes.index[-1]
    try:
        rec = float(recall_score(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=[minority], average="macro", zero_division=0))
    except Exception:
        rec = float("nan")

    if not missing and (rec != rec or rec > 0.25):
        return []

    never_predicts = bool(missing)
    severity = Severity.CRITICAL if never_predicts else Severity.HIGH
    confidence = 0.95 if never_predicts else 0.8
    share = float(pred_classes.iloc[0] / pred_classes.sum())

    return [
        Finding(
            check_id="imbalance.model_ignores_minority_class",
            title=(f"The model never predicts '{missing[0]}'" if never_predicts else f"The model catches almost none of the '{minority}' cases"),
            category=Category.IMBALANCE,
            severity=severity,
            confidence=confidence,
            plain_english=(
                f"Across the evaluation rows, {label} assigns {share:.1%} of all predictions to a single class. "
                + (
                    f"It has never once predicted '{missing[0]}' — the class it exists to find."
                    if never_predicts
                    else f"It correctly identifies only {rec:.1%} of the actual '{minority}' cases."
                )
            ),
            why_it_matters=(
                "A model that cannot flag the rare case delivers zero business value no matter what its "
                "accuracy says — every alert it was supposed to raise is silently missed."
            ),
            suggested_fix=(
                "Re-train with balanced class weights, lower the decision threshold to match the cost of a "
                "miss versus a false alarm, and track recall on the rare class as the primary metric."
            ),
            evidence={
                "evaluated_on": label,
                "predicted_class_distribution": {str(k): int(v) for k, v in pred_classes.items()},
                "actual_class_distribution": {str(k): int(v) for k, v in actual_classes.items()},
                "never_predicted_classes": [str(m) for m in missing],
                "minority_recall": None if rec != rec else round(rec, 4),
            },
            auto_fixable=True,
        )
    ]
