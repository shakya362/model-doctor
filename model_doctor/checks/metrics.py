"""Misleading metrics.

Accuracy on a 97/3 split is a lie told with true numbers. This check compares
the headline metric against a do-nothing baseline and against the metrics that
actually reflect the business question.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..context import AuditContext
from ..evaluation import holdout_evaluation, majority_baseline, provided_model_scores
from ..findings import Category, Finding, Severity
from . import register


@register("metrics.accuracy_on_imbalanced")
def accuracy_on_imbalanced(ctx: AuditContext) -> List[Finding]:
    if ctx.task != "classification":
        return []
    ratio = ctx.imbalance_ratio()
    if ratio < 3:
        return []

    prov = provided_model_scores(ctx)
    hold = holdout_evaluation(ctx)
    scores = (prov or {}).get("scores") or (hold or {}).get("test")
    if not scores:
        return []

    base = majority_baseline(ctx)
    base_acc = base.get("accuracy", ctx.class_counts().iloc[0] / len(ctx.y))
    acc = scores.get("accuracy", float("nan"))
    bal = scores.get("balanced_accuracy", float("nan"))
    f1 = scores.get("f1", scores.get("f1_macro", float("nan")))
    lift = acc - base_acc

    code_mentions_accuracy = False
    if ctx.code_text:
        low = ctx.code_text.lower()
        code_mentions_accuracy = ("accuracy" in low) and not any(
            k in low for k in ("f1_score", "roc_auc", "recall_score", "precision_score", "average_precision", "classification_report")
        )

    severity = Severity.HIGH if (lift < 0.05 or code_mentions_accuracy) else Severity.MEDIUM
    confidence = 0.9 if code_mentions_accuracy else (0.85 if lift < 0.03 else 0.65)
    minority = ctx.class_counts().index[-1]

    return [
        Finding(
            check_id="metrics.accuracy_on_imbalanced_classes",
            title="Accuracy is a misleading headline for this dataset",
            category=Category.METRICS,
            severity=severity,
            confidence=confidence,
            plain_english=(
                f"The outcome is heavily skewed — the commonest class is {ratio:.1f}× more frequent than "
                f"'{minority}'. A model that does nothing at all and always guesses the majority class "
                f"already scores {base_acc:.1%} accuracy. The model here scores {acc:.1%}, a gain of only "
                f"{lift:.1%}. On the measures that account for the skew it scores "
                f"{bal:.1%} balanced accuracy and {f1:.1%} F1."
                + (" The training script reports accuracy and nothing else." if code_mentions_accuracy else "")
            ),
            why_it_matters=(
                "The rare class is almost always the one with the money attached — the fraud, the churn, "
                "the failure. Accuracy hides whether the model catches any of them, so a 'great' model "
                "can be worth precisely nothing."
            ),
            suggested_fix=(
                "Report precision, recall, F1 and ROC-AUC (or PR-AUC when the positive class is very "
                "rare) alongside a majority-class baseline, and pick the operating threshold from the "
                "precision/recall trade-off the business actually wants."
            ),
            evidence={
                "imbalance_ratio": round(ratio, 2),
                "majority_baseline_accuracy": round(float(base_acc), 4),
                "model_accuracy": round(float(acc), 4),
                "accuracy_lift_over_baseline": round(float(lift), 4),
                "balanced_accuracy": None if bal != bal else round(float(bal), 4),
                "f1": None if f1 != f1 else round(float(f1), 4),
                "script_reports_accuracy_only": code_mentions_accuracy,
            },
            auto_fixable=True,
        )
    ]


@register("metrics.regression_baseline")
def weak_regression_signal(ctx: AuditContext) -> List[Finding]:
    if ctx.task != "regression":
        return []
    hold = holdout_evaluation(ctx)
    if not hold:
        return []
    r2 = hold["test"].get("r2", float("nan"))
    if r2 != r2 or r2 > 0.1:
        return []
    return [
        Finding(
            check_id="metrics.regression_no_better_than_mean",
            title="The model barely beats predicting the average",
            category=Category.METRICS,
            severity=Severity.HIGH,
            confidence=0.8,
            plain_english=(
                f"On held-out data the model explains {max(r2, 0):.1%} of the variation in the target. "
                "Always predicting the historical average would do about as well."
            ),
            why_it_matters="Error metrics like MAE can look acceptable in absolute terms while the model adds no information at all over a constant guess.",
            suggested_fix="Always report R² (or error relative to a mean/naive-seasonal baseline). If it stays near zero, the current features do not explain the target — revisit feature engineering before tuning the model.",
            evidence={"test_r2": round(float(r2), 4), "test_mae": round(float(hold["test"].get("mae", float("nan"))), 4)},
            auto_fixable=False,
        )
    ]
