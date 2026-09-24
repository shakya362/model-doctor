"""Train/inference mismatch — does the saved artifact still work on this data?

A model is delivered as a file. If that file cannot reproduce the preparation
steps applied during training, it silently produces different numbers in
production from the ones signed off in the notebook. These checks compare the
artifact against the dataset it is supposed to score.
"""

from __future__ import annotations

from typing import List

from ..context import AuditContext, pick_headline
from ..evaluation import _cache, holdout_evaluation, provided_model_scores
from ..findings import Category, Finding, Severity
from . import register


@register("inference.artifact_mismatch")
def artifact_mismatch(ctx: AuditContext) -> List[Finding]:
    """The saved model will not accept the dataset in the form it arrives."""
    if ctx.model is None:
        return []
    prov = provided_model_scores(ctx)
    if prov is not None:
        return []
    error = _cache(ctx).get("provided_error", "")
    expected = getattr(ctx.model, "n_features_in_", None)
    return [
        Finding(
            check_id="inference.model_cannot_score_dataset",
            title="The saved model cannot read the dataset it is meant to score",
            category=Category.DATA_QUALITY,
            severity=Severity.HIGH,
            confidence=0.9,
            plain_english=(
                "The model file was handed over on its own, but it will not accept the dataset in the "
                f"form the dataset arrives in"
                + (f" — it expects {expected} prepared columns, and the raw data has {len(ctx.features)}." if expected else ".")
                + " The steps that turned raw data into model inputs live in the training script rather "
                "than inside the model file."
            ),
            why_it_matters=(
                "Whoever deploys this has to re-create those steps by hand from the training code. Any "
                "small difference — a column dropped, a category encoded in a different order — changes "
                "the predictions silently, with no error message."
            ),
            suggested_fix=(
                "Save the preprocessing and the estimator together as one `sklearn.pipeline.Pipeline` "
                "object, so the artifact carries its own preparation and can score raw rows directly."
            ),
            evidence={"expected_features": expected, "dataset_features": len(ctx.features), "error": error[:300]},
            auto_fixable=False,
        )
    ]


@register("inference.artifact_underperforms")
def artifact_underperforms(ctx: AuditContext) -> List[Finding]:
    """The artifact scores far worse than a refit — usually missing preprocessing."""
    prov = provided_model_scores(ctx)
    hold = holdout_evaluation(ctx)
    if not prov or not hold:
        return []
    key, refit = pick_headline(hold["test"], ctx.task)
    supplied = prov["scores"].get(key)
    if supplied is None or refit != refit:
        return []
    gap = float(refit) - float(supplied)
    if gap < 0.12:
        return []
    return [
        Finding(
            check_id="inference.artifact_underperforms_refit",
            title="The delivered model scores worse than a straight re-fit of the same type",
            category=Category.DATA_QUALITY,
            severity=Severity.HIGH,
            confidence=0.7,
            plain_english=(
                f"Scored on this dataset, the delivered model file achieves {key} of {supplied:.3f}. "
                f"Re-fitting the same kind of model on the same data reaches {refit:.3f}. A gap that large "
                "normally means the model expects inputs that have been scaled or encoded in a particular "
                "way, and that step is not packaged with it."
            ),
            why_it_matters=(
                "Predictions served in production will not match the ones measured during development. "
                "This is the failure mode that produces 'it worked in the notebook' incidents."
            ),
            suggested_fix=(
                "Bundle the scaler/encoder and the estimator into a single Pipeline and re-export the "
                f"artifact, then confirm it reproduces the development score of about {refit:.3f} on this data."
            ),
            evidence={
                "metric": key,
                "delivered_model_score": round(float(supplied), 4),
                "refit_score": round(float(refit), 4),
                "gap": round(gap, 4),
                "input_form_accepted": prov.get("input_used"),
            },
            auto_fixable=False,
        )
    ]
