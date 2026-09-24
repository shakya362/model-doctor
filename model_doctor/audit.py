"""The orchestrator: ``audit()`` is the one function most users need."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .autofix import run_autofix
from .checks import load_all
from .checks.code_smells import read_source
from .context import (
    AuditContext,
    describe_model,
    infer_task,
    load_model,
    load_table,
    model_task,
    pick_headline,
)
from .evaluation import holdout_evaluation, majority_baseline, provided_model_scores
from .findings import AuditResult, Finding


def audit(
    data: Union[str, Path, pd.DataFrame],
    target: str,
    model: Any = None,
    code: Union[str, Path, None] = None,
    goal: Optional[str] = None,
    task: Optional[str] = None,
    time_column: Optional[str] = None,
    group_column: Optional[str] = None,
    autofix: bool = True,
    only: Optional[List[str]] = None,
) -> AuditResult:
    """Audit a dataset, a training pipeline and (optionally) a trained model.

    Parameters
    ----------
    data : path to a tabular file, or a DataFrame.
    target : name of the column being predicted.
    model : a fitted estimator, or a path to a .pkl/.joblib file. Optional.
    code : path to the training script (.py/.ipynb) for static analysis. Optional.
    goal : the client's stated objective, echoed back in the report.
    task : 'classification' or 'regression'; inferred when omitted.
    autofix : apply the mechanical fixes and re-measure, to show before/after.
    only : run a subset of checks (prefix match on the registered name).
    """
    df = data if isinstance(data, pd.DataFrame) else load_table(data)
    if isinstance(model, (str, Path)):
        model = load_model(model)

    code_text, code_path = None, None
    if code is not None:
        code_path = str(code)
        code_text = read_source(code)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset. Columns: {list(df.columns)}")

    resolved_task = task or model_task(model) or infer_task(df[target])
    ctx = AuditContext(
        df=df, target=target, task=resolved_task, model=model,
        code_text=code_text, code_path=code_path, goal=goal,
        time_column=time_column, group_column=group_column,
    )

    result = AuditResult(dataset_summary=ctx.summary(), model_summary=describe_model(model))
    for name, fn in sorted(load_all().items()):
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        try:
            found = fn(ctx) or []
        except Exception as exc:  # one broken check must never kill the audit
            result.errors.append(f"check '{name}' failed: {type(exc).__name__}: {exc}")
            continue
        result.findings.extend(f for f in found if isinstance(f, Finding))

    result.metrics = _collect_metrics(ctx)
    if autofix:
        result.autofix = run_autofix(ctx, result.findings)
    result.errors.extend(ctx.notes)
    result._ctx = ctx  # type: ignore[attr-defined]  # convenient in notebooks and tests
    return result


def _collect_metrics(ctx: AuditContext) -> Dict[str, Any]:
    out: Dict[str, Any] = {"task": ctx.task}
    hold = holdout_evaluation(ctx)
    if hold:
        key, _ = pick_headline(hold["test"], ctx.task)
        out["headline_metric"] = key
        out["holdout"] = {
            "estimator": hold["estimator"],
            "train": {k: round(float(v), 4) for k, v in hold["train"].items()},
            "test": {k: round(float(v), 4) for k, v in hold["test"].items()},
            "n_train": hold["n_train"],
            "n_test": hold["n_test"],
        }
    prov = provided_model_scores(ctx)
    if prov:
        out["supplied_model"] = {k: round(float(v), 4) for k, v in prov["scores"].items()}
    base = majority_baseline(ctx)
    if base:
        out["baseline"] = {k: round(float(v), 4) for k, v in base.items()}
    return out
