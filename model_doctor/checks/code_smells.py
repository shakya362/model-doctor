"""Static analysis of the training pipeline source.

The dataset tells you *that* something is wrong; the source tells you *where*.
This module parses the training script with `ast` (never executing it) and
looks for ordering mistakes that no amount of staring at metrics will reveal —
most importantly, preprocessing fitted before the train/test split.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..context import AuditContext
from ..findings import Category, Finding, Severity
from . import register

PREPROCESSOR_HINT = re.compile(
    r"(scaler|standardscaler|minmax|robustscaler|normalizer|encoder|labelencoder|onehot|ordinal|"
    r"imputer|selector|selectkbest|pca|vectorizer|discretizer|power|quantile)", re.I
)
RESAMPLER_HINT = re.compile(r"(smote|adasyn|randomoversampler|randomundersampler|resample|oversample)", re.I)
SPLIT_FUNCS = {"train_test_split"}
CV_NAIVE = {"KFold", "StratifiedKFold", "ShuffleSplit", "cross_val_score", "cross_validate", "GridSearchCV", "RandomizedSearchCV"}
CV_SAFE = {"TimeSeriesSplit", "GroupKFold", "GroupShuffleSplit", "StratifiedGroupKFold", "LeaveOneGroupOut"}


def read_source(path: str | Path) -> str:
    """Read a .py file or flatten a .ipynb into plain source."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".ipynb":
        nb = json.loads(text)
        cells = [
            "".join(c.get("source", []))
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        return "\n\n".join(cells)
    return text


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.splits: List[int] = []
        self.fits: List[Tuple[int, str, str]] = []      # (line, receiver, method)
        self.calls: List[Tuple[int, str]] = []          # (line, func name)
        self.pipeline_lines: List[int] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        line = getattr(node, "lineno", 0)
        func = node.func
        if isinstance(func, ast.Name):
            self.calls.append((line, func.id))
            if func.id in ("Pipeline", "make_pipeline", "ColumnTransformer"):
                self.pipeline_lines.append(line)
        elif isinstance(func, ast.Attribute):
            self.calls.append((line, func.attr))
            if func.attr in ("fit", "fit_transform", "fit_resample", "fit_predict"):
                recv = ast.unparse(func.value) if hasattr(ast, "unparse") else ""
                self.fits.append((line, recv, func.attr))
            if func.attr in ("Pipeline", "make_pipeline"):
                self.pipeline_lines.append(line)
        self.generic_visit(node)


def _analyse(source: str) -> Optional[_Visitor]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    v = _Visitor()
    v.visit(tree)
    for line, name in list(v.calls):
        if name in SPLIT_FUNCS:
            v.splits.append(line)
    v.splits.sort()
    return v


@register("code.preprocessing_before_split")
def preprocessing_before_split(ctx: AuditContext) -> List[Finding]:
    if not ctx.code_text:
        return []
    v = _analyse(ctx.code_text)
    if v is None or not v.splits:
        return []
    first_split = v.splits[0]
    lines = ctx.code_text.splitlines()

    offenders = []
    for line, recv, method in v.fits:
        if line >= first_split or method == "fit_predict":
            continue
        snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if PREPROCESSOR_HINT.search(recv) or PREPROCESSOR_HINT.search(snippet):
            offenders.append({"line": line, "object": recv, "method": method, "code": snippet[:160]})

    if not offenders:
        return []
    return [
        Finding(
            check_id="code.preprocessing_fit_before_split",
            title="Data preparation is fitted on the full dataset before the split",
            category=Category.LEAKAGE,
            severity=Severity.CRITICAL,
            confidence=0.92,
            plain_english=(
                "In the training script, the scaling/encoding step is fitted on every row "
                f"(line{'s' if len(offenders) > 1 else ''} {', '.join(str(o['line']) for o in offenders)}) "
                f"before the data is split into training and test sets on line {first_split}. The test set "
                "therefore influences how the training data is prepared."
            ),
            why_it_matters=(
                "The averages, ranges and category lists used to prepare the data are computed using "
                "information from the test rows. The test score is quietly inflated, and the numbers "
                "cannot be reproduced on genuinely new data where those statistics are unknown."
            ),
            suggested_fix=(
                "Split first, then wrap every preprocessing step and the estimator in a single "
                "`sklearn.pipeline.Pipeline` and call `fit` on the training portion only. The pipeline "
                "then applies the *training* statistics to test and production data automatically."
            ),
            evidence={"split_line": first_split, "offending_calls": offenders, "pipeline_used": bool(v.pipeline_lines)},
            auto_fixable=True,
        )
    ]


@register("code.resample_before_split")
def resample_before_split(ctx: AuditContext) -> List[Finding]:
    if not ctx.code_text:
        return []
    v = _analyse(ctx.code_text)
    if v is None or not v.splits:
        return []
    lines = ctx.code_text.splitlines()
    first_split = v.splits[0]
    offenders = []
    for line, name in v.calls:
        if line >= first_split:
            continue
        snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if RESAMPLER_HINT.search(name) or RESAMPLER_HINT.search(snippet):
            offenders.append({"line": line, "code": snippet[:160]})
    if not offenders:
        return []
    return [
        Finding(
            check_id="code.resample_before_split",
            title="Rows are duplicated to balance classes before the split",
            category=Category.CONTAMINATION,
            severity=Severity.CRITICAL,
            confidence=0.9,
            plain_english=(
                f"Oversampling happens on line {offenders[0]['line']}, before the train/test split on line "
                f"{first_split}. Copies of the same record therefore end up in both the training and the "
                "test set."
            ),
            why_it_matters="The model is tested on rows it has already been trained on, so the reported score is closer to a memory test than a performance measure.",
            suggested_fix="Split first, then resample the training fold only — `imblearn.pipeline.Pipeline` does this correctly inside cross-validation.",
            evidence={"split_line": first_split, "offending_calls": offenders},
            auto_fixable=True,
        )
    ]


@register("code.cv_strategy")
def cv_strategy(ctx: AuditContext) -> List[Finding]:
    if not ctx.code_text:
        return []
    v = _analyse(ctx.code_text)
    if v is None:
        return []
    names = {n for _, n in v.calls}
    out: List[Finding] = []

    if ctx.time_column and not (names & CV_SAFE):
        if names & CV_NAIVE or v.splits:
            has_shuffle_false = bool(re.search(r"shuffle\s*=\s*False", ctx.code_text))
            if not has_shuffle_false:
                out.append(
                    Finding(
                        check_id="code.random_split_on_time_series",
                        title="Time-ordered data is validated with a random split",
                        category=Category.LEAKAGE,
                        severity=Severity.HIGH,
                        confidence=0.7,
                        plain_english=(
                            f"The dataset has a time column ('{ctx.time_column}') but the script validates "
                            "with a random shuffle rather than a chronological split."
                        ),
                        why_it_matters="Random shuffling trains on future records and tests on past ones, which is impossible in production and inflates the score.",
                        suggested_fix="Replace KFold/train_test_split with TimeSeriesSplit, or split on a cut-off date with shuffle=False.",
                        evidence={"time_column": ctx.time_column, "cv_calls_found": sorted(names & (CV_NAIVE | SPLIT_FUNCS))},
                        auto_fixable=True,
                    )
                )

    if v.splits and ctx.task == "classification" and ctx.imbalance_ratio() >= 3:
        if not re.search(r"stratify\s*=", ctx.code_text):
            out.append(
                Finding(
                    check_id="code.split_without_stratify",
                    title="The split is not stratified on an imbalanced target",
                    category=Category.METRICS,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    plain_english=(
                        f"`train_test_split` is called without `stratify` while the target is imbalanced "
                        f"({ctx.imbalance_ratio():.1f}:1). The proportion of rare cases in the test set is "
                        "left to chance."
                    ),
                    why_it_matters="Results swing between runs purely because of which rare rows landed in the test set, so improvements cannot be told apart from luck.",
                    suggested_fix="Pass `stratify=y` to train_test_split and use StratifiedKFold for cross validation.",
                    evidence={"imbalance_ratio": round(ctx.imbalance_ratio(), 2), "split_lines": v.splits},
                    auto_fixable=True,
                )
            )
    return out


@register("code.no_pipeline")
def no_pipeline(ctx: AuditContext) -> List[Finding]:
    if not ctx.code_text:
        return []
    v = _analyse(ctx.code_text)
    if v is None or v.pipeline_lines:
        return []
    preprocessing_present = any(
        PREPROCESSOR_HINT.search(recv) or PREPROCESSOR_HINT.search(m) for _, recv, m in v.fits
    ) or bool(PREPROCESSOR_HINT.search(ctx.code_text))
    if not preprocessing_present:
        return []
    return [
        Finding(
            check_id="code.preprocessing_outside_pipeline",
            title="Preprocessing is not wrapped in a pipeline",
            category=Category.CODE,
            severity=Severity.LOW,
            confidence=0.6,
            plain_english=(
                "Scaling/encoding steps are applied as standalone statements rather than inside an "
                "sklearn Pipeline. Nothing is broken today, but the training and inference paths are "
                "maintained separately and will drift apart."
            ),
            why_it_matters="Most train/inference mismatches in production start here: a step is updated in one script and forgotten in the other.",
            suggested_fix="Move every transformation into `Pipeline`/`ColumnTransformer` and persist that single object — it guarantees inference repeats training exactly.",
            evidence={"fit_calls": [{"line": l, "object": r, "method": m} for l, r, m in v.fits][:10]},
            auto_fixable=False,
        )
    ]
