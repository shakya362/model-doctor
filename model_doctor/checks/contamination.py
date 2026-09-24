"""Train/test contamination.

Even a flawless estimator reports meaningless numbers if the same rows — or the
same entity — appear on both sides of the split. We look for exact duplicate
rows, near-duplicates, and grouped data cross-validated as if every row were
independent.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score

from ..context import ID_HINT, AuditContext, fresh_like
from ..findings import Category, Finding, Severity
from . import register


@register("contamination.duplicate_rows")
def duplicate_rows(ctx: AuditContext) -> List[Finding]:
    n = len(ctx.X)
    if n < 20:
        return []
    feat_dupes = int(ctx.X.duplicated(keep="first").sum())
    full = pd.concat([ctx.X, ctx.y.rename("__target__")], axis=1)
    n_full = int(full.duplicated(keep="first").sum())
    if n_full == 0 and feat_dupes == 0:
        return []

    out: List[Finding] = []
    if n_full:
        share = n_full / n
        expected_crossing = int(round(n_full * 2 * 0.25 * 0.75))
        severity = Severity.CRITICAL if share > 0.15 else (Severity.HIGH if share > 0.02 else Severity.MEDIUM)
        out.append(
            Finding(
                check_id="contamination.duplicate_rows",
                title=f"{n_full:,} duplicate rows ({share:.1%} of the dataset)",
                category=Category.CONTAMINATION,
                severity=severity,
                confidence=0.95,
                plain_english=(
                    f"{n_full:,} of {n:,} rows are exact copies of another row, target value included. "
                    "When the data is split into training and test sets, copies of the same record land "
                    f"on both sides — roughly {expected_crossing:,} rows here. The model is then tested "
                    "on records it has already memorised."
                ),
                why_it_matters=(
                    "The test score stops measuring 'can this handle a new customer' and starts measuring "
                    "'can this remember a row it has already seen'. It is the commonest reason an audited "
                    "model performs worse in production than on paper."
                ),
                suggested_fix=(
                    "De-duplicate before splitting (`df.drop_duplicates()`). If the duplicates were added "
                    "deliberately to balance the classes, move that oversampling inside the training fold "
                    "only — never before the split."
                ),
                evidence={"duplicate_rows": n_full, "total_rows": int(n), "share": round(share, 4)},
                auto_fixable=True,
            )
        )
    if feat_dupes > n_full:
        conflicting = feat_dupes - n_full
        out.append(
            Finding(
                check_id="contamination.conflicting_duplicates",
                title=f"{conflicting:,} rows share identical inputs but disagree on the outcome",
                category=Category.DATA_QUALITY,
                severity=Severity.MEDIUM,
                confidence=0.75,
                plain_english=(
                    f"{conflicting:,} rows have exactly the same feature values as another row but a "
                    "different target. No model can tell them apart, so they set a hard ceiling on the "
                    "accuracy anyone can achieve here."
                ),
                why_it_matters=(
                    "This usually means a distinguishing field was never collected, or that labels were "
                    "recorded inconsistently upstream."
                ),
                suggested_fix="Review a sample of the conflicting pairs with the data owner: add the missing field, or resolve the label conflict before training.",
                evidence={"conflicting_rows": int(conflicting)},
                auto_fixable=False,
            )
        )
    return out


def _candidate_group_columns(ctx: AuditContext) -> List[str]:
    n = len(ctx.X)
    cands = []
    for col in ctx.features:
        s = ctx.X[col]
        if pd.api.types.is_float_dtype(s):
            continue
        uniq = int(s.nunique(dropna=True))
        if uniq < 2 or uniq > n * 0.5:
            continue
        if (n - uniq) < max(5, 0.1 * n):
            continue
        if ID_HINT.search(col) or s.dtype == object or pd.api.types.is_integer_dtype(s):
            cands.append((col, uniq))
    if ctx.group_column and ctx.group_column in ctx.X.columns:
        cands.insert(0, (ctx.group_column, int(ctx.X[ctx.group_column].nunique())))
    cands.sort(key=lambda t: -t[1])
    return [c for c, _ in cands[:6]]


@register("contamination.group_leakage")
def group_leakage(ctx: AuditContext) -> List[Finding]:
    """The same entity (customer, store, patient) on both sides of the split."""
    n = len(ctx.X)
    if n < 120:
        return []
    out: List[Finding] = []
    if ctx.task == "regression":
        scoring = "r2"
    elif len(np.unique(ctx.y_enc)) == 2:
        scoring = "roc_auc"
    else:
        scoring = "f1_macro"

    for col in _candidate_group_columns(ctx):
        groups = ctx.X[col].astype(str).to_numpy()
        n_groups = int(len(np.unique(groups)))
        if n_groups < 4:
            continue
        X = ctx.X_enc.drop(columns=[col], errors="ignore")
        if X.shape[1] == 0:
            continue
        try:
            if ctx.task == "classification":
                k = int(max(2, min(4, pd.Series(ctx.y_enc).value_counts().min())))
                cv_naive = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            else:
                k = 4
                cv_naive = KFold(n_splits=k, shuffle=True, random_state=0)
            k_g = int(min(k, n_groups))
            if k_g < 2:
                continue
            naive = float(np.nanmean(cross_val_score(
                fresh_like(None, ctx.task), X, ctx.y_enc, cv=cv_naive, scoring=scoring, error_score=np.nan)))
            grouped = float(np.nanmean(cross_val_score(
                fresh_like(None, ctx.task), X, ctx.y_enc, cv=GroupKFold(n_splits=k_g),
                groups=groups, scoring=scoring, error_score=np.nan)))
        except Exception:
            continue
        if np.isnan(naive) or np.isnan(grouped):
            continue
        gap = naive - grouped
        if gap < 0.08:
            continue
        severity = Severity.HIGH if gap < 0.25 else Severity.CRITICAL
        out.append(
            Finding(
                check_id="contamination.group_leakage",
                title=f"Rows from the same '{col}' appear in both training and test data",
                category=Category.CONTAMINATION,
                severity=severity,
                confidence=0.7 if gap < 0.2 else 0.85,
                plain_english=(
                    f"The data contains repeated entities — {n_groups:,} distinct values of '{col}' across "
                    f"{n:,} rows — so splitting rows at random puts the same entity on both sides. Scored "
                    f"the usual way the model gets {naive:.3f}; holding out whole entities instead it gets "
                    f"{grouped:.3f}."
                ),
                why_it_matters=(
                    "The honest question is 'how well does this work for a customer we have never seen "
                    f"before'. The random split answers a much easier question and overstates performance "
                    f"by roughly {gap:.3f}."
                ),
                suggested_fix=f"Use GroupKFold / GroupShuffleSplit with groups='{col}' so every entity sits entirely in train or entirely in test.",
                evidence={
                    "group_column": col,
                    "n_groups": n_groups,
                    "random_cv_score": round(naive, 4),
                    "grouped_cv_score": round(grouped, 4),
                    "metric": scoring,
                    "optimism_gap": round(gap, 4),
                },
                affected_columns=[col],
                auto_fixable=True,
            )
        )
        break  # one clear group finding is enough; correlated columns would repeat it
    return out


@register("contamination.near_duplicate")
def near_duplicates(ctx: AuditContext) -> List[Finding]:
    n = len(ctx.X)
    if n < 100 or n > 20000:
        return []
    num = ctx.X_enc.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    exact = int(ctx.X.duplicated().sum())
    near = int(num.round(6).duplicated().sum()) - exact
    if near <= max(3, 0.005 * n):
        return []
    return [
        Finding(
            check_id="contamination.near_duplicate_rows",
            title=f"{near:,} near-identical rows beyond the exact duplicates",
            category=Category.CONTAMINATION,
            severity=Severity.MEDIUM,
            confidence=0.55,
            plain_english=(
                f"Beyond exact copies, {near:,} rows become identical once tiny numeric differences are "
                "ignored. During a random split these behave exactly like duplicates."
            ),
            why_it_matters="Near-duplicates leak the same way exact duplicates do, but they slip straight past `drop_duplicates()`.",
            suggested_fix="Round or bucket the offending numeric columns before de-duplicating, or de-duplicate on the business key rather than the whole row.",
            evidence={"near_duplicate_rows": near, "exact_duplicates": exact, "total_rows": int(n)},
            auto_fixable=False,
        )
    ]
