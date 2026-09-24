"""Data quality issues that silently degrade performance.

These rarely make a model *look* broken — that is exactly the problem. They
quietly cost a few points of accuracy and turn into incidents at inference time.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..context import AuditContext
from ..findings import Category, Finding, Severity
from . import register

MISSING_TOKENS = {"", " ", "na", "n/a", "nan", "null", "none", "-", "?", "unknown", "missing", "nil"}


@register("quality.missing_values")
def missing_values(ctx: AuditContext) -> List[Finding]:
    n = len(ctx.X)
    if n == 0:
        return []
    out: List[Finding] = []

    rates = (ctx.X.isna().sum() / n).sort_values(ascending=False)
    bad = rates[rates > 0.05]
    if len(bad):
        top_rate = float(bad.iloc[0])
        out.append(
            Finding(
                check_id="quality.missing_values",
                title=f"{len(bad)} column(s) have substantial missing data",
                category=Category.DATA_QUALITY,
                severity=Severity.HIGH if top_rate > 0.4 else Severity.MEDIUM,
                confidence=0.9,
                plain_english=(
                    f"{len(bad)} columns are missing more than 5% of their values; the worst is "
                    f"'{bad.index[0]}' at {top_rate:.1%}. If those gaps are filled with a single value "
                    "(zero, or the column average) the model can learn 'this was blank' as if it were a "
                    "real measurement."
                ),
                why_it_matters=(
                    "Missingness is often informative — a blank income field means something different "
                    "from an income of zero. Filling it silently either destroys that signal or invents "
                    "one that will not repeat at inference time."
                ),
                suggested_fix=(
                    "Impute inside the pipeline so the same statistics are reused at inference, add an "
                    "explicit `was_missing` indicator, and drop columns missing more than ~60% unless the "
                    "missingness itself is the signal."
                ),
                evidence={"missing_rate_by_column": {c: round(float(v), 4) for c, v in list(bad.items())[:15]}},
                affected_columns=list(bad.index[:15]),
                auto_fixable=True,
            )
        )

    disguised = {}
    for col in ctx.X.columns:
        s = ctx.X[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            vals = s.dropna().astype(str).str.strip().str.lower()
            hits = int(vals.isin(MISSING_TOKENS).sum())
            if hits:
                disguised[col] = hits
    if disguised:
        out.append(
            Finding(
                check_id="quality.disguised_missing_values",
                title="Missing values are hiding as text placeholders",
                category=Category.DATA_QUALITY,
                severity=Severity.MEDIUM,
                confidence=0.8,
                plain_english=(
                    "Some columns store blanks as text such as 'NA', 'unknown' or '?'. Python does not "
                    "treat those as missing, so they are silently encoded as ordinary categories: "
                    + ", ".join(f"{c} ({v:,} rows)" for c, v in list(disguised.items())[:6])
                    + "."
                ),
                why_it_matters="Every downstream missing-value check passes while the data is quietly wrong, and the placeholder becomes a category the model can lean on.",
                suggested_fix="Normalise placeholders to real nulls on load (`pd.read_csv(..., na_values=[...])`) and handle them in the imputation step.",
                evidence={"placeholder_counts": {k: int(v) for k, v in list(disguised.items())[:15]}},
                affected_columns=list(disguised)[:15],
                auto_fixable=True,
            )
        )
    return out


@register("quality.categorical_consistency")
def categorical_consistency(ctx: AuditContext) -> List[Finding]:
    """Label-encoding mismatches waiting to happen between train and inference."""
    out: List[Finding] = []
    n = len(ctx.X)
    fragile, dirty = {}, {}
    for col in ctx.X.columns:
        s = ctx.X[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
            continue
        vals = s.dropna().astype(str)
        if vals.empty:
            continue
        uniq = int(vals.nunique())
        norm_uniq = int(vals.str.strip().str.lower().nunique())
        if norm_uniq < uniq:
            dirty[col] = uniq - norm_uniq
        rare = int((vals.value_counts() == 1).sum())
        if uniq > 20 and rare / uniq > 0.3 and uniq < n * 0.9:
            fragile[col] = {"distinct": uniq, "appearing_once": rare}

    if dirty:
        out.append(
            Finding(
                check_id="quality.inconsistent_category_labels",
                title="The same category is spelled several different ways",
                category=Category.DATA_QUALITY,
                severity=Severity.MEDIUM,
                confidence=0.85,
                plain_english=(
                    "Some categories differ only by capitalisation or stray spaces — "
                    + ", ".join(f"'{c}' ({v} redundant variants)" for c, v in list(dirty.items())[:5])
                    + ". Each spelling is treated as a completely separate category."
                ),
                why_it_matters="Training rows get scattered across duplicate categories, and at inference a differently-cased value arrives as a category the encoder has never seen.",
                suggested_fix="Normalise text columns on ingest (trim and case-fold) and pin the category vocabulary so training and inference agree.",
                evidence={"redundant_variants_by_column": {k: int(v) for k, v in dirty.items()}},
                affected_columns=list(dirty)[:15],
                auto_fixable=True,
            )
        )

    if fragile:
        out.append(
            Finding(
                check_id="quality.high_cardinality_categoricals",
                title="High-cardinality text columns will break at inference",
                category=Category.DATA_QUALITY,
                severity=Severity.MEDIUM,
                confidence=0.7,
                plain_english=(
                    "Columns such as "
                    + ", ".join(
                        f"'{c}' ({v['distinct']} distinct values, {v['appearing_once']} seen only once)"
                        for c, v in list(fragile.items())[:4]
                    )
                    + " contain many one-off values, so anything new in production has no encoding at all."
                ),
                why_it_matters=(
                    "Label encoders fitted on these columns either crash on unseen values or map them to "
                    "an arbitrary integer, producing predictions nobody can explain."
                ),
                suggested_fix=(
                    "Group rare levels into an 'other' bucket, use frequency/target encoding fitted inside "
                    "the cross-validation folds, or use OneHotEncoder(handle_unknown='ignore')."
                ),
                evidence={"columns": fragile},
                affected_columns=list(fragile)[:15],
                auto_fixable=True,
            )
        )
    return out


@register("quality.degenerate_columns")
def degenerate_columns(ctx: AuditContext) -> List[Finding]:
    const = [c for c in ctx.X.columns if ctx.X[c].nunique(dropna=False) <= 1]
    if not const:
        return []
    return [
        Finding(
            check_id="quality.constant_columns",
            title=f"{len(const)} column(s) contain a single value",
            category=Category.DATA_QUALITY,
            severity=Severity.LOW,
            confidence=0.95,
            plain_english=f"These columns never change, so they cannot help the model: {', '.join(const[:10])}.",
            why_it_matters="Dead columns clutter feature-importance reports and cost time on every retrain; a constant column is often the first sign of a broken upstream extract.",
            suggested_fix="Drop them, and check with the data owner whether the column was supposed to be populated.",
            evidence={"constant_columns": const},
            affected_columns=const,
            auto_fixable=True,
        )
    ]


@register("quality.scale_and_outliers")
def scale_and_outliers(ctx: AuditContext) -> List[Finding]:
    num = ctx.X.select_dtypes(include=[np.number])
    if num.shape[1] < 2 or len(num) < 30:
        return []
    extreme = {}
    for col in num.columns:
        s = pd.to_numeric(num[col], errors="coerce").dropna()
        if s.empty or s.nunique() < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        share = float(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean())
        if share > 0.01:
            extreme[col] = round(share, 4)
    if not extreme:
        return []
    return [
        Finding(
            check_id="quality.extreme_outliers",
            title="Several columns contain extreme outliers",
            category=Category.DATA_QUALITY,
            severity=Severity.LOW,
            confidence=0.6,
            plain_english=(
                "More than 1% of values sit far outside the normal range in: "
                + ", ".join(f"{c} ({v:.1%})" for c, v in list(extreme.items())[:6])
                + ". These are often sentinel codes (-1, 9999) rather than real measurements."
            ),
            why_it_matters="Linear and distance-based models are dominated by extreme values, and sentinel codes get treated as enormous genuine quantities.",
            suggested_fix="Check whether the extremes are sentinels; clip or winsorise real outliers and convert sentinel codes to nulls before imputation.",
            evidence={"outlier_share_by_column": extreme},
            affected_columns=list(extreme)[:15],
            auto_fixable=False,
        )
    ]


@register("quality.target_definition")
def target_definition(ctx: AuditContext) -> List[Finding]:
    out: List[Finding] = []
    if ctx.dropped_target_na:
        share = ctx.dropped_target_na / max(len(ctx.df), 1)
        out.append(
            Finding(
                check_id="quality.missing_target_values",
                title=f"{ctx.dropped_target_na:,} rows have no target value",
                category=Category.DATA_QUALITY,
                severity=Severity.MEDIUM if share > 0.05 else Severity.LOW,
                confidence=0.95,
                plain_english=f"{ctx.dropped_target_na:,} rows ({share:.1%}) are missing '{ctx.target}' and were excluded from this audit.",
                why_it_matters="If the outcome is absent for a reason — for example, it has not happened yet — excluding those rows quietly biases the training population.",
                suggested_fix="Establish why the outcome is missing. If the event has simply not occurred yet, exclude that period explicitly or model it as censored data.",
                evidence={"rows_dropped": int(ctx.dropped_target_na), "share": round(share, 4)},
                auto_fixable=False,
            )
        )
    if ctx.task == "classification" and ctx.y.nunique() == 1:
        out.append(
            Finding(
                check_id="quality.single_class_target",
                title="The target has only one value",
                category=Category.DATA_QUALITY,
                severity=Severity.CRITICAL,
                confidence=1.0,
                plain_english=f"Every row has the same value of '{ctx.target}'. There is nothing here to learn.",
                why_it_matters="Any accuracy figure quoted from this data is 100% by construction and completely meaningless.",
                suggested_fix="Check the extract filters — the other class was probably removed upstream.",
                evidence={"distinct_target_values": 1},
                auto_fixable=False,
            )
        )
    return out
