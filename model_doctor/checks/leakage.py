"""Data leakage detection.

Three independent probes:

1. **Target proxy features** — a single feature that alone predicts the target
   almost perfectly. Measured with a shallow decision tree under cross
   validation, so it catches numeric, categorical and non-linear proxies that
   a plain correlation check would miss.
2. **Identifier leakage** — ID-like columns used as features, which either add
   nothing or (worse) carry signal from how the extract was ordered.
3. **Temporal leakage** — on time-indexed data, a random split scoring far
   better than a chronological split means the model is reading the future.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ..context import ID_HINT, AuditContext, fit_score, fresh_like, probe_estimator
from ..findings import Category, Finding, Severity
from . import register

PROXY_STRONG = 0.99
PROXY_SUSPECT = 0.93


def _cv_and_scoring(task: str, y, n_splits: int = 4):
    if task == "classification":
        counts = pd.Series(y).value_counts()
        n_splits = int(max(2, min(n_splits, counts.min())))
        scoring = "roc_auc" if len(counts) == 2 else "roc_auc_ovr"
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0), scoring
    return KFold(n_splits=n_splits, shuffle=True, random_state=0), "r2"


def single_feature_power(ctx: AuditContext, col: str) -> float:
    """Cross-validated predictive power of one feature on its own (0..1)."""
    X = ctx.X_enc[[col]]
    if X[col].nunique() <= 1:
        return 0.0
    cv, scoring = _cv_and_scoring(ctx.task, ctx.y_enc)
    try:
        scores = cross_val_score(
            probe_estimator(ctx.task, depth=4), X, ctx.y_enc,
            cv=cv, scoring=scoring, error_score=np.nan,
        )
        val = float(np.nanmean(scores))
    except Exception:
        return 0.0
    return 0.0 if np.isnan(val) else max(0.0, val)


@register("leakage.target_proxy")
def target_proxy(ctx: AuditContext) -> List[Finding]:
    """Features that individually predict the target near-perfectly."""
    out: List[Finding] = []
    if len(ctx.X_enc) < 30:
        return out

    for col in ctx.features:
        s = ctx.X[col]
        corr = float("nan")
        if ctx.task == "regression" and pd.api.types.is_numeric_dtype(s):
            try:
                corr = float(np.corrcoef(ctx.X_enc[col], ctx.y_enc)[0, 1])
            except Exception:
                corr = float("nan")

        power = single_feature_power(ctx, col)
        strong_corr = bool(corr == corr and abs(corr) > 0.995)
        if power < PROXY_SUSPECT and not strong_corr:
            continue

        exact = False
        try:
            if ctx.task == "regression" and pd.api.types.is_numeric_dtype(s):
                denom = pd.to_numeric(ctx.y, errors="coerce").replace(0, np.nan)
                ratio = pd.to_numeric(s, errors="coerce") / denom
                exact = bool(ratio.dropna().round(9).nunique() == 1)
            elif ctx.task == "classification":
                exact = bool(
                    list(pd.factorize(s.astype(str))[0]) == list(pd.factorize(ctx.y.astype(str))[0])
                )
        except Exception:
            exact = False

        near_perfect = power >= PROXY_STRONG or exact or strong_corr
        severity = Severity.CRITICAL if near_perfect else Severity.HIGH
        confidence = 0.97 if exact else (0.9 if near_perfect else (0.72 if power >= 0.96 else 0.6))
        metric = "AUC" if ctx.task == "classification" else "R²"

        out.append(
            Finding(
                check_id="leakage.target_proxy_feature",
                title=f"Feature '{col}' already contains the answer",
                category=Category.LEAKAGE,
                severity=severity,
                confidence=confidence,
                plain_english=(
                    f"The column '{col}' on its own predicts the outcome almost perfectly "
                    f"({metric} {power:.3f}, using no other information). That means the model is not "
                    "learning a pattern — it is reading a column derived from, or recorded at the same "
                    "time as, the thing you are trying to predict."
                ),
                why_it_matters=(
                    "Test scores look outstanding and then collapse in production, because this column "
                    "will not exist (or will not be filled in the same way) at the moment a real "
                    "prediction has to be made."
                ),
                suggested_fix=(
                    f"Confirm with the data owner when '{col}' is populated relative to the outcome. If "
                    "it is created at or after the outcome is known, drop it and re-train. If it really "
                    "is available beforehand, re-validate on a strictly out-of-time sample."
                ),
                evidence={
                    "single_feature_score": round(power, 4),
                    "metric": metric,
                    "pearson_corr_with_target": None if corr != corr else round(corr, 4),
                    "exact_functional_duplicate_of_target": exact,
                },
                affected_columns=[col],
                auto_fixable=True,
            )
        )
    return out


@register("leakage.identifier")
def identifier_leakage(ctx: AuditContext) -> List[Finding]:
    out: List[Finding] = []
    n = len(ctx.X)
    if n < 30:
        return out
    for col in ctx.features:
        s = ctx.X[col]
        uniq = int(s.nunique(dropna=True))
        ratio = uniq / max(n, 1)
        # a name like store_id on 40 stores across 7,000 rows is a category,
        # not an identifier — require near-uniqueness either way
        looks_id = (bool(ID_HINT.search(col)) and ratio > 0.5) or ratio > 0.9
        if not looks_id or col == ctx.time_column:
            continue
        if pd.api.types.is_float_dtype(s) and not ID_HINT.search(col):
            continue
        power = single_feature_power(ctx, col)
        baseline = 0.5 if ctx.task == "classification" else 0.0
        if power <= baseline + 0.12:
            severity, conf = Severity.LOW, 0.5
            tail = (
                "It carries no signal, so it is harmless noise — but it bloats the model and clutters "
                "feature-importance reports."
            )
        else:
            severity, conf = Severity.HIGH, 0.8
            tail = (
                f"Worryingly it *does* carry signal (score {power:.3f}), which usually means rows were "
                "sorted or numbered by outcome when the data was exported."
            )
        out.append(
            Finding(
                check_id="leakage.identifier_column",
                title=f"Identifier column '{col}' is being used as a feature",
                category=Category.LEAKAGE,
                severity=severity,
                confidence=conf,
                plain_english=f"'{col}' looks like a record identifier ({uniq:,} distinct values across {n:,} rows). {tail}",
                why_it_matters=(
                    "Identifiers are assigned by a system, not earned by behaviour. A model leaning on "
                    "them memorises the training file, and every genuinely new record arrives with a "
                    "value the model has never seen."
                ),
                suggested_fix=f"Remove '{col}' from the feature set — keep it beside the predictions for joining, not as an input.",
                evidence={"distinct_values": uniq, "rows": int(n), "single_feature_score": round(power, 4)},
                affected_columns=[col],
                auto_fixable=True,
            )
        )
    return out


@register("leakage.temporal")
def temporal_leakage(ctx: AuditContext) -> List[Finding]:
    """Random split on time-ordered data lets the model see the future."""
    ts = ctx.time_split()
    if ts is None or len(ctx.X_enc) < 80:
        return []
    Xtr_t, Xte_t, ytr_t, yte_t = ts
    Xtr_r, Xte_r, ytr_r, yte_r = ctx.split()

    try:
        rand_scores = fit_score(fresh_like(None, ctx.task), Xtr_r, ytr_r, Xte_r, yte_r, ctx.task)
        time_scores = fit_score(fresh_like(None, ctx.task), Xtr_t, ytr_t, Xte_t, yte_t, ctx.task)
    except Exception:
        return []

    if ctx.task == "regression":
        key = "r2"
    elif "roc_auc" in rand_scores and "roc_auc" in time_scores:
        key = "roc_auc"
    else:
        key = "f1_macro"
    if key not in rand_scores or key not in time_scores:
        return []

    rnd, tim = float(rand_scores[key]), float(time_scores[key])
    gap = rnd - tim
    if gap < 0.08:
        return []

    confidence = 0.6 if gap < 0.15 else (0.8 if gap < 0.3 else 0.92)
    severity = Severity.HIGH if gap < 0.25 else Severity.CRITICAL
    return [
        Finding(
            check_id="leakage.temporal_split",
            title="The data is time-ordered but is being split randomly",
            category=Category.LEAKAGE,
            severity=severity,
            confidence=confidence,
            plain_english=(
                f"This dataset has a time column ('{ctx.time_column}'). Shuffling the rows at random, the "
                f"model scores {rnd:.3f}. Training on the earlier period and testing on the later one — "
                f"which is what actually happens in production — it scores {tim:.3f}. The shuffled number "
                "is the one that is wrong."
            ),
            why_it_matters=(
                "A random shuffle puts tomorrow's records into training and yesterday's into testing. The "
                "model gets to peek at the future, so the reported accuracy measures something that can "
                "never be repeated live."
            ),
            suggested_fix=(
                f"Split chronologically on '{ctx.time_column}' (train on the past, test on the most recent "
                "window) and use TimeSeriesSplit instead of KFold. Also check any rolling or aggregate "
                "features for being computed across the whole series."
            ),
            evidence={
                "time_column": ctx.time_column,
                "random_split_score": round(rnd, 4),
                "chronological_split_score": round(tim, 4),
                "metric": key,
                "optimism_gap": round(gap, 4),
            },
            affected_columns=[ctx.time_column] if ctx.time_column else [],
            auto_fixable=True,
        )
    ]


@register("leakage.future_window")
def future_window_feature(ctx: AuditContext) -> List[Finding]:
    """Features built from a window that includes the present or the future.

    A legitimate lagged feature knows much more about *past* target values than
    about future ones. A centred or forward-looking window knows both equally
    well. Comparing the correlation profile either side of each row separates
    the two without needing to read the feature-engineering code.
    """
    if ctx.time_column is None or ctx.task not in ("regression", "classification"):
        return []
    if len(ctx.X) < 150:
        return []

    from .contamination import _candidate_group_columns

    order = pd.to_datetime(ctx.X[ctx.time_column], errors="coerce")
    if order.isna().mean() > 0.5:
        order = pd.to_numeric(ctx.X[ctx.time_column], errors="coerce")
    if order.isna().all():
        return []

    group_candidates = [c for c in _candidate_group_columns(ctx) if c != ctx.time_column]
    group = ctx.X[group_candidates[0]].astype(str) if group_candidates else pd.Series("all", index=ctx.X.index)

    frame = pd.DataFrame({"__g": group.to_numpy(), "__t": order.to_numpy(), "__y": ctx.y_enc})
    numeric = [
        c for c in ctx.features
        if c != ctx.time_column and pd.api.types.is_numeric_dtype(ctx.X[c]) and ctx.X[c].nunique() > 5
    ]
    if not numeric:
        return []
    for c in numeric:
        frame[c] = pd.to_numeric(ctx.X[c], errors="coerce").to_numpy()
    frame = frame.sort_values(["__g", "__t"]).reset_index(drop=True)
    if frame.groupby("__g").size().median() < 20:
        return []

    # remove between-entity level differences, which otherwise swamp the profile
    y_dm = frame["__y"] - frame.groupby("__g")["__y"].transform("mean")
    out: List[Finding] = []

    for col in numeric:
        f_dm = frame[col] - frame.groupby("__g")[col].transform("mean")
        if f_dm.std(skipna=True) in (0, None) or not np.isfinite(f_dm.std(skipna=True)) or f_dm.std(skipna=True) == 0:
            continue
        profile = {}
        for k in (-3, -2, -1, 0, 1, 2, 3):
            shifted = y_dm.groupby(frame["__g"]).shift(-k)
            mask = f_dm.notna() & shifted.notna()
            if mask.sum() < 50:
                profile[k] = 0.0
                continue
            try:
                profile[k] = float(np.corrcoef(f_dm[mask], shifted[mask])[0, 1])
            except Exception:
                profile[k] = 0.0
        profile = {k: (0.0 if v != v else v) for k, v in profile.items()}

        best_past = max(abs(profile[-1]), abs(profile[-2]), abs(profile[-3]))
        best_future = max(abs(profile[1]), abs(profile[2]), abs(profile[3]))
        now = abs(profile[0])
        if best_future < 0.15 or now < 0.15:
            continue
        if best_past <= 0 or best_future / best_past < 0.8:
            continue

        out.append(
            Finding(
                check_id="leakage.future_window_feature",
                title=f"Feature '{col}' appears to be built from a window including the future",
                category=Category.LEAKAGE,
                severity=Severity.CRITICAL,
                confidence=0.62,
                plain_english=(
                    f"'{col}' tells us as much about what happens *after* each row as about what happened "
                    f"before it (correlation {best_future:.2f} looking forward versus {best_past:.2f} "
                    "looking back). A figure computed only from history cannot do that. This one looks "
                    "like a rolling or aggregate value whose window includes the day being predicted, and "
                    "the days after it."
                ),
                why_it_matters=(
                    "At prediction time those future days have not happened, so this column cannot be "
                    "filled in the same way. Whatever accuracy it is buying today disappears completely "
                    "the moment the model is used for a real forecast."
                ),
                suggested_fix=(
                    f"Recompute '{col}' using only completed periods — for a rolling mean that means "
                    "`.shift(1).rolling(window)` within each entity, never `center=True` — then re-measure. "
                    "Confirm with whoever built the feature which days its window covers."
                ),
                evidence={
                    "correlation_profile_by_offset": {str(k): round(v, 4) for k, v in profile.items()},
                    "best_correlation_with_past_targets": round(best_past, 4),
                    "best_correlation_with_future_targets": round(best_future, 4),
                    "grouped_by": group_candidates[0] if group_candidates else None,
                    "note": "correlations are measured within entity, after removing each entity's mean",
                },
                affected_columns=[col],
                auto_fixable=True,
            )
        )
    return out
