"""Auto-fix engine.

Takes the findings, applies the mechanical remedies, and re-measures. The
"before" number is what the flawed protocol *reports*; the "after" number is
what the same model family achieves once the protocol is honest. The gap
between them is the whole point of the audit — it is the number the client has
been budgeting against.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .context import (
    RANDOM_STATE,
    AuditContext,
    encode_matrix,
    encode_target,
    fresh_like,
    pick_headline,
    score_predictions,
)
from .findings import Finding

DROPPABLE = {
    "leakage.target_proxy_feature",
    "leakage.future_window_feature",
    "leakage.identifier_column",
    "quality.constant_columns",
}


def _fit_and_score(est, X_tr, y_tr, X_te, y_te, task: str) -> Dict[str, float]:
    est.fit(X_tr, y_tr)
    proba = None
    if hasattr(est, "predict_proba"):
        try:
            proba = est.predict_proba(X_te)
        except Exception:
            proba = None
    return score_predictions(y_te, est.predict(X_te), proba, task)


def _balanced(est, task: str):
    """Apply balanced class weighting where the estimator supports it."""
    if task != "classification":
        return est, False
    try:
        params = est.get_params()
    except Exception:
        return est, False
    if "class_weight" in params and params.get("class_weight") is None:
        try:
            est.set_params(class_weight="balanced")
            return est, True
        except Exception:
            return est, False
    return est, False


def run_autofix(ctx: AuditContext, findings: List[Finding]) -> Optional[Dict[str, Any]]:
    ids = {f.check_id for f in findings}
    drop_cols: List[str] = []
    for f in findings:
        if f.check_id in DROPPABLE:
            drop_cols.extend(c for c in f.affected_columns if c in ctx.X.columns)
    drop_cols = sorted(set(drop_cols))

    dedupe = "contamination.duplicate_rows" in ids
    use_time = "leakage.temporal_split" in ids or "code.random_split_on_time_series" in ids
    group_finding = next((f for f in findings if f.check_id == "contamination.group_leakage"), None)
    rebalance = bool({"imbalance.skewed_target", "imbalance.model_ignores_minority_class",
                      "metrics.accuracy_on_imbalanced_classes"} & ids)

    if not (drop_cols or dedupe or use_time or group_finding or rebalance):
        return None

    actions: List[str] = []
    try:
        # ---------------- before: the flawed protocol, as it stands ------- #
        Xb, yb = ctx.X_enc, ctx.y_enc
        strat = yb if (ctx.task == "classification" and pd.Series(yb).value_counts().min() >= 2) else None
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
            Xb, yb, test_size=0.25, random_state=RANDOM_STATE, stratify=strat
        )
        before = _fit_and_score(fresh_like(ctx.model, ctx.task), Xb_tr, yb_tr, Xb_te, yb_te, ctx.task)

        # ---------------- after: the corrected protocol -------------------- #
        X_raw, y_raw = ctx.X.copy(), ctx.y.copy()
        if dedupe:
            full = pd.concat([X_raw, y_raw.rename("__target__")], axis=1)
            keep = ~full.duplicated(keep="first")
            removed = int((~keep).sum())
            X_raw, y_raw = X_raw.loc[keep].reset_index(drop=True), y_raw.loc[keep].reset_index(drop=True)
            actions.append(f"Removed {removed:,} duplicate rows before splitting")

        groups = None
        if group_finding:
            gcol = group_finding.evidence.get("group_column")
            if gcol in X_raw.columns:
                groups = X_raw[gcol].astype(str).to_numpy()

        time_vals = None
        if use_time and ctx.time_column in X_raw.columns:
            tv = pd.to_datetime(X_raw[ctx.time_column], errors="coerce")
            if tv.isna().mean() > 0.5:
                tv = pd.to_numeric(X_raw[ctx.time_column], errors="coerce")
            time_vals = tv

        if drop_cols:
            X_raw = X_raw.drop(columns=[c for c in drop_cols if c in X_raw.columns])
            actions.append("Dropped leaking/identifier columns: " + ", ".join(drop_cols))

        Xa = encode_matrix(X_raw)
        ya = encode_target(y_raw, ctx.task)

        if time_vals is not None and len(Xa) > 40:
            order = np.argsort(time_vals.fillna(time_vals.min()).to_numpy(), kind="stable")
            cut = int(len(order) * 0.75)
            tr_idx, te_idx = order[:cut], order[cut:]
            actions.append(f"Split chronologically on '{ctx.time_column}' (train on the past, test on the most recent 25%)")
        elif groups is not None and len(np.unique(groups)) >= 4:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
            tr_idx, te_idx = next(gss.split(Xa, ya, groups=groups))
            actions.append(f"Held out whole groups of '{group_finding.evidence.get('group_column')}' instead of individual rows")
        else:
            strat_a = ya if (ctx.task == "classification" and pd.Series(ya).value_counts().min() >= 2) else None
            idx = np.arange(len(Xa))
            tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=RANDOM_STATE, stratify=strat_a)
            if strat_a is not None:
                actions.append("Used a stratified split so the rare class is represented in both sides")

        est = fresh_like(ctx.model, ctx.task)
        if rebalance:
            est, applied = _balanced(est, ctx.task)
            if applied:
                actions.append("Re-trained with class_weight='balanced' so the rare class is not ignored")

        after = _fit_and_score(
            est, Xa.iloc[tr_idx], ya[tr_idx], Xa.iloc[te_idx], ya[te_idx], ctx.task
        )
    except Exception as exc:  # pragma: no cover - defensive
        ctx.notes.append(f"auto-fix could not complete: {exc}")
        return None

    key, before_val = pick_headline(before, ctx.task)
    after_val = after.get(key, float("nan"))
    return {
        "headline_metric": key,
        "before": {k: round(float(v), 4) for k, v in before.items()},
        "after": {k: round(float(v), 4) for k, v in after.items()},
        "before_headline": round(float(before_val), 4),
        "after_headline": None if after_val != after_val else round(float(after_val), 4),
        "delta": None if after_val != after_val else round(float(after_val - before_val), 4),
        "actions": actions,
        "dropped_columns": drop_cols,
        "estimator": type(fresh_like(ctx.model, ctx.task)).__name__,
    }
