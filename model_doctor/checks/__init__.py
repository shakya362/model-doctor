"""Check registry.

A check is a callable ``(AuditContext) -> List[Finding]``. Registering rather
than hardcoding means the auditor is extensible: a new failure pattern is one
decorated function, with no change to the orchestrator.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ..context import AuditContext
from ..findings import Finding

REGISTRY: Dict[str, Callable[[AuditContext], List[Finding]]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def load_all() -> Dict[str, Callable[[AuditContext], List[Finding]]]:
    from . import (  # noqa: F401
        code_smells,
        contamination,
        data_quality,
        imbalance,
        inference,
        leakage,
        metrics,
        overfitting,
    )
    return REGISTRY
