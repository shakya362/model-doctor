"""Data model for audit findings.

Every check in Model Doctor returns zero or more `Finding` objects. A finding is
deliberately written for two audiences at once: `plain_english` is what a
non-technical client reads, `evidence` is what an engineer verifies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    CRITICAL = "critical"   # results are not trustworthy at all; do not ship
    HIGH = "high"           # results are materially overstated
    MEDIUM = "medium"       # real risk to production performance
    LOW = "low"             # hygiene issue, worth fixing

    @property
    def rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}[self.value]


class Category(str, Enum):
    LEAKAGE = "Data leakage"
    CONTAMINATION = "Train/test contamination"
    METRICS = "Misleading metrics"
    OVERFITTING = "Overfitting"
    DATA_QUALITY = "Data quality"
    IMBALANCE = "Class imbalance blindness"
    CODE = "Pipeline code smells"


@dataclass
class Finding:
    check_id: str                 # stable id, e.g. "leakage.target_proxy_feature"
    title: str                    # short headline
    category: Category
    severity: Severity
    confidence: float             # 0..1 — how sure we are this is real, not a false positive
    plain_english: str            # what a non-technical client reads
    why_it_matters: str           # business/technical consequence
    suggested_fix: str            # concrete remedy
    evidence: Dict[str, Any] = field(default_factory=dict)
    affected_columns: List[str] = field(default_factory=list)
    auto_fixable: bool = False

    def __post_init__(self) -> None:
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    @property
    def priority(self) -> float:
        """Ranking score used to sort the report: severity weighted by confidence."""
        return self.severity.rank * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        d["priority"] = round(self.priority, 3)
        return d


@dataclass
class AuditResult:
    findings: List[Finding] = field(default_factory=list)
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    model_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    autofix: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)

    def sorted_findings(self) -> List[Finding]:
        return sorted(self.findings, key=lambda f: (-f.priority, f.check_id))

    def by_severity(self, severity: Severity) -> List[Finding]:
        return [f for f in self.findings if f.severity is severity]

    def has(self, check_id: str) -> bool:
        """Used heavily by the test-suite: did the auditor catch this bug?"""
        return any(f.check_id == check_id for f in self.findings)

    @property
    def health_score(self) -> int:
        """0-100. Starts at 100; each finding deducts severity x confidence points."""
        weights = {"critical": 34.0, "high": 18.0, "medium": 8.0, "low": 3.0}
        penalty = sum(weights[f.severity.value] * f.confidence for f in self.findings)
        return int(max(0, round(100 - penalty)))

    @property
    def verdict(self) -> str:
        if any(f.severity is Severity.CRITICAL and f.confidence >= 0.6 for f in self.findings):
            return "Not safe to deploy"
        score = self.health_score
        if score >= 85:
            return "Healthy"
        if score >= 60:
            return "Needs attention"
        return "Not safe to deploy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "health_score": self.health_score,
            "dataset_summary": self.dataset_summary,
            "model_summary": self.model_summary,
            "metrics": self.metrics,
            "findings": [f.to_dict() for f in self.sorted_findings()],
            "autofix": self.autofix,
            "errors": self.errors,
        }
