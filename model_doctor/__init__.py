"""Model Doctor — an automated audit toolkit for machine-learning pipelines.

    from model_doctor import audit, write_report

    result = audit("data.csv", target="churned", model="model.pkl", code="train.py")
    print(result.verdict, result.health_score)
    write_report(result, "audit.html")

Works on arbitrary tabular datasets and any scikit-learn-compatible estimator
(logistic regression, random forest, XGBoost, LightGBM, Pipelines).
"""

from .audit import audit
from .findings import AuditResult, Category, Finding, Severity
from .report import to_html, to_markdown, write_report

__version__ = "1.0.0"
__all__ = [
    "audit", "AuditResult", "Finding", "Severity", "Category",
    "to_html", "to_markdown", "write_report", "__version__",
]
