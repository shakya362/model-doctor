"""Run every fixture pipeline through Model Doctor and print a summary table.

    python run_demo.py            # audits all five, writes reports/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model_doctor import audit, write_report
from model_doctor.findings import Severity

CASES = [
    ("pipeline_01", "defaulted", "broken_pipelines/pipeline_01_leaky_preprocessing.py",
     "Predict which loan customers will default"),
    ("pipeline_02", "churned", "broken_pipelines/pipeline_02_duplicate_contamination.py",
     "Predict which subscribers will churn next month"),
    ("pipeline_03", "is_fraud", "broken_pipelines/pipeline_03_imbalance_accuracy.py",
     "Flag fraudulent card transactions in real time"),
    ("pipeline_04", "sales", "broken_pipelines/pipeline_04_temporal_leakage.py",
     "Forecast daily sales per store"),
    ("pipeline_05", "churned", "broken_pipelines/pipeline_05_clean_control.py",
     "Predict which subscribers will churn next month (control: correct pipeline)"),
]


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    rows = []
    for name, target, code, goal in CASES:
        base = Path("artifacts") / name
        result = audit(
            data=base / "data.csv", target=target,
            model=base / "model.joblib", code=code, goal=goal,
        )
        write_report(result, f"reports/{name}.html", title=f"Model audit — {name}")
        write_report(result, f"reports/{name}.md", title=f"Model audit — {name}")
        write_report(result, f"reports/{name}.json")
        counts = {s.value: len(result.by_severity(s)) for s in Severity}
        rows.append((name, result.verdict, result.health_score, counts, result))
        print(f"\n=== {name}: {result.verdict} (health {result.health_score}/100)")
        for f in result.sorted_findings():
            print(f"    [{f.severity.value:<8} {f.confidence:.0%}] {f.check_id}")
        if result.autofix and result.autofix.get("after_headline") is not None:
            fx = result.autofix
            print(f"    -> {fx['headline_metric']}: reported {fx['before_headline']:.4f} "
                  f"| honest {fx['after_headline']:.4f}")
        for e in result.errors:
            print(f"    !! {e}")

    print("\n" + "=" * 78)
    print(f"{'pipeline':<14}{'verdict':<20}{'health':>7}{'crit':>6}{'high':>6}{'med':>6}{'low':>6}")
    for name, verdict, score, c, _ in rows:
        print(f"{name:<14}{verdict:<20}{score:>7}{c['critical']:>6}{c['high']:>6}{c['medium']:>6}{c['low']:>6}")


if __name__ == "__main__":
    main()
