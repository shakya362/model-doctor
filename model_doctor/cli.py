"""Command-line interface.

    model-doctor --data data.csv --target churned \
                 --model model.pkl --code train.py \
                 --out reports/audit.html --goal "Flag customers likely to churn"

Works on any .pkl/.joblib model plus any CSV — nothing is hardcoded to a demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit
from .findings import Severity
from .report import to_markdown, write_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model-doctor",
        description="Audit a trained model / training pipeline / dataset for common ML failure patterns.",
    )
    p.add_argument("--data", required=True, help="dataset (csv/tsv/parquet/xlsx/json)")
    p.add_argument("--target", required=True, help="name of the column being predicted")
    p.add_argument("--model", help="trained model (.pkl/.joblib) — optional")
    p.add_argument("--code", help="training script (.py/.ipynb) for static analysis — optional")
    p.add_argument("--goal", help="the stated prediction task/goal, echoed in the report")
    p.add_argument("--task", choices=["classification", "regression"], help="override task inference")
    p.add_argument("--time-column", help="name of the time column, if the data is time-ordered")
    p.add_argument("--group-column", help="entity column that must not straddle the split")
    p.add_argument("--out", default="audit_report.html", help="output path (.html/.md/.json)")
    p.add_argument("--json", dest="json_out", help="additionally write machine-readable JSON here")
    p.add_argument("--no-autofix", action="store_true", help="skip the fix-and-remeasure step")
    p.add_argument("--only", nargs="*", help="run only checks whose name starts with these prefixes")
    p.add_argument("--quiet", action="store_true", help="suppress the console summary")
    p.add_argument(
        "--fail-on", choices=["critical", "high", "medium", "low", "never"], default="never",
        help="exit non-zero if an issue of this severity or worse is found (for CI)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = audit(
        data=args.data,
        target=args.target,
        model=args.model,
        code=args.code,
        goal=args.goal,
        task=args.task,
        time_column=args.time_column,
        group_column=args.group_column,
        autofix=not args.no_autofix,
        only=args.only,
    )

    out = write_report(result, args.out, title=f"Model audit — {Path(args.data).stem}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

    if not args.quiet:
        print(f"\nVerdict: {result.verdict}   (health score {result.health_score}/100)")
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
            items = result.by_severity(sev)
            for f in items:
                print(f"  [{sev.value:<8}] {f.title}   (confidence {f.confidence:.0%})")
        if result.autofix and result.autofix.get("after_headline") is not None:
            fx = result.autofix
            print(f"\n  {fx['headline_metric']}: reported {fx['before_headline']:.4f} "
                  f"-> honest {fx['after_headline']:.4f}")
        print(f"\nReport written to {out}")

    if args.fail_on != "never":
        threshold = Severity(args.fail_on).rank
        if any(f.severity.rank >= threshold for f in result.findings):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
