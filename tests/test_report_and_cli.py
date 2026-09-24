import json

from model_doctor import audit, to_html, to_markdown, write_report
from model_doctor.cli import main


def test_html_report_is_self_contained_and_readable(leaky_classification, tmp_path):
    result = audit(leaky_classification, target="y")
    html = to_html(result, title="Client audit", client="Acme Ltd")
    assert html.startswith("<!doctype html>")
    assert "<script" not in html and "http://" not in html   # no external dependencies
    assert "Acme Ltd" in html and result.verdict in html
    assert "settlement_flag" in html
    assert "Why it matters" in html and "Recommended fix" in html


def test_markdown_and_json_outputs(clean_classification, tmp_path):
    result = audit(clean_classification, target="y", autofix=False)
    assert "# " in to_markdown(result)
    p = write_report(result, tmp_path / "r.json")
    payload = json.loads(p.read_text())
    assert {"verdict", "health_score", "findings", "dataset_summary"} <= set(payload)


def test_health_score_and_verdict_track_severity(leaky_classification, clean_classification):
    bad = audit(leaky_classification, target="y", autofix=False)
    good = audit(clean_classification, target="y", autofix=False)
    assert bad.health_score < good.health_score
    assert good.verdict in {"Healthy", "Needs attention"}


def test_autofix_reports_before_and_after(leaky_classification):
    result = audit(leaky_classification, target="y", autofix=True)
    fix = result.autofix
    assert fix is not None
    assert fix["after_headline"] < fix["before_headline"]   # optimism removed
    assert "settlement_flag" in fix["dropped_columns"]
    assert fix["actions"]


def test_cli_end_to_end(leaky_classification, tmp_path):
    data = tmp_path / "data.csv"
    leaky_classification.to_csv(data, index=False)
    out = tmp_path / "audit.html"
    code = main(["--data", str(data), "--target", "y", "--out", str(out),
                 "--json", str(tmp_path / "audit.json"), "--quiet"])
    assert code == 0
    assert out.exists() and out.stat().st_size > 2000
    assert (tmp_path / "audit.json").exists()


def test_cli_fail_on_severity_for_ci(leaky_classification, tmp_path):
    data = tmp_path / "data.csv"
    leaky_classification.to_csv(data, index=False)
    code = main(["--data", str(data), "--target", "y", "--out", str(tmp_path / "a.md"),
                 "--fail-on", "high", "--quiet", "--no-autofix"])
    assert code == 1
