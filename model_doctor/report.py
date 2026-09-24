"""Audit report rendering.

The report is the deliverable a client actually reads, so it is written as a
clinical chart rather than a log dump: a verdict, the vital signs, then one
entry per problem in the order a practitioner would work through them. Every
entry answers three questions in plain English — what is wrong, why it costs
money, what to do about it — with the technical evidence folded away
underneath for the client's own engineers.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .findings import AuditResult, Finding, Severity

SEV_LABEL = {
    Severity.CRITICAL: "Critical",
    Severity.HIGH: "High",
    Severity.MEDIUM: "Medium",
    Severity.LOW: "Low",
}
SEV_COLOR = {
    Severity.CRITICAL: "--sev-critical",
    Severity.HIGH: "--sev-high",
    Severity.MEDIUM: "--sev-medium",
    Severity.LOW: "--sev-low",
}


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _confidence_words(c: float) -> str:
    if c >= 0.9:
        return "near-certain"
    if c >= 0.75:
        return "confident"
    if c >= 0.55:
        return "probable"
    return "worth checking"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v[:12]) + ("…" if len(v) > 12 else "")
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_fmt(x)}" for k, x in list(v.items())[:12])
    return str(v)


# --------------------------------------------------------------------------- #
# executive summary
# --------------------------------------------------------------------------- #
def executive_summary(result: AuditResult) -> str:
    n = len(result.findings)
    crit = len(result.by_severity(Severity.CRITICAL))
    high = len(result.by_severity(Severity.HIGH))
    ds = result.dataset_summary
    task = ds.get("task", "prediction")
    target = ds.get("target", "the target")

    if n == 0:
        return (
            f"We examined this {task} pipeline for the fifteen failure patterns that most often make a "
            f"model look better in testing than it performs in production. None of them were found. The "
            f"measured performance on data the model had never seen is, on this evidence, a fair estimate "
            f"of what it will do live."
        )

    lead = (
        f"We examined this {task} pipeline — predicting '{target}' from {ds.get('rows', 0):,} rows — for "
        f"the failure patterns that make a model look better in testing than it performs in production. "
    )
    if crit:
        lead += (
            f"{crit} critical problem{'s' if crit != 1 else ''} "
            f"{'were' if crit != 1 else 'was'} found. The performance figures currently attached to this "
            "model should not be relied on for any business decision until they are fixed and the model "
            "is re-measured."
        )
    elif high:
        lead += (
            f"No outright blockers, but {high} serious issue{'s' if high != 1 else ''} "
            f"{'mean' if high != 1 else 'means'} the reported performance is overstated and will not "
            "reproduce on new data."
        )
    else:
        lead += (
            f"Nothing structural is broken. {n} smaller issue{'s' if n != 1 else ''} "
            f"{'are' if n != 1 else 'is'} costing accuracy and robustness and should be tidied up before "
            "the next retrain."
        )

    fix = result.autofix
    if fix and fix.get("after_headline") is not None:
        metric = fix["headline_metric"]
        before, after = fix["before_headline"], fix["after_headline"]
        drop = before - after
        if drop > 0.02:
            lead += (
                f" To put a number on it: the current setup reports {metric} of {before:.3f}. After "
                f"correcting the issues below and measuring honestly, the same model family achieves "
                f"{after:.3f} — the difference of {drop:.3f} is optimism, not performance."
            )
        elif after - before > 0.02:
            lead += (
                f" Applying the recommended fixes also improved genuine performance: {metric} moved from "
                f"{before:.3f} to {after:.3f}."
            )
    return lead


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def to_markdown(result: AuditResult, title: str = "Model audit report") -> str:
    ds, out = result.dataset_summary, []
    out.append(f"# {title}\n")
    out.append(f"*Generated {datetime.now():%d %B %Y, %H:%M} by Model Doctor*\n")
    out.append(f"**Verdict: {result.verdict}** · Model health score {result.health_score}/100\n")
    if ds.get("goal"):
        out.append(f"> Stated goal: {ds['goal']}\n")
    out.append("## What we found\n")
    out.append(executive_summary(result) + "\n")

    counts = {s: len(result.by_severity(s)) for s in Severity}
    out.append("| Severity | Issues |\n|---|---|")
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        out.append(f"| {SEV_LABEL[sev]} | {counts[sev]} |")
    out.append("")

    fix = result.autofix
    if fix and fix.get("after_headline") is not None:
        out.append("## Before and after\n")
        out.append(f"| | {fix['headline_metric']} |\n|---|---|")
        out.append(f"| As it stands today (reported) | {fix['before_headline']:.4f} |")
        out.append(f"| After the fixes, measured honestly | {fix['after_headline']:.4f} |")
        out.append("")
        if fix["actions"]:
            out.append("Fixes applied automatically:\n")
            out.extend(f"- {a}" for a in fix["actions"])
            out.append("")

    out.append("## Issues found\n")
    for i, f in enumerate(result.sorted_findings(), 1):
        out.append(f"### {i}. {f.title}")
        out.append(
            f"*{SEV_LABEL[f.severity]} · {f.category.value} · confidence {f.confidence:.0%} "
            f"({_confidence_words(f.confidence)})*\n"
        )
        out.append(f"{f.plain_english}\n")
        out.append(f"**Why it matters.** {f.why_it_matters}\n")
        out.append(f"**Recommended fix.** {f.suggested_fix}\n")
        if f.evidence:
            out.append("<details><summary>Technical evidence</summary>\n")
            out.append("```json")
            out.append(json.dumps(f.evidence, indent=2, default=str))
            out.append("```\n</details>\n")

    out.append("## Dataset and model examined\n")
    for k, v in ds.items():
        if v not in (None, {}, []):
            out.append(f"- **{k.replace('_', ' ')}**: {_fmt(v)}")
    if result.model_summary.get("type"):
        out.append(f"- **model**: {result.model_summary['type']}")
    if result.metrics.get("holdout"):
        h = result.metrics["holdout"]
        out.append(f"- **held-out measurement**: {h['estimator']} — train {_fmt(h['train'])} / test {_fmt(h['test'])}")
    if result.errors:
        out.append("\n### Notes\n")
        out.extend(f"- {_short(e)}" for e in result.errors)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --paper:#f4f6f7; --card:#ffffff; --ink:#151b21; --ink-soft:#4a565f; --rule:#d5dde1;
  --sev-critical:#b3261e; --sev-high:#b4641a; --sev-medium:#7a6a1d; --sev-low:#3f6b73;
  --ok:#2b6a4c; --accent:#134b63;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}
.sheet{max-width:1000px;margin:0 auto;padding:40px 28px 80px}
header.chart{background:var(--card);border:1px solid var(--rule);border-top:5px solid var(--accent);padding:28px 30px}
.masthead{display:flex;justify-content:space-between;align-items:baseline;gap:20px;flex-wrap:wrap;
  border-bottom:1px solid var(--rule);padding-bottom:12px;margin-bottom:22px}
.masthead h1{margin:0;font-size:22px;font-weight:650;letter-spacing:-.01em}
.masthead .meta{color:var(--ink-soft);font-size:13px}
.verdict-row{display:grid;grid-template-columns:1.25fr 1fr;gap:34px;align-items:start}
.verdict h2{font:400 34px/1.15 Georgia,"Iowan Old Style","Times New Roman",serif;margin:0 0 10px;letter-spacing:-.015em}
.verdict h2.bad{color:var(--sev-critical)} .verdict h2.warn{color:var(--sev-high)} .verdict h2.ok{color:var(--ok)}
.verdict p{margin:0;color:var(--ink-soft);font-size:14.5px;max-width:52ch}
.vitals{border-left:1px solid var(--rule);padding-left:26px}
.vitals dl{display:grid;grid-template-columns:auto 1fr;gap:7px 16px;margin:0;font-size:14px}
.vitals dt{color:var(--ink-soft)} .vitals dd{margin:0;font-variant-numeric:tabular-nums;font-weight:600}
.score{font:600 40px/1 ui-sans-serif,sans-serif;font-variant-numeric:tabular-nums}
.score small{font-size:15px;font-weight:400;color:var(--ink-soft)}
section{background:var(--card);border:1px solid var(--rule);border-top:none;padding:26px 30px}
section h3{margin:0 0 14px;font-size:13px;font-weight:700;color:var(--accent)}
.summary{font:400 17px/1.65 Georgia,"Iowan Old Style","Times New Roman",serif;max-width:68ch;margin:0}
.tally{display:flex;gap:0;margin:18px 0 0;border:1px solid var(--rule)}
.tally div{flex:1;padding:11px 14px;border-right:1px solid var(--rule)}
.tally div:last-child{border-right:none}
.tally b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.tally span{font-size:12.5px;color:var(--ink-soft)}
table.ba{width:100%;border-collapse:collapse;font-size:15px;margin-top:6px}
table.ba th,table.ba td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--rule)}
table.ba td.num{font-variant-numeric:tabular-nums;font-weight:600;text-align:right;width:130px}
table.ba tr.after td{color:var(--ok)}
ul.actions{margin:14px 0 0;padding-left:18px;font-size:14.5px;color:var(--ink-soft)}
article.finding{border-left:4px solid var(--sev);padding:18px 0 18px 20px;border-bottom:1px solid var(--rule)}
article.finding:last-of-type{border-bottom:none;padding-bottom:4px}
.f-head{display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap}
.f-head h4{margin:0;font-size:18px;font-weight:620;letter-spacing:-.01em}
.tag{font-size:12px;font-weight:700;color:var(--sev);white-space:nowrap}
.f-meta{font-size:12.5px;color:var(--ink-soft);margin:4px 0 10px}
.finding p{margin:0 0 9px;max-width:74ch;font-size:15px}
.finding p.label b{color:var(--ink)}
details{margin-top:10px;font-size:13.5px}
summary{cursor:pointer;color:var(--accent);font-weight:600}
pre{background:#eef2f4;border:1px solid var(--rule);padding:12px;overflow-x:auto;font-size:12.5px;margin:9px 0 0}
dl.facts{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;margin:0;font-size:14px}
dl.facts dt{color:var(--ink-soft)} dl.facts dd{margin:0}
footer{color:var(--ink-soft);font-size:12.5px;padding:18px 2px;max-width:74ch}
@media (max-width:760px){
  .verdict-row{grid-template-columns:1fr}
  .vitals{border-left:none;border-top:1px solid var(--rule);padding:18px 0 0}
  .tally{flex-wrap:wrap} .tally div{flex:1 1 50%}
  .sheet{padding:18px 14px 50px} header.chart,section{padding:20px 18px}
}
@media print{
  body{background:#fff} .sheet{padding:0;max-width:none}
  header.chart,section{border:none;page-break-inside:avoid} details{display:none}
  article.finding{page-break-inside:avoid}
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){--paper:#11161a;--card:#171e24;--ink:#e7edf1;--ink-soft:#9fb0bb;
    --rule:#2b363e;--accent:#79b6d2;--ok:#5fbf90;--sev-critical:#ef6b61;--sev-high:#e0a04e;
    --sev-medium:#cbbc5e;--sev-low:#7fb3bd}
  :root:not([data-theme="light"]) pre{background:#10161b}
}
:root[data-theme="dark"]{--paper:#11161a;--card:#171e24;--ink:#e7edf1;--ink-soft:#9fb0bb;
  --rule:#2b363e;--accent:#79b6d2;--ok:#5fbf90;--sev-critical:#ef6b61;--sev-high:#e0a04e;
  --sev-medium:#cbbc5e;--sev-low:#7fb3bd}
"""


def _finding_html(f: Finding, index: int) -> str:
    var = SEV_COLOR[f.severity]
    ev = (
        "<details><summary>Technical evidence</summary><pre>"
        + html.escape(json.dumps(f.evidence, indent=2, default=str))
        + f"\n\ncheck id: {html.escape(f.check_id)}</pre></details>"
        if f.evidence
        else ""
    )
    cols = ""
    if f.affected_columns:
        cols = (
            '<p class="label"><b>Columns involved.</b> '
            + html.escape(", ".join(f.affected_columns[:12]))
            + ("…" if len(f.affected_columns) > 12 else "")
            + "</p>"
        )
    return f"""
    <article class="finding" style="--sev:var({var})">
      <div class="f-head">
        <h4>{index}. {html.escape(f.title)}</h4>
        <span class="tag">{SEV_LABEL[f.severity]}</span>
      </div>
      <p class="f-meta">{html.escape(f.category.value)} · confidence {f.confidence:.0%}
         ({_confidence_words(f.confidence)}){' · fix applied automatically' if f.auto_fixable else ''}</p>
      <p>{html.escape(f.plain_english)}</p>
      <p class="label"><b>Why it matters.</b> {html.escape(f.why_it_matters)}</p>
      <p class="label"><b>Recommended fix.</b> {html.escape(f.suggested_fix)}</p>
      {cols}
      {ev}
    </article>"""


def to_html(result: AuditResult, title: str = "Model audit report", client: Optional[str] = None) -> str:
    ds = result.dataset_summary
    verdict_class = {"Healthy": "ok", "Needs attention": "warn"}.get(result.verdict, "bad")
    counts = {s: len(result.by_severity(s)) for s in Severity}

    vitals = [
        ("Rows examined", f"{ds.get('rows', 0):,}"),
        ("Features", str(ds.get("features", "—"))),
        ("Task", (ds.get("task") or "—").title()),
        ("Target", ds.get("target", "—")),
    ]
    if result.metrics.get("holdout"):
        h = result.metrics["holdout"]
        key = result.metrics.get("headline_metric", "accuracy")
        vitals.append((f"Held-out {key}", f"{h['test'].get(key, float('nan')):.3f}"))
        vitals.append(("Model examined", result.model_summary.get("type") or h["estimator"]))
    vitals_html = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in vitals)

    tally = "".join(
        f'<div><b style="color:var({SEV_COLOR[s]})">{counts[s]}</b><span>{SEV_LABEL[s]}</span></div>'
        for s in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW)
    )

    ba_html = ""
    fix = result.autofix
    if fix and fix.get("after_headline") is not None:
        acts = "".join(f"<li>{html.escape(a)}</li>" for a in fix["actions"])
        ba_html = f"""
    <section>
      <h3>Before and after the fixes</h3>
      <table class="ba">
        <tr><th>Measurement</th><th style="text-align:right">{html.escape(fix['headline_metric'])}</th></tr>
        <tr><td>The pipeline as it stands today, scored the way it scores itself</td>
            <td class="num">{fix['before_headline']:.4f}</td></tr>
        <tr class="after"><td>The same model after the fixes below, measured honestly</td>
            <td class="num">{fix['after_headline']:.4f}</td></tr>
      </table>
      {'<ul class="actions">' + acts + '</ul>' if acts else ''}
    </section>"""

    findings_html = "".join(_finding_html(f, i) for i, f in enumerate(result.sorted_findings(), 1))
    if not findings_html:
        findings_html = "<p>No issues were detected by any of the checks in this suite.</p>"

    facts = "".join(
        f"<dt>{html.escape(k.replace('_', ' '))}</dt><dd>{html.escape(_fmt(v))}</dd>"
        for k, v in ds.items() if v not in (None, {}, [])
    )
    if result.metrics.get("holdout"):
        h = result.metrics["holdout"]
        facts += (
            f"<dt>held-out measurement</dt><dd>{html.escape(h['estimator'])} — "
            f"train {html.escape(_fmt(h['train']))} / test {html.escape(_fmt(h['test']))}</dd>"
        )
    if result.metrics.get("supplied_model"):
        facts += f"<dt>supplied model scored on this data</dt><dd>{html.escape(_fmt(result.metrics['supplied_model']))}</dd>"

    notes = ""
    if result.errors:
        notes = "<h3 style='margin-top:20px'>Notes</h3><ul class='actions'>" + "".join(
            f"<li>{html.escape(_short(e))}</li>" for e in result.errors
        ) + "</ul>"

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><div class="sheet">
  <header class="chart">
    <div class="masthead">
      <h1>{html.escape(title)}</h1>
      <div class="meta">{html.escape(client) + ' · ' if client else ''}{datetime.now():%d %B %Y}
        · prepared with Model Doctor</div>
    </div>
    <div class="verdict-row">
      <div class="verdict">
        <h2 class="{verdict_class}">{html.escape(result.verdict)}</h2>
        <p>{html.escape(ds.get('goal') or 'Automated review of the training pipeline, the dataset and the reported performance.')}</p>
        <p style="margin-top:14px" class="score">{result.health_score}<small> / 100 health score</small></p>
      </div>
      <div class="vitals"><dl>{vitals_html}</dl></div>
    </div>
    <div class="tally">{tally}</div>
  </header>

  <section>
    <h3>What we found</h3>
    <p class="summary">{html.escape(executive_summary(result))}</p>
  </section>
  {ba_html}
  <section>
    <h3>Issues, most serious first</h3>
    {findings_html}
  </section>

  <section>
    <h3>What was examined</h3>
    <dl class="facts">{facts}</dl>
    {notes}
  </section>

  <footer>Each issue carries a confidence score: how sure the tool is that the pattern is a real
  problem rather than a false alarm. Low-confidence findings are worth a look rather than an
  immediate change. This report is produced by automated checks and is not a substitute for a
  review of how the data was collected.</footer>
</div></body></html>"""


def write_report(
    result: AuditResult,
    path: str | Path,
    title: str = "Model audit report",
    client: Optional[str] = None,
) -> Path:
    """Write .html, .md or .json based on the file extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in {".html", ".htm"}:
        p.write_text(to_html(result, title=title, client=client), encoding="utf-8")
    elif suffix == ".json":
        p.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    else:
        p.write_text(to_markdown(result, title=title), encoding="utf-8")
    return p
