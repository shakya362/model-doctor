# Model Doctor — Automated ML Audit Toolkit

> You hand it a dataset, a trained model and (optionally) the training script.
> It hands back a report that tells a **non-technical client** what is wrong with the model,
> why that matters in business terms, and what to do about it.

Built for the Digicrome *Model Doctor* hackathon (ML track).
Pure Python — pandas / numpy / scikit-learn / joblib / jinja2. No notebook required, no hardcoded dataset,
no hardcoded model class.

---

## Quickstart

```bash
pip install -r requirements.txt        # or: pip install -e .

# 1) run the five demo pipelines and generate their audit reports
python3 run_demo.py

# 2) prove the auditor catches what it claims to catch
python3 -m pytest tests -q             # 41 passed

# 3) audit anything else from the command line
python3 -m model_doctor \
    --data      artifacts/pipeline_01/data.csv \
    --target    defaulted \
    --model     artifacts/pipeline_01/model.joblib \
    --code      broken_pipelines/pipeline_01_leaky_preprocessing.py \
    --goal      "Predict which loan applicants will default" \
    --out       reports/my_audit.html
```

Open `reports/pipeline_01.html` in a browser to see what the client receives.

### Library use

```python
from model_doctor import audit, write_report

result = audit(
    data="churn.csv",
    target="churned",
    model="model.joblib",          # optional — any object with .predict
    code="train.py",               # optional — parsed, never executed
    goal="Flag customers likely to cancel next month",
    time_column="signup_date",     # optional
    group_column="customer_id",    # optional
)

print(result.verdict, result.health_score)   # "Not safe to deploy", 30
for f in result.findings:                    # already sorted by priority
    print(f.severity.value, f.check_id, round(f.confidence, 2), f.title)

write_report(result, "audit.html")           # .html / .md / .json by extension
```

---

## What it detects

All seven issue types from the brief are covered (the brief asks for five), across **26 distinct findings**
from 8 check modules. Every check is a registered callable `(AuditContext) -> List[Finding]`.

| Issue type (brief) | check_id | How it is detected — programmatically |
|---|---|---|
| **Data leakage** | `leakage.target_proxy_feature` | Single-feature AUC / R² of each column against the target; ≥0.99 = near-certain proxy, ≥0.93 = suspect. Catches the "column that is really the answer in disguise". |
| | `leakage.identifier_column` | Name hints **plus** near-uniqueness (ratio > 0.5 with an ID-ish name, or > 0.9 regardless) so a 40-value `store_id` is treated as a category, not an ID. |
| | `leakage.temporal_split` | Detects a genuine time column (regex + a numeric fallback requiring ≥90 % non-decreasing values), then compares random-split vs time-ordered-split performance. |
| | `leakage.future_window_feature` | **Novel check.** Within-entity de-meaned cross-correlation of each feature against the target at offsets −3…+3. If the best *future* offset is ≥0.8× the best *past* offset (both ≥0.15), the feature is built from a window that peeks forward. Separates a centred rolling mean (ratio 0.97 → flagged) from a properly lagged one (0.46 → clean). |
| | `code.preprocessing_fit_before_split` | AST line-ordering: a `.fit()` / `.fit_transform()` on a scaler or encoder appearing *before* `train_test_split`. |
| **Train/test contamination** | `contamination.duplicate_rows` | Exact duplicate rows that would straddle a random split, with the expected optimism quantified. |
| | `contamination.conflicting_duplicates` | Identical features carrying different labels — an irreducible-error ceiling. |
| | `contamination.near_duplicate_rows` | Row-hash on rounded numerics to catch jittered copies. |
| | `contamination.group_leakage` | Same entity present on both sides of a random split; scores grouped vs ungrouped CV to show the inflation. |
| | `code.random_split_on_time_series`, `code.split_without_stratify`, `code.resample_before_split` | AST analysis of the training script's CV strategy. |
| **Misleading metrics** | `metrics.accuracy_on_imbalanced_classes` | Compares reported accuracy to the majority-class baseline and re-states the honest precision / recall / F1 / AUC. |
| | `metrics.regression_no_better_than_mean` | R² against a mean-predictor baseline. |
| **Overfitting** | `overfitting.train_test_gap` | Train-vs-holdout gap. **Attribution-aware:** with no supplied model it only fires at a gap ≥0.30 and says explicitly that the number came from a surrogate random forest, not the client's model. |
| | `overfitting.suspiciously_perfect_score` | Holdout scores at the "too good to be real" ceiling, cross-referenced with leakage findings. |
| **Data quality** | `quality.missing_values`, `quality.disguised_missing_values` | Null rates plus sentinel values (−999, "NA", "unknown", empty strings) that survive `isnull()`. |
| | `quality.inconsistent_category_labels` | Case/whitespace/near-duplicate category variants that become separate one-hot columns. |
| | `quality.high_cardinality_categoricals`, `quality.constant_columns`, `quality.extreme_outliers`, `quality.missing_target_values`, `quality.single_class_target` | Structural data defects that silently degrade fit. |
| **Train/inference mismatch** | `inference.model_cannot_score_dataset` | Tries the supplied artifact against `feature_names_in_` → raw frame → encoded matrix → ID-trimmed matrix, and reports which form (if any) worked. |
| | `inference.artifact_underperforms_refit` | The shipped artifact scores materially below a fresh refit on the same data — the classic "encoder wasn't saved with the model" symptom. |
| **Class-imbalance blindness** | `imbalance.skewed_target` | Class ratio and its consequences for thresholding. |
| | `imbalance.model_ignores_minority_class` | Measures minority-class recall directly; falls back to a refit when the supplied artifact scores ≥0.12 below it (i.e. the artifact is broken rather than the model being lazy). |
| **Code smells** | `code.preprocessing_outside_pipeline` | Preprocessing applied manually instead of inside a `Pipeline`, the root cause of most of the above. |

Static analysis is **AST-based and never executes the training script**; `.ipynb` files are flattened and parsed the same way.

---

## Severity, confidence and the health score

Every finding carries a `severity` (critical / high / medium / low), a `confidence` in 0–1, a
`plain_english` sentence for the client, `why_it_matters`, a `suggested_fix`, an `evidence` dict for the
engineer, and `auto_fixable`.

```
priority     = severity.rank × confidence
health_score = 100 − Σ (weight × confidence)      weights: critical 34, high 18, medium 8, low 3
verdict      = Healthy (≥85) | Needs attention | Not safe to deploy (<55)
```

Confidence is deliberately honest rather than decorative: a hard structural fact (an exact duplicate count)
carries ~0.95, while a heuristic like `future_window_feature` carries 0.62. The score therefore degrades
gracefully instead of collapsing on a single uncertain signal.

## Auto-fix with before/after re-measure

`run_autofix()` (on by default; `--no-autofix` to skip) drops the columns implicated by auto-fixable findings,
retrains the same model family, and re-measures on an honest holdout. The report shows both numbers side by
side — the drop from the reported score to the honest one is usually the single most persuasive line for a client.

## Report design

The HTML report is deliberately styled as a **clinical chart**, not a dashboard: a Georgia-serif "doctor's
letter" executive summary the client actually reads, triage-coloured severity rails, and all technical evidence
folded into `<details>` blocks so engineers can drill in without the client having to. It is fully
self-contained — no scripts, no external URLs — and prints cleanly, adapts to dark mode and works on mobile.
Markdown and JSON renderers share the same `Finding` objects.

---

## Proof of work — the five fixture pipelines

Each pipeline in `broken_pipelines/` plants a specific, known bug. `run_demo.py` trains all five, saves
`artifacts/pipeline_0N/{data.csv, model.joblib}` and writes `reports/pipeline_0N.{html,md,json}`.

| # | Planted bug | Verdict | Health | crit/high/med/low | Reported → honest |
|---|---|---|---|---|---|
| 01 | Scaler + target-proxy feature fit on full data before split | Not safe to deploy | 30 | 1/2/1/3 | roc_auc 0.9847 → **0.6905** |
| 02 | Duplicate rows straddling the train/test split | Not safe to deploy | 40 | 1/2/0/0 | roc_auc 0.7736 → **0.6116** |
| 03 | 97:3 imbalance reported as accuracy | Not safe to deploy | 9 | 1/3/1/2 | roc_auc 0.6500 → 0.6466 |
| 04 | Centred rolling window (future leakage) + random split on time data | Not safe to deploy | 41 | 1/2/2/0 | r2 0.9189 → **0.7923** |
| 05 | **Clean control** — nothing planted | **Healthy** | 95 | 0/0/1/0 | — |

Pipeline 05 is the important one: a correct pipeline produces **zero critical and zero high findings**, which
is the evidence that the tool discriminates rather than just alarms.

## Test suite

`python3 -m pytest tests -q` → **41 passed**.

- Each bug type is caught by the check that claims it, **and** the matching clean counterpart is *not* flagged.
- Model-agnosticism parametrised over four families: LogisticRegression, RandomForest, a sklearn `Pipeline`, and XGBoost.
- Dataset-agnosticism: regression, multiclass, wide (many columns), and tiny frames.
- Robustness: notebook parsing, syntactically malformed scripts, and a deliberately exploding check that must be
  recorded in `result.errors` without killing the audit.
- Output: HTML self-containment, autofix before < after assertion, CLI end-to-end and `--fail-on` exit code 1.

---

## Layout

```
model_doctor/
  audit.py          orchestrator — runs the registry, collects metrics, attaches context
  context.py        AuditContext: loaders, task inference, encoding, metrics, safe_predict
  findings.py       Severity, Category, Finding, AuditResult (has(), health_score, verdict)
  evaluation.py     cached holdout / provided-model / baseline evaluations
  autofix.py        fix-and-remeasure
  report.py         to_html / to_markdown / write_report
  cli.py            argparse CLI with --fail-on for CI
  checks/           leakage, contamination, metrics, overfitting, imbalance,
                    data_quality, code_smells, inference  (+ REGISTRY / @register)
broken_pipelines/   five fixtures + datasets.py
tests/              conftest.py + 8 test modules
run_demo.py         builds artifacts/ and reports/ end to end
```

Adding a check is one decorated function — no changes to the orchestrator:

```python
from model_doctor.checks import register
from model_doctor.findings import Finding, Severity, Category

@register("leakage.my_new_check")
def my_check(ctx):
    ...
    return [Finding(check_id="leakage.my_thing", severity=Severity.HIGH, confidence=0.8, ...)]
```

## Mapping to the judging criteria

| Criterion | Where it is answered |
|---|---|
| Breadth & correctness (30 %) | 26 findings covering all 7 brief issue types, not the required 5; each one demonstrated on a fixture and asserted in tests |
| Generalizability (25 %) | Registry architecture, duck-typed model access with `safe_predict` fallbacks, 4 model families and 4 dataset shapes under test, CLI over arbitrary CSV + `.pkl`/`.joblib`, and a clean control that stays clean |
| Report quality (20 %) | Plain-English summary per finding, clinical-chart HTML, before/after numbers, technical evidence hidden behind `<details>` |
| Code quality & tests (15 %) | 41 tests, isolated check modules, failure containment via `result.errors`, typed dataclasses |
| Presentation (10 %) | This README, `run_demo.py` as a one-command story, and five ready-made reports |

## Known limits (stated deliberately)

- `future_window_feature` is a heuristic and ships at confidence 0.62; it can miss leakage expressed through
  a non-linear transform of a future window.
- Static analysis reads structure, not semantics — preprocessing hidden behind a custom helper function
  will not be traced into.
- Group leakage needs `--group-column`; it is not inferred.
- Tabular data only.

---

MIT licensed. Built by Ashish Kumar.
