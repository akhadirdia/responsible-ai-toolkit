# rai-toolkit

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)

**Bias detection, tamper-proof audit trail, and automated model card generation — built for ML teams operating under AI Act and OSFI E-23 constraints.**

---

## Demo

```
$ rai bias analyze --model credit_model.pkl --data credit_data.csv --target approved --sensitive gender

Demographic Parity Diff : 0.3312   ⚠ HIGH
Equal Opportunity Diff  : 0.2874   ⚠ HIGH
Disparate Impact Ratio  : 0.4118   ⚠ HIGH
Risk level              : HIGH
Recommendations:
  → Apply ThresholdOptimizer on the 'gender' attribute.
  → Notify the compliance team before production deployment.
  → Report saved → outputs/bias_report.json
```

```
$ rai audit export --log-dir audit_logs --output export.csv --start-date 2026-04-01

Exported 847 records → export.csv
All HMAC signatures verified ✓
```

---

## Problem Statement

Financial institutions deploying ML models face three compliance obligations that most MLOps stacks ignore:

1. **Bias documentation** — regulators require evidence that protected groups are treated equitably, not just overall accuracy metrics.
2. **Decision auditability** — every automated decision must be reconstructible and tamper-proof, often for 7+ years.
3. **Model transparency** — model cards are increasingly required by internal risk committees and external auditors, but generating them manually is error-prone.

Standard MLOps tools (MLflow, W&B, SageMaker) track model performance. They do not address regulatory fairness obligations. This toolkit fills that gap with a CLI-first interface that integrates into existing CI/CD pipelines.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │          rai CLI (Click)         │
                        └────────┬──────────┬─────────────┘
                                 │          │
              ┌──────────────────▼──┐    ┌──▼──────────────────┐
              │    bias analyze     │    │    card generate     │
              │  audit verify/export│    │                      │
              └──────────┬──────────┘    └──────────┬──────────┘
                         │                          │
         ┌───────────────▼───────┐    ┌─────────────▼──────────┐
         │    BiasDetector       │    │  ModelCardGenerator     │
         │  (Fairlearn wrapper)  │    │  (MLflow + Jinja2)      │
         └───────────┬───────────┘    └─────────────┬──────────┘
                     │                              │
         ┌───────────▼───────────┐    ┌─────────────▼──────────┐
         │     BiasReport        │    │   MLflow Model Registry │
         │  (Pydantic v2 model)  │    │   (metrics, params,     │
         │  → JSON / CLI output  │    │    tags, run history)   │
         └───────────────────────┘    └────────────────────────┘

              ┌─────────────────────────────────────┐
              │           AuditTrail                │
              │  log() → JSONL append               │
              │  verify() → HMAC-SHA256 check       │
              │  export_csv() → date-filtered dump  │
              └─────────────────────────────────────┘
```

---

## Key Technical Decisions

### 1. JSONL + HMAC-SHA256 instead of a database for audit logs

A database can be updated. An append-only JSONL file with per-record HMAC signatures cannot be altered without invalidating the signature — which is exactly what tamper-evidence requires. Each record signs its payload (excluding the signature field) with a SHA-256 HMAC keyed by an environment secret. `hmac.compare_digest()` is used instead of `==` to prevent timing attacks. The tradeoff: no indexed queries, but regulators don't run SQL — they request exports.

### 2. Fairlearn MetricFrame instead of custom group metrics

Implementing group fairness metrics from scratch introduces subtle bugs (e.g., handling zero-division when a group has no positive labels). Fairlearn is co-developed with Microsoft Research and aligned with the EU AI Act Article 10 requirements for statistical fairness. Using it as a dependency signals regulatory alignment to auditors, not just correctness.

### 3. `model.feature_names_in_` as the column selection contract

When BiasDetector receives a DataFrame that includes the sensitive column, it cannot blindly pass it to `model.predict()` — the model was trained without it. Rather than requiring callers to pre-filter their data, the detector reads `model.feature_names_in_` (set automatically by scikit-learn on fit) to select exactly the columns the model expects. This makes the API safer and eliminates a class of silent shape-mismatch bugs.

### 4. Pydantic v2 for BiasReport instead of plain dicts

A `BiasReport` that is a Pydantic model can be serialized to JSON with `.model_dump()`, validated on load with `model_validate()`, and used as a typed parameter across the pipeline — from `BiasDetector.analyze()` to `ModelCardGenerator.generate()` to the CLI. Plain dicts would require manual validation at every boundary. The cost is one dependency; the benefit is a single schema definition that propagates everywhere.

### 5. Deferred imports inside Click command functions

Importing Fairlearn, MLflow, and scikit-learn at module load time adds ~2 seconds to `rai --help`. Since the CLI is invoked frequently in CI/CD pipelines, this matters. Each command function imports only what it needs, when it needs it. Python caches module imports, so repeated calls within the same process pay the cost once.

---

## Features

- **Group fairness metrics** — demographic parity difference, equal opportunity difference, disparate impact ratio with configurable risk thresholds
- **Tamper-proof audit trail** — HMAC-SHA256 signed JSONL records, verify-by-ID, date-filtered CSV export
- **Automated model cards** — pulls metrics/params/tags from MLflow, renders a Jinja2 Markdown template, embeds bias analysis when available
- **CLI-first interface** — three command groups (`bias`, `audit`, `card`) composable in shell scripts and CI pipelines
- **94% test coverage** — 49 unit tests, zero real network calls (MLflow and Fairlearn fully mocked)

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Fairness metrics | Fairlearn 0.13 | EU AI Act alignment, avoids zero-division edge cases in group metrics |
| Data validation | Pydantic v2 | Single schema propagates across CLI, API, and JSON serialization |
| Model registry | MLflow 3.x | Industry standard; provides metrics, params, tags, and run lineage |
| Template rendering | Jinja2 | Logic-in-template (conditionals, loops) without string concatenation |
| CLI framework | Click 8 | Composable command groups, automatic `--help`, type coercion |
| Audit signing | `hmac` (stdlib) | No dependency; HMAC-SHA256 is the correct primitive for this use case |
| Testing | pytest + pytest-mock | Fixture composition, mock patching at module boundary |
| Packaging | hatchling + pyproject.toml | PEP 517/518 compliant, no setup.py |

---

## Quick Start

```bash
git clone https://github.com/<your-username>/rai-toolkit.git
cd rai-toolkit
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python examples/credit_scoring_demo.py
```

This runs the end-to-end demo: trains a logistic regression on a synthetic biased credit dataset, detects gender bias, logs 20 HMAC-signed audit records, and generates a model card.

---

## Project Structure

```
rai-toolkit/
├── rai_toolkit/
│   ├── bias/
│   │   ├── metrics.py          # Pure Fairlearn wrappers (demographic parity, DIR)
│   │   └── detector.py         # BiasDetector + BiasReport (Pydantic) + risk thresholds
│   ├── audit/
│   │   └── trail.py            # AuditRecord, AuditTrail, _sign(), verify(), export_csv()
│   ├── model_card/
│   │   ├── generator.py        # ModelCardGenerator, _parse_model_uri(), load_bias_report()
│   │   └── templates/
│   │       └── model_card.md.j2  # Jinja2 template with conditional bias section
│   └── cli.py                  # Click CLI — bias / audit / card command groups
├── tests/
│   ├── test_bias_detector.py   # 20 tests — metrics, risk levels, edge cases
│   ├── test_audit_trail.py     # 15 tests — HMAC signing, tampering, CSV export
│   └── test_model_card.py      # 14 tests — MLflow mocked, URI parsing, headers
├── examples/
│   └── credit_scoring_demo.py  # End-to-end demo on synthetic credit dataset
├── .env.example                # All required variables with descriptions
├── pyproject.toml              # Build config, dependencies, CLI entry point
└── docker-compose.yml          # (Phase 2) Qdrant for vector storage
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | For `card generate` | MLflow backend — SQLite (`sqlite:///mlflow.db`) or remote URI |
| `AUDIT_LOG_DIR` | Optional | Directory for JSONL audit files — defaults to `./audit_logs` |
| `AUDIT_HMAC_SECRET` | For production | Secret key for HMAC signing — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |

Copy `.env.example` to `.env` and fill in your values. `.env` is in `.gitignore` and must never be committed.

---

## Roadmap

**Phase 2A — LLM Red Teaming**
Wrap [Giskard](https://github.com/Giskard-AI/giskard) scan logic behind a `LLMRedTeamScanner` class with a provider-agnostic `endpoint_fn(prompt: str) -> str` interface. Expose via `rai redteam scan --endpoint <url>`. Target: jailbreak, prompt injection, and hallucination detection for financial chatbots.

**Phase 2B — Compliance Checker**
YAML-defined rule sets per regulation (EU AI Act Article 9/10, OSFI E-23, Loi 25). A `ComplianceChecker` takes a `SystemDescriptor` Pydantic model describing a deployment and returns a structured gap report. Goal: automated pre-deployment checklist for risk committees.

**Phase 2C — Bias Mitigation**
Integrate Fairlearn's `ThresholdOptimizer` and `ExponentiatedGradient` directly. When `BiasReport.risk_level == "HIGH"`, expose `rai bias mitigate --strategy threshold` that trains a mitigated wrapper model and re-runs the analysis for comparison.
