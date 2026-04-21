import json
import pickle
from pathlib import Path

import click
import pandas as pd


@click.group()
@click.version_option(version="0.1.0", prog_name="rai-toolkit")
def cli() -> None:
    """Responsible AI Toolkit — bias detection, audit trail, model cards."""


# ---------------------------------------------------------------------------
# bias commands
# ---------------------------------------------------------------------------

@cli.group()
def bias() -> None:
    """Bias detection and fairness metrics."""


@bias.command("analyze")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True), help="Path to a pickled Scikit-learn model (.pkl)")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to a CSV file with features and target")
@click.option("--target", required=True, help="Name of the target column in the CSV")
@click.option("--sensitive", required=True, help="Name of the sensitive column to analyse")
def bias_analyze(model_path: str, data_path: str, target: str, sensitive: str) -> None:
    """Run a bias analysis and display the report in the terminal."""
    from rai_toolkit.bias.detector import BiasDetector

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise click.BadParameter(f"Target column '{target}' not found in {data_path}")

    X = df.drop(columns=[target])
    y_true = df[target].to_numpy()

    report = BiasDetector().analyze(model, X, y_true, sensitive_col=sensitive)

    color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}[report.risk_level]
    click.echo(f"\nBias Analysis — {report.sensitive_column}")
    click.echo(f"  Groups               : {', '.join(report.groups)}")
    click.echo(f"  Demographic Parity Δ : {report.demographic_parity_diff:.4f}")
    click.echo(f"  Equal Opportunity Δ  : {report.equal_opportunity_diff:.4f}")
    click.echo(f"  Disparate Impact     : {report.disparate_impact_ratio:.4f}")
    click.echo(f"  Risk level           : {click.style(report.risk_level, fg=color, bold=True)}\n")
    click.echo("Recommendations:")
    for rec in report.recommendations:
        click.echo(f"  • {rec}")


@bias.command("report")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--target", required=True)
@click.option("--sensitive", required=True)
@click.option("--output", required=True, type=click.Path(), help="Output path for the JSON report")
def bias_report(model_path: str, data_path: str, target: str, sensitive: str, output: str) -> None:
    """Run a bias analysis and export the report as JSON."""
    from rai_toolkit.bias.detector import BiasDetector

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y_true = df[target].to_numpy()

    report = BiasDetector().analyze(model, X, y_true, sensitive_col=sensitive)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    click.echo(f"Report saved to {out_path}")


# ---------------------------------------------------------------------------
# audit commands
# ---------------------------------------------------------------------------

@cli.group()
def audit() -> None:
    """Audit trail management and verification."""


@audit.command("verify")
@click.option("--log-dir", default=None, help="Directory containing JSONL audit files (default: $AUDIT_LOG_DIR or ./audit_logs/)")
@click.option("--record-id", required=True, help="UUID of the record to verify")
def audit_verify(log_dir: str | None, record_id: str) -> None:
    """Verify the HMAC signature of a single audit record."""
    from rai_toolkit.audit.trail import AuditTrail

    trail = AuditTrail(log_dir=log_dir)
    valid = trail.verify(record_id)
    if valid:
        click.echo(click.style(f"✓ Record {record_id} is intact.", fg="green"))
    else:
        click.echo(click.style(f"✗ Record {record_id} is INVALID or not found.", fg="red"), err=True)
        raise SystemExit(1)


@audit.command("export")
@click.option("--log-dir", default=None)
@click.option("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
@click.option("--end", default=None, help="End date YYYY-MM-DD (inclusive)")
@click.option("--output", required=True, type=click.Path(), help="Output CSV path")
def audit_export(log_dir: str | None, start: str | None, end: str | None, output: str) -> None:
    """Export audit records to a CSV file, optionally filtered by date range."""
    from rai_toolkit.audit.trail import AuditTrail

    trail = AuditTrail(log_dir=log_dir)
    count = trail.export_csv(output, start_date=start, end_date=end)
    if count:
        click.echo(f"Exported {count} records to {output}")


# ---------------------------------------------------------------------------
# model-card commands
# ---------------------------------------------------------------------------

@cli.group("model-card")
def model_card() -> None:
    """Model card generation from MLflow."""


@model_card.command("generate")
@click.option("--model-uri", required=True, help="MLflow model URI, e.g. models:/credit_model/1")
@click.option("--output", required=True, type=click.Path(), help="Output Markdown file path")
@click.option("--bias-report", "bias_report_path", default=None, type=click.Path(exists=True), help="Path to a JSON bias report (from rai bias report)")
def model_card_generate(model_uri: str, output: str, bias_report_path: str | None) -> None:
    """Generate a model card Markdown file from MLflow metadata."""
    from rai_toolkit.model_card.generator import ModelCardGenerator, load_bias_report

    bias_report = load_bias_report(bias_report_path) if bias_report_path else None

    try:
        generator = ModelCardGenerator()
        out = generator.generate(model_uri=model_uri, output_path=output, bias_report=bias_report)
        click.echo(f"Model card written to {out}")
    except Exception as e:
        click.echo(f"Erreur : impossible de générer la fiche — {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
