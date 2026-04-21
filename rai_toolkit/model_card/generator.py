"""Model card generator — reads MLflow metadata and renders a Jinja2 template."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger
from mlflow.tracking import MlflowClient

from rai_toolkit.bias.detector import BiasReport

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Matches models:/<name>/<version_or_stage>
_MODEL_URI_RE = re.compile(r"^models:/([^/]+)/(.+)$")


def _parse_model_uri(model_uri: str) -> tuple[str, str]:
    """Extract (model_name, version_or_stage) from a models:/ URI."""
    m = _MODEL_URI_RE.match(model_uri.strip())
    if not m:
        raise ValueError(
            f"Invalid model URI '{model_uri}'. "
            "Expected format: models:/<model_name>/<version>"
        )
    return m.group(1), m.group(2)


class ModelCardGenerator:
    """Generates a Markdown model card from MLflow metadata.

    Usage:
        generator = ModelCardGenerator()
        path = generator.generate(
            model_uri="models:/credit_model/1",
            output_path="outputs/model_card.md",
        )
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=False,
        )
        self._jinja_env.tests["number"] = lambda v: isinstance(v, (int, float))

    def generate(
        self,
        model_uri: str,
        output_path: str | Path,
        bias_report: BiasReport | None = None,
    ) -> Path:
        """Render the model card and write it to output_path.

        Args:
            model_uri:   MLflow model URI — ``models:/<name>/<version>``.
            output_path: Destination Markdown file.
            bias_report: Optional BiasReport to populate the fairness section.

        Returns:
            Path to the written file.

        Raises:
            ValueError: If model_uri format is invalid.
            Exception:  Re-raises MLflow errors so the CLI can fail explicitly.
        """
        model_name, version = _parse_model_uri(model_uri)
        logger.info(f"Fetching MLflow metadata for {model_name} v{version}")

        model_version = self._client.get_model_version(model_name, version)
        run_id = model_version.run_id

        metrics: dict = {}
        params: dict = {}
        tags: dict = {}

        if run_id:
            run = self._client.get_run(run_id)
            metrics = dict(run.data.metrics)
            params = dict(run.data.params)
            tags = dict(run.data.tags)

        context = {
            "model_name": model_name,
            "model_version": version,
            "description": tags.pop("description", None),
            "intended_use": tags.pop("intended_use", None),
            "forbidden_use": tags.pop("forbidden_use", None),
            "metrics": metrics,
            "params": params,
            "tags": tags,
            "bias_report": bias_report,
            "run_id": run_id,
            "source": model_version.source,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

        template = self._jinja_env.get_template("model_card.md.j2")
        content = template.render(**context)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")

        logger.info(f"Model card written to {out}")
        return out


def load_bias_report(json_path: str) -> BiasReport:
    """Load a BiasReport from a JSON file produced by `rai bias report`."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return BiasReport(**data)
