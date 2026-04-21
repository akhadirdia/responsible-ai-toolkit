"""Unit tests for model_card/generator.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rai_toolkit.bias.detector import BiasReport
from rai_toolkit.model_card.generator import ModelCardGenerator, _parse_model_uri, load_bias_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_HEADERS = [
    "## Description",
    "## Performance",
    "## Biais",
    "## Limitations",
    "## Date de",
]

def make_mock_client(
    run_metrics: dict | None = None,
    run_params: dict | None = None,
    run_tags: dict | None = None,
    run_id: str = "abc123",
    source: str = "s3://bucket/model",
):
    """Return a fully mocked MlflowClient."""
    mock_version = MagicMock()
    mock_version.run_id = run_id
    mock_version.source = source

    mock_run = MagicMock()
    mock_run.data.metrics = run_metrics or {"accuracy": 0.90, "f1_score": 0.88}
    mock_run.data.params = run_params or {"C": "1.0"}
    mock_run.data.tags = run_tags or {"author": "test", "dataset": "synthetic"}

    mock_client = MagicMock()
    mock_client.get_model_version.return_value = mock_version
    mock_client.get_run.return_value = mock_run
    return mock_client


@pytest.fixture()
def generator(tmp_path):
    """ModelCardGenerator with a mocked MlflowClient."""
    with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
        gen = ModelCardGenerator()
    return gen


@pytest.fixture()
def bias_report():
    return BiasReport(
        sensitive_column="gender",
        groups=["female", "male"],
        demographic_parity_diff=0.33,
        equal_opportunity_diff=0.28,
        disparate_impact_ratio=0.43,
        risk_level="HIGH",
        recommendations=["Apply ThresholdOptimizer.", "Notify compliance team."],
    )


# ---------------------------------------------------------------------------
# _parse_model_uri
# ---------------------------------------------------------------------------

class TestParseModelUri:
    def test_valid_uri(self):
        name, version = _parse_model_uri("models:/credit_model/1")
        assert name == "credit_model"
        assert version == "1"

    def test_valid_uri_with_alias(self):
        name, version = _parse_model_uri("models:/credit_model/champion")
        assert name == "credit_model"
        assert version == "champion"

    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="Invalid model URI"):
            _parse_model_uri("not-a-valid-uri")

    def test_strips_whitespace(self):
        name, version = _parse_model_uri("  models:/my_model/2  ")
        assert name == "my_model"


# ---------------------------------------------------------------------------
# ModelCardGenerator.generate
# ---------------------------------------------------------------------------

class TestGenerateModelCard:
    def test_file_is_created(self, tmp_path):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        gen.generate("models:/credit_model/1", output_path=out)
        assert out.exists()

    def test_all_required_headers_present(self, tmp_path):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        gen.generate("models:/credit_model/1", output_path=out)
        content = out.read_text(encoding="utf-8")
        for header in REQUIRED_HEADERS:
            assert header in content, f"Missing header: {header}"

    def test_metrics_in_output(self, tmp_path):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        gen.generate("models:/credit_model/1", output_path=out)
        content = out.read_text(encoding="utf-8")
        assert "accuracy" in content
        assert "0.9000" in content

    def test_model_name_in_output(self, tmp_path):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        gen.generate("models:/credit_model/1", output_path=out)
        content = out.read_text(encoding="utf-8")
        assert "credit_model" in content

    def test_bias_report_section_populated(self, tmp_path, bias_report):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        gen.generate("models:/credit_model/1", output_path=out, bias_report=bias_report)
        content = out.read_text(encoding="utf-8")
        assert "gender" in content
        assert "HIGH" in content
        assert "ThresholdOptimizer" in content

    def test_no_bias_report_shows_placeholder(self, tmp_path):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        gen.generate("models:/credit_model/1", output_path=out, bias_report=None)
        content = out.read_text(encoding="utf-8")
        assert "rai bias analyze" in content

    def test_returns_path_object(self, tmp_path):
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=make_mock_client()):
            gen = ModelCardGenerator()
        result = gen.generate("models:/credit_model/1", output_path=tmp_path / "card.md")
        assert isinstance(result, Path)

    def test_mlflow_error_propagates(self, tmp_path):
        """If MLflow is unreachable, the exception must bubble up (CLI will catch it)."""
        broken_client = MagicMock()
        broken_client.get_model_version.side_effect = Exception("Connection refused")
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=broken_client):
            gen = ModelCardGenerator()
        with pytest.raises(Exception, match="Connection refused"):
            gen.generate("models:/credit_model/1", output_path=tmp_path / "card.md")

    def test_no_run_id_still_generates(self, tmp_path):
        """A model version with no associated run should still produce a card."""
        mock = make_mock_client(run_id=None)
        mock.get_run.side_effect = Exception("No run")
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=mock):
            gen = ModelCardGenerator()
        out = tmp_path / "card.md"
        # run_id=None means we skip get_run — no exception expected
        mock2 = make_mock_client()
        mock2.get_model_version.return_value.run_id = None
        with patch("rai_toolkit.model_card.generator.MlflowClient", return_value=mock2):
            gen2 = ModelCardGenerator()
        gen2.generate("models:/credit_model/1", output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# load_bias_report
# ---------------------------------------------------------------------------

class TestLoadBiasReport:
    def test_loads_valid_json(self, tmp_path, bias_report):
        import json
        p = tmp_path / "report.json"
        p.write_text(json.dumps(bias_report.model_dump(), ensure_ascii=False), encoding="utf-8")
        loaded = load_bias_report(str(p))
        assert loaded.risk_level == "HIGH"
        assert loaded.sensitive_column == "gender"

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(Exception):
            load_bias_report(str(p))
