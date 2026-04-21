"""Unit tests for audit/trail.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rai_toolkit.audit.trail import AuditTrail, _sign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trail(tmp_path: Path, secret: str = "test-secret") -> AuditTrail:
    with patch.dict("os.environ", {"AUDIT_HMAC_SECRET": secret, "AUDIT_LOG_DIR": str(tmp_path)}):
        return AuditTrail(log_dir=str(tmp_path))


def log_one(trail: AuditTrail, **kwargs) -> str:
    defaults = dict(
        model_name="m", model_version="1",
        input_features={"age": 30, "income": 50000},
        prediction=1, prediction_proba=0.82, user_id="u1",
    )
    defaults.update(kwargs)
    return trail.log(**defaults)


# ---------------------------------------------------------------------------
# _sign helper
# ---------------------------------------------------------------------------

class TestSign:
    def test_deterministic(self):
        payload = {"a": 1, "b": "x"}
        assert _sign(payload, "key") == _sign(payload, "key")

    def test_different_keys_differ(self):
        payload = {"a": 1}
        assert _sign(payload, "key1") != _sign(payload, "key2")

    def test_different_payloads_differ(self):
        assert _sign({"a": 1}, "k") != _sign({"a": 2}, "k")


# ---------------------------------------------------------------------------
# AuditTrail.log + verify
# ---------------------------------------------------------------------------

class TestLogAndVerify:
    def test_verify_returns_true_on_intact_record(self, tmp_path):
        trail = make_trail(tmp_path)
        rid = log_one(trail)
        assert trail.verify(rid) is True

    def test_verify_returns_false_on_tampered_prediction(self, tmp_path):
        trail = make_trail(tmp_path)
        rid = log_one(trail)

        # Tamper: overwrite prediction in the JSONL file
        jsonl = next(tmp_path.glob("audit_*.jsonl"))
        lines = jsonl.read_text().splitlines()
        data = json.loads(lines[0])
        data["prediction"] = 999
        lines[0] = json.dumps(data)
        jsonl.write_text("\n".join(lines) + "\n")

        assert trail.verify(rid) is False

    def test_verify_returns_false_on_tampered_features(self, tmp_path):
        trail = make_trail(tmp_path)
        rid = log_one(trail)

        jsonl = next(tmp_path.glob("audit_*.jsonl"))
        lines = jsonl.read_text().splitlines()
        data = json.loads(lines[0])
        data["input_features"]["income"] = 999999
        lines[0] = json.dumps(data)
        jsonl.write_text("\n".join(lines) + "\n")

        assert trail.verify(rid) is False

    def test_verify_unknown_id_returns_false(self, tmp_path):
        trail = make_trail(tmp_path)
        assert trail.verify("00000000-0000-0000-0000-000000000000") is False

    def test_log_returns_uuid_string(self, tmp_path):
        trail = make_trail(tmp_path)
        rid = log_one(trail)
        assert isinstance(rid, str)
        assert len(rid) == 36  # UUID v4 format

    def test_log_ten_records(self, tmp_path):
        trail = make_trail(tmp_path)
        ids = [log_one(trail, prediction=i) for i in range(10)]
        assert len(set(ids)) == 10  # all unique
        assert all(trail.verify(rid) for rid in ids)

    def test_log_graceful_on_missing_dir(self, tmp_path):
        """log() raises PermissionError when the JSONL file cannot be written."""
        trail = make_trail(tmp_path)
        # pathlib.Path.open is what _append calls — mock it at the module level
        with patch("rai_toolkit.audit.trail.Path.open", side_effect=PermissionError("no write")):
            with pytest.raises(PermissionError):
                log_one(trail)


# ---------------------------------------------------------------------------
# AuditTrail.export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_exports_all_records(self, tmp_path):
        trail = make_trail(tmp_path)
        for i in range(5):
            log_one(trail, prediction=i)
        csv_path = tmp_path / "out.csv"
        count = trail.export_csv(csv_path)
        assert count == 5
        assert csv_path.exists()

    def test_csv_has_correct_headers(self, tmp_path):
        trail = make_trail(tmp_path)
        log_one(trail)
        csv_path = tmp_path / "out.csv"
        trail.export_csv(csv_path)
        first_line = csv_path.read_text().splitlines()[0]
        for col in ("record_id", "timestamp", "model_name", "prediction", "hmac_signature"):
            assert col in first_line

    def test_export_no_records_returns_zero(self, tmp_path):
        trail = make_trail(tmp_path)
        csv_path = tmp_path / "out.csv"
        count = trail.export_csv(csv_path)
        assert count == 0
        assert not csv_path.exists()

    def test_date_filter_excludes_future_files(self, tmp_path):
        trail = make_trail(tmp_path)
        log_one(trail)  # today's date
        csv_path = tmp_path / "out.csv"
        # Filter to a past date range — should match nothing
        count = trail.export_csv(csv_path, start_date="2000-01-01", end_date="2000-01-02")
        assert count == 0

    def test_date_filter_includes_today(self, tmp_path):
        from datetime import date
        trail = make_trail(tmp_path)
        log_one(trail)
        today = date.today().isoformat()
        csv_path = tmp_path / "out.csv"
        count = trail.export_csv(csv_path, start_date=today, end_date=today)
        assert count == 1
