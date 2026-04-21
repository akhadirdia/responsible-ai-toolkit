"""Cryptographically signed audit trail for model decisions."""

import csv
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class AuditRecord(BaseModel):
    record_id: str
    timestamp: str           # ISO 8601 UTC
    model_name: str
    model_version: str
    input_features: dict
    prediction: float | int | str
    prediction_proba: float | None
    user_id: str | None
    hmac_signature: str


def _sign(payload: dict, secret: str) -> str:
    """Return HMAC-SHA256 hex digest of the JSON-serialised payload."""
    message = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


class AuditTrail:
    """Append-only audit trail stored as signed JSONL files (one per day).

    Files are written to ``log_dir`` and named ``audit_YYYY-MM-DD.jsonl``.
    Each record is signed with HMAC-SHA256 using ``AUDIT_HMAC_SECRET`` from
    the environment. If the secret is absent, records are signed with an empty
    key and a warning is emitted — logging is never blocked.
    """

    def __init__(self, log_dir: str | None = None) -> None:
        self._log_dir = Path(log_dir or os.getenv("AUDIT_LOG_DIR", "./audit_logs"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        secret = os.getenv("AUDIT_HMAC_SECRET", "")
        if not secret:
            logger.warning(
                "AUDIT_HMAC_SECRET is not set — records will be signed with an empty "
                "key. Set the variable in your .env for tamper-evident logging."
            )
        self._secret = secret

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        model_name: str,
        model_version: str,
        input_features: dict,
        prediction: float | int | str,
        prediction_proba: float | None = None,
        user_id: str | None = None,
    ) -> str:
        """Record a model decision and return the record_id."""
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "record_id": record_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "model_version": model_version,
            "input_features": input_features,
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "user_id": user_id,
        }
        signature = _sign(payload, self._secret)

        record = AuditRecord(**payload, hmac_signature=signature)
        self._append(record)

        logger.debug(f"Audit record logged: {record_id}")
        return record_id

    def verify(self, record_id: str) -> bool:
        """Return True if the record exists and its HMAC signature is intact."""
        record = self._find(record_id)
        if record is None:
            logger.warning(f"Record not found: {record_id}")
            return False

        payload = record.model_dump(exclude={"hmac_signature"})
        expected = _sign(payload, self._secret)
        valid = hmac.compare_digest(expected, record.hmac_signature)

        if not valid:
            logger.error(f"Signature mismatch for record {record_id} — possible tampering")
        return valid

    def export_csv(
        self,
        output_path: str | Path,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Export matching records to a CSV file. Returns the number of rows written.

        Args:
            output_path: Destination CSV file path.
            start_date: ISO date string (YYYY-MM-DD), inclusive. None = no lower bound.
            end_date:   ISO date string (YYYY-MM-DD), inclusive. None = no upper bound.
        """
        records = list(self._iter_records(start_date, end_date))
        if not records:
            logger.info("No records matched the filter — CSV not written.")
            return 0

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(AuditRecord.model_fields.keys())
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = record.model_dump()
                row["input_features"] = json.dumps(row["input_features"], ensure_ascii=False)
                writer.writerow(row)

        logger.info(f"Exported {len(records)} records to {out}")
        return len(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_file(self, date: str) -> Path:
        """Return the JSONL path for a given YYYY-MM-DD date string."""
        return self._log_dir / f"audit_{date}.jsonl"

    def _append(self, record: AuditRecord) -> None:
        date = record.timestamp[:10]  # YYYY-MM-DD
        path = self._log_file(date)
        with path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def _find(self, record_id: str) -> AuditRecord | None:
        """Search all JSONL files for the given record_id."""
        for path in sorted(self._log_dir.glob("audit_*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("record_id") == record_id:
                        return AuditRecord(**data)
        return None

    def _iter_records(
        self,
        start_date: str | None,
        end_date: str | None,
    ):
        """Yield AuditRecord objects filtered by date range."""
        for path in sorted(self._log_dir.glob("audit_*.jsonl")):
            # Fast-path: skip files outside the date range using the filename
            file_date = path.stem.replace("audit_", "")  # YYYY-MM-DD
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line in {path}")
                        continue
                    record_date = data.get("timestamp", "")[:10]
                    if start_date and record_date < start_date:
                        continue
                    if end_date and record_date > end_date:
                        continue
                    yield AuditRecord(**data)
