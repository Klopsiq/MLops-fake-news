"""Application events as JSON in the standard output stream."""

import json
import logging
import os
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		data = {
			"timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
			"level": record.levelname,
			"logger": record.name,
			"event": record.getMessage(),
		}
		for field in (
			"version",
			"method",
			"path",
			"status_code",
			"duration_ms",
			"component",
			"error_type",
		):
			if hasattr(record, field):
				data[field] = getattr(record, field)
		return json.dumps(data, ensure_ascii=False)


def configure_logging() -> None:
	handler = logging.StreamHandler(sys.stdout)
	handler.setFormatter(JsonFormatter())
	logging.basicConfig(
		level=os.getenv("LOG_LEVEL", "INFO").upper(), handlers=[handler]
	)
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpx2").setLevel(logging.WARNING)
