import json
import logging
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.logging_config import JsonFormatter
from app.main import app


def application_events(caplog):
	return [record for record in caplog.records if record.name.startswith("app.")]


def test_request_and_lifecycle_events(caplog):
	with caplog.at_level(logging.INFO), TestClient(app) as client:
		response = client.get("/healthz?token=secret-query-value")

	assert response.status_code == 200
	events = application_events(caplog)
	assert {event.getMessage() for event in events} >= {
		"application_started",
		"application_stopped",
		"http_request_completed",
	}
	request = next(e for e in events if e.getMessage() == "http_request_completed")
	data = json.loads(JsonFormatter().format(request))
	assert data["method"] == "GET"
	assert data["path"] == "/healthz"
	assert data["status_code"] == 200
	assert data["duration_ms"] >= 0
	assert data["level"] == "INFO"
	assert data["timestamp"].endswith("+00:00")
	assert "secret-query-value" not in JsonFormatter().format(request)


def test_database_error_logs_type_without_credentials(caplog):
	with (
		caplog.at_level(logging.INFO),
		TestClient(app) as client,
		patch(
			"sqlalchemy.ext.asyncio.AsyncEngine.connect",
			side_effect=OSError("postgresql://user:secret-password@host/db"),
		),
	):
		response = client.get("/api/v1/health")

	assert response.status_code == 503
	events = application_events(caplog)
	failure = next(e for e in events if e.getMessage() == "dependency_unavailable")
	data = json.loads(JsonFormatter().format(failure))
	assert data["component"] == "postgres"
	assert data["error_type"] == "OSError"
	assert data["level"] == "WARNING"
	assert all("secret-password" not in JsonFormatter().format(e) for e in events)


def test_unexpected_request_error_is_logged_and_returns_500(caplog):
	with (
		caplog.at_level(logging.INFO),
		TestClient(app, raise_server_exceptions=False) as client,
		patch(
			"sqlalchemy.ext.asyncio.AsyncEngine.connect",
			side_effect=ValueError("internal-sensitive-details"),
		),
	):
		response = client.get("/api/v1/health")

	assert response.status_code == 500
	events = application_events(caplog)
	failure = next(e for e in events if e.getMessage() == "http_request_failed")
	data = json.loads(JsonFormatter().format(failure))
	assert data["status_code"] == 500
	assert data["error_type"] == "ValueError"
	assert data["level"] == "ERROR"
	assert "internal-sensitive-details" not in JsonFormatter().format(failure)
