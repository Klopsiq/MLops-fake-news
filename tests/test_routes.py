from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_healthz():
	with TestClient(app) as client:
		response = client.get("/healthz")

	assert response.status_code == 200
	assert response.json() == {"status": "ok"}
	print(type(response), response)


def test_version():
	with TestClient(app) as client:
		response = client.get("/api/v1/version")

	assert response.status_code == 200
	assert response.json()["version"] == app.version


def test_health():
	with TestClient(app) as client:
		response = client.get("/api/v1/health")

	assert response.status_code == 200
	data = response.json()

	postgres = data["postgres"]
	assert postgres["status"] == "ok"
	assert isinstance(postgres["version"], str)
	assert postgres["version"].startswith("PostgreSQL ")

	elapsed = postgres["response_time_ms"]
	assert isinstance(elapsed, (int, float))
	assert elapsed >= 0


def test_health_db_unavailable():
	with (
		TestClient(app) as client,
		patch(
			"sqlalchemy.ext.asyncio.AsyncEngine.connect",
			side_effect=OSError("Database unavailable"),
		),
	):
		response = client.get("/api/v1/health")
		healthz = client.get("/healthz")

	assert response.status_code == 503

	data = response.json()
	assert data["status"] == "error"
	assert data["postgres"]["status"] == "unavailable"
	assert data["postgres"]["version"] is None
	assert data["postgres"]["response_time_ms"] >= 0

	assert healthz.status_code == 200
	assert healthz.json() == {"status": "ok"}
