from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import load_champion


def test_process_reuses_model(model_loader):
	with TestClient(app) as client:
		for _ in range(2):
			response = client.post(
				"/process", json={"text": "  News about transport.  "}
			)
			assert response.status_code == 200
			assert response.json() == {
				"label": "fake",
				"model_name": "fake-news-classifier",
				"model_version": "7",
			}
	model_loader.assert_called_once_with()
	model_loader.return_value[0].predict.assert_called_with(["News about transport."])
	assert model_loader.return_value[0].predict.call_count == 2
	assert app.state.model is None


@pytest.mark.parametrize(
	"body",
	[
		{},
		{"text": ""},
		{"text": " \n "},
		{"text": "x" * 50001},
		{"text": 5},
		{"text": None},
	],
)
def test_process_rejects_invalid_input(body, model_loader):
	with TestClient(app) as client:
		assert client.post("/process", json=body).status_code == 422
	model_loader.return_value[0].predict.assert_not_called()


def test_process_real_label(model_loader):
	model_loader.return_value[0].predict.return_value = np.array([0])
	with TestClient(app) as client:
		assert (
			client.post("/process", json={"text": "Example news."}).json()["label"]
			== "real"
		)


def test_new_registry_version_is_used_only_after_restart(model_loader):
	model = model_loader.return_value[0]
	with TestClient(app) as client:
		assert (
			client.post("/process", json={"text": "Example news."}).json()[
				"model_version"
			]
			== "7"
		)
		# Имитируем новый результат разрешения alias; процесс уже держит версию 7.
		model_loader.return_value = (model, "fake-news-classifier", "8")
		assert (
			client.post("/process", json={"text": "Example news."}).json()[
				"model_version"
			]
			== "7"
		)
	with TestClient(app) as client:
		assert (
			client.post("/process", json={"text": "Example news."}).json()[
				"model_version"
			]
			== "8"
		)
	assert model_loader.call_count == 2


def test_startup_fails_and_disposes_database(model_loader):
	model_loader.side_effect = RuntimeError("Alias champion not found")
	engine = SimpleNamespace(dispose=AsyncMock())
	with patch("app.db.create_async_engine", return_value=engine):
		with (
			pytest.raises(RuntimeError, match="Alias champion not found"),
			TestClient(app),
		):
			pass
	engine.dispose.assert_awaited_once()


def test_loader_resolves_alias_then_loads_exact_version(monkeypatch):
	monkeypatch.setenv("MODEL_NAME", "fake-news-classifier")
	monkeypatch.setenv("MODEL_ALIAS", "champion")
	model = SimpleNamespace(classes_=np.array([0, 1]))
	client = Mock()
	client.get_model_version_by_alias.return_value.version = "12"
	with (
		patch("app.model.MlflowClient", return_value=client),
		patch("app.model.mlflow.sklearn.load_model", return_value=model) as load,
	):
		assert load_champion() == (model, "fake-news-classifier", "12")
	client.get_model_version_by_alias.assert_called_once_with(
		"fake-news-classifier", "champion"
	)
	load.assert_called_once_with("models:/fake-news-classifier/12")
