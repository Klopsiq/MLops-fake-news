"""Рабочая модель выбирается alias, но загружается по точному номеру версии."""

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


def load_champion():
	mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
	name = os.getenv("MODEL_NAME", "fake-news-classifier")
	alias = os.getenv("MODEL_ALIAS", "champion")
	version = MlflowClient().get_model_version_by_alias(name, alias)
	# Если alias поменяли во время загрузки, номер в API всё равно будет точным.
	model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
	if set(model.classes_) != {0, 1}:
		raise ValueError("Ожидалась модель с классами 0=real и 1=fake.")
	return model, name, str(version.version)
