import os
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def model_loader(monkeypatch):
	# Unit-тесты не зависят от настоящего Registry и не скачивают модель.
	monkeypatch.setenv(
		"DATABASE_URL",
		os.getenv("DATABASE_URL", "postgresql+asyncpg://test:test@127.0.0.1:5433/test"),
	)
	model = Mock()
	model.predict.return_value = np.array([1])
	with patch(
		"app.db.load_champion", return_value=(model, "fake-news-classifier", "7")
	) as loader:
		yield loader
