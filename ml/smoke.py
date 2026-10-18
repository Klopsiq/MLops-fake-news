"""Первая проверка Tracking Server без настоящего обучения."""

from pathlib import Path

import mlflow

from ml.common import configure_tracking


def main():
	configure_tracking()
	with mlflow.start_run(run_name="smoke") as run:
		mlflow.set_tag("purpose", "smoke")
		mlflow.log_param("exercise", "first_run")
		mlflow.log_metric("smoke_value", 1.0)
		mlflow.log_text("Tracking works. No model was trained.", "notes.txt")
		uri = mlflow.get_artifact_uri("notes.txt")
		path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
		print(f"run_id={run.info.run_id}\nartifact_uri={uri}")
		print(Path(path).read_text())


if __name__ == "__main__":
	main()
