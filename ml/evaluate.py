"""Финальная оценка уже выбранной версии. Не используем test для подбора."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from ml.common import configure_tracking, log_dataset, log_evaluation, log_provenance
from ml.data import DATA_DIR, file_hash, read_split


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-name", default="fake-news-classifier")
	parser.add_argument("--version", required=True)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	args = parser.parse_args()
	configure_tracking()
	client = MlflowClient()
	version = client.get_model_version(args.model_name, args.version)
	training = client.get_run(version.run_id)
	if training.data.tags.get("split_manifest_sha256") != file_hash(
		args.data_dir / "manifest.json"
	):
		raise ValueError(
			"Manifest не совпадает с обучающим run; test может быть другим."
		)
	frame, manifest = read_split("test", args.data_dir)
	model = mlflow.sklearn.load_model(f"models:/{args.model_name}/{version.version}")
	with mlflow.start_run(run_name="final-test"):
		log_provenance(args.data_dir, manifest)
		log_dataset(frame, "test", manifest, "testing")
		mlflow.set_tags(
			{
				"model_name": args.model_name,
				"model_version": version.version,
				"training_run_id": version.run_id,
			}
		)
		metrics = log_evaluation(frame, model.predict(frame.text.tolist()), "test")
		print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
