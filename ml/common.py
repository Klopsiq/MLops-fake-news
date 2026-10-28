"""Небольшие общие операции логирования, без скрытого autolog."""

import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	classification_report,
	f1_score,
	precision_score,
	recall_score,
)

from ml.data import LABELS, ROOT, file_hash


def configure_tracking():
	mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
	mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fake-news-hw3"))


def log_provenance(data_dir: Path, manifest: dict):
	try:
		commit = subprocess.run(
			["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
		)
		dirty = subprocess.run(
			["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
		)
		commit_id = commit.stdout.strip() or "unknown"
		dirty_state = bool(dirty.stdout.strip()) if dirty.returncode == 0 else "unknown"
	except FileNotFoundError:
		commit_id, dirty_state = "unknown", "unknown"
	mlflow.set_tags(
		{
			"git_commit": commit_id,
			"git_dirty": dirty_state,
			"purpose": "smoke" if manifest["sample_size"] else "training",
			"split_manifest_sha256": file_hash(data_dir / "manifest.json"),
		}
	)
	mlflow.log_dict(manifest, "data/split_manifest.json")
	# При dirty checkout одного commit hash недостаточно для воспроизводимости.
	for directory in ("ml", "app"):
		for path in sorted((ROOT / directory).glob("*.py")):
			mlflow.log_artifact(str(path), f"source/{directory}")
	for name in ("pyproject.toml", "uv.lock"):
		mlflow.log_artifact(str(ROOT / name), "source")
	for name in ("README.md", "label-audit.json"):
		path = ROOT / "data" / name
		if path.exists():
			mlflow.log_artifact(str(path), "data")


def log_dataset(frame, name: str, manifest: dict, context: str):
	dataset = mlflow.data.from_pandas(
		frame[["row_id", "text", "label"]],
		source=manifest["source"],
		targets="label",
		name=f"welfake-{name}",
	)
	mlflow.log_input(dataset, context=context)
	return dataset


def log_evaluation(frame, predictions, prefix: str) -> dict:
	y = frame.label
	metrics = {
		f"{prefix}_f1_macro": f1_score(
			y, predictions, average="macro", zero_division=0
		),
		f"{prefix}_accuracy": accuracy_score(y, predictions),
		f"{prefix}_precision_fake": precision_score(
			y, predictions, pos_label=1, zero_division=0
		),
		f"{prefix}_recall_fake": recall_score(
			y, predictions, pos_label=1, zero_division=0
		),
		f"{prefix}_f1_fake": f1_score(y, predictions, pos_label=1, zero_division=0),
	}
	mlflow.log_metrics(metrics)
	mlflow.log_dict(
		classification_report(
			y,
			predictions,
			labels=[0, 1],
			target_names=["real", "fake"],
			output_dict=True,
			zero_division=0,
		),
		f"{prefix}/classification_report.json",
	)
	fig, ax = plt.subplots()
	ConfusionMatrixDisplay.from_predictions(
		y, predictions, labels=[0, 1], display_labels=["real", "fake"], ax=ax
	)
	mlflow.log_figure(fig, f"{prefix}/confusion_matrix.png")
	plt.close(fig)
	errors = frame[["row_id", "text", "label"]].copy()
	errors["prediction"] = predictions
	errors = errors.loc[errors.label != errors.prediction]
	errors["text"] = errors.text.str.slice(0, 1000)
	mlflow.log_text(errors.to_csv(index=False), f"{prefix}/errors.csv")
	mlflow.log_dict(LABELS, "label_mapping.json")
	return metrics
