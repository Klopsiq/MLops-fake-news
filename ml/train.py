"""Один вызов — один эксперимент. Сравнивай runs по validation, не по test."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.common import configure_tracking, log_dataset, log_evaluation, log_provenance
from ml.data import DATA_DIR, read_split

VARIANTS = ("dummy", "baseline", "bigram", "c01")


def build_model(variant: str) -> Pipeline:
	if variant not in VARIANTS:
		raise ValueError(f"Неизвестный variant: {variant}")
	classifier = (
		DummyClassifier(strategy="most_frequent")
		if variant == "dummy"
		else LogisticRegression(
			C=0.1 if variant == "c01" else 1.0, max_iter=1000, random_state=42
		)
	)
	return Pipeline(
		[
			(
				"tfidf",
				TfidfVectorizer(
					ngram_range=(1, 2) if variant == "bigram" else (1, 1),
					min_df=2,
					max_features=30000,
				),
			),
			("classifier", classifier),
		]
	)


def train(variant: str, data_dir: Path = DATA_DIR) -> dict:
	train_df, manifest = read_split("train", data_dir)
	val_df, _ = read_split("validation", data_dir)
	model = build_model(variant)
	configure_tracking()
	with mlflow.start_run(run_name=f"{variant}-v1") as run:
		log_provenance(data_dir, manifest)
		log_dataset(train_df, "train", manifest, "training")
		log_dataset(val_df, "validation", manifest, "validation")
		mlflow.log_params(
			{
				"variant": variant,
				"seed": manifest["seed"],
				"preprocessing_version": manifest["preprocessing_version"],
				"train_rows": len(train_df),
				"validation_rows": len(val_df),
				"source_fake_label": manifest["source_fake_label"],
				**{
					key: value
					for key, value in model.get_params().items()
					if "__" in key
				},
			}
		)
		started = perf_counter()
		model.fit(train_df.text.tolist(), train_df.label)
		mlflow.log_metric("fit_seconds", perf_counter() - started)
		predictions = model.predict(val_df.text.tolist())
		metrics = log_evaluation(val_df, predictions, "val")
		# numpy-массив даёт tensor signature. Список MLflow может трактовать
		# как таблицу; pyfunc тогда передаст DataFrame, который TF-IDF не ждёт.
		example = train_df.text.head(2).to_numpy(dtype=str)
		info = mlflow.sklearn.log_model(
			sk_model=model,
			name="model",
			input_example=example,
			signature=infer_signature(example, model.predict(example)),
			pip_requirements=[
				f"{package}=={version(package)}"
				for package in ("scikit-learn", "numpy", "scipy", "cloudpickle")
			],
			metadata={"label_mapping": {"0": "real", "1": "fake"}},
		)
		mlflow.set_tag("logged_model_uri", info.model_uri)
		# Проверяем сериализацию сразу, пока исходные предсказания ещё в памяти.
		reloaded = mlflow.sklearn.load_model(info.model_uri)
		if reloaded.predict(example).tolist() != model.predict(example).tolist():
			raise ValueError("Предсказания изменились после загрузки из MLflow.")
		served = mlflow.pyfunc.load_model(info.model_uri)
		if served.predict(example).tolist() != model.predict(example).tolist():
			raise ValueError("Signature/pyfunc изменили предсказания модели.")
		mlflow.set_tag("validation_status", "passed")
		result = {
			"run_id": run.info.run_id,
			"model_uri": info.model_uri,
			"metrics": metrics,
		}
		print(json.dumps(result, indent=2))
		return result


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--variant", choices=VARIANTS, required=True)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	args = parser.parse_args()
	train(args.variant, args.data_dir)


if __name__ == "__main__":
	main()
