"""Регистрация выбранного Pipeline. Alias назначается только явно."""

import argparse

import mlflow
from mlflow import MlflowClient

from ml.common import configure_tracking


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	source = parser.add_mutually_exclusive_group(required=True)
	source.add_argument("--model-uri")
	source.add_argument(
		"--version", help="Уже зарегистрированная версия для переноса alias."
	)
	parser.add_argument("--model-name", default="fake-news-classifier")
	parser.add_argument(
		"--alias", help="Например champion; без аргумента только регистрация."
	)
	args = parser.parse_args()
	configure_tracking()
	client = MlflowClient()
	if args.model_uri:
		registered = mlflow.register_model(args.model_uri, args.model_name)
		version = registered.version
	else:
		version = client.get_model_version(args.model_name, args.version).version
	if args.alias:
		client.set_registered_model_alias(args.model_name, args.alias, version)
	print(
		f"model={args.model_name} version={version} alias={args.alias or '(не назначен)'}"
	)


if __name__ == "__main__":
	main()
