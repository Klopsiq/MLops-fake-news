"""Подготовка WELFake. Запуск: python -m ml.data --download --fake-label 1."""

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RAW_PATH = ROOT / "data" / "raw" / "WELFake_Dataset.csv"
SOURCE_URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv"
EXPECTED_MD5 = "73c9675a4b3d09f86a6933d0b8d7d908"
SEED = 42
LABELS = {0: "real", 1: "fake"}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
	digest = hashlib.new(algorithm)
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def download(path: Path) -> None:
	"""Не подменяем уже скачанный исходник; проверяем MD5 из карточки Zenodo."""
	if not path.exists():
		path.parent.mkdir(parents=True, exist_ok=True)
		partial = path.with_suffix(".csv.part")
		request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "HW3/1.0"})
		with urllib.request.urlopen(request, timeout=60) as response:
			with partial.open("wb") as output:
				while chunk := response.read(1024 * 1024):
					output.write(chunk)
		if file_hash(partial, "md5") != EXPECTED_MD5:
			raise ValueError(f"MD5 не совпадает: {partial}. Исходник не принят.")
		partial.replace(path)
	if file_hash(path, "md5") != EXPECTED_MD5:
		raise ValueError("Файл отличается от WELFake Zenodo 4561253; проверь источник.")
	metadata = {
		"source": SOURCE_URL,
		"verified_at_utc": datetime.now(UTC).isoformat(),
		"md5": EXPECTED_MD5,
		"sha256": file_hash(path),
	}
	path.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")


def clean_data(raw: pd.DataFrame, fake_label: int) -> tuple[pd.DataFrame, dict]:
	if not {"text", "label"}.issubset(raw.columns):
		raise ValueError("CSV должен содержать столбцы text и label.")
	frame = raw[["text", "label"]].copy()
	frame.insert(0, "row_id", range(len(frame)))
	quality = {"raw_rows": len(frame)}
	valid_text = frame["text"].map(
		lambda value: isinstance(value, str) and bool(value.strip())
	)
	quality["removed_empty_text"] = int((~valid_text).sum())
	frame = frame.loc[valid_text].copy()
	frame["text"] = frame["text"].str.strip()
	numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
	valid_label = numeric_labels.isin([0, 1])
	quality["removed_invalid_label"] = int((~valid_label).sum())
	frame = frame.loc[valid_label].copy()
	# Внутреннее соглашение едино для метрик и API: 1 = fake, 0 = real.
	frame["label"] = (numeric_labels.loc[valid_label] == fake_label).astype(int)
	normalized = frame["text"].map(
		lambda value: re.sub(r"\s+", " ", value.casefold()).strip()
	)
	frame["text_hash"] = normalized.map(
		lambda value: hashlib.sha256(value.encode()).hexdigest()
	)
	conflicts = frame.groupby("text_hash")["label"].nunique()
	conflict_hashes = conflicts[conflicts > 1].index
	conflicting = frame["text_hash"].isin(conflict_hashes)
	quality["removed_conflicting_rows"] = int(conflicting.sum())
	frame = frame.loc[~conflicting].copy()
	quality["removed_duplicate_rows"] = int(frame.duplicated("text_hash").sum())
	frame = frame.drop_duplicates("text_hash").reset_index(drop=True)
	quality["clean_rows"] = len(frame)
	return frame, quality


def prepare(
	raw_path: Path, output_dir: Path, fake_label: int, sample_size: int | None = None
) -> dict:
	frame, quality = clean_data(pd.read_csv(raw_path), fake_label)
	if sample_size is not None:
		if not 20 <= sample_size < len(frame):
			raise ValueError(
				"sample-size должен быть >= 20 и меньше очищенного датасета."
			)
		frame, _ = train_test_split(
			frame, train_size=sample_size, stratify=frame.label, random_state=SEED
		)
	train, holdout = train_test_split(
		frame, test_size=0.30, stratify=frame.label, random_state=SEED
	)
	validation, test = train_test_split(
		holdout, test_size=0.50, stratify=holdout.label, random_state=SEED
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	manifest = {
		"source": SOURCE_URL,
		"raw_file": raw_path.name,
		"raw_sha256": file_hash(raw_path),
		"preprocessing_version": "body-text-dedup-v1",
		"source_fake_label": fake_label,
		"labels": LABELS,
		"seed": SEED,
		"sample_size": sample_size,
		"quality": quality,
		"splits": {},
	}
	seen = set()
	for name, split in (("train", train), ("validation", validation), ("test", test)):
		if seen.intersection(split.text_hash):
			raise ValueError("Обнаружено пересечение текстов между split.")
		seen.update(split.text_hash)
		path = output_dir / f"{name}.csv"
		split.to_csv(path, index=False, lineterminator="\n")
		manifest["splits"][name] = {
			"file": path.name,
			"sha256": file_hash(path),
			"rows": len(split),
			"class_counts": {
				str(k): int(v) for k, v in split.label.value_counts().items()
			},
		}
	(output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
	return manifest


def read_split(name: str, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
	manifest = json.loads((data_dir / "manifest.json").read_text())
	path = data_dir / manifest["splits"][name]["file"]
	if file_hash(path) != manifest["splits"][name]["sha256"]:
		raise ValueError(f"{name}: SHA-256 не совпадает с manifest; данные изменились.")
	frame = pd.read_csv(path, keep_default_na=False)
	if not {"row_id", "text", "label", "text_hash"}.issubset(frame.columns):
		raise ValueError(f"{name}: отсутствуют обязательные столбцы.")
	if not frame.label.isin([0, 1]).all() or frame.text.str.strip().eq("").any():
		raise ValueError(f"{name}: невалидный текст или метка.")
	return frame, manifest


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--download", action="store_true"
	)  # получим True при использовании, без флага - False
	parser.add_argument("--raw", type=Path, default=RAW_PATH)
	parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
	parser.add_argument(
		"--fake-label",
		type=int,
		choices=[0, 1],
		required=True,
		help="Какая исходная метка означает fake; проверь data/README.md.",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		help="Только техническая проверка; отдельная подвыборка.",
	)
	args = parser.parse_args()
	if args.download:
		download(args.raw)
	manifest = prepare(args.raw, args.output_dir, args.fake_label, args.sample_size)
	print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
	main()
