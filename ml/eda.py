"""EDA: сначала вопрос к данным, затем график и измеренный вывод."""

import argparse
from pathlib import Path

import mlflow

from ml.common import configure_tracking, log_dataset, log_provenance, plt
from ml.data import DATA_DIR, read_split


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	args = parser.parse_args()
	frame, manifest = read_split("train", args.data_dir)
	configure_tracking()
	with mlflow.start_run(run_name="eda-v1"):
		log_provenance(args.data_dir, manifest)
		log_dataset(frame, "train", manifest, "eda")
		mlflow.log_dict(manifest["quality"], "eda/data_quality.json")
		counts = frame.label.value_counts().reindex([0, 1], fill_value=0)
		fig, ax = plt.subplots()
		ax.bar(["real", "fake"], counts.values)
		ax.set(title="Train class balance", ylabel="Number of articles")
		mlflow.log_figure(fig, "eda/class_balance.png")
		plt.close(fig)
		fig, ax = plt.subplots()
		for label, title in [(0, "real"), (1, "fake")]:
			lengths = frame.loc[frame.label == label, "text"].str.len()
			ax.hist(lengths, bins=70, alpha=0.5, label=title)
		ax.set(
			title="Train text length by class",
			xlabel="Characters",
			ylabel="Number of articles",
			yscale="log",
		)
		ax.legend()
		mlflow.log_figure(fig, "eda/text_lengths.png")
		plt.close(fig)
		medians = (
			frame.assign(characters=frame.text.str.len())
			.groupby("label")
			.characters.median()
		)
		quality = manifest["quality"]
		notes = (
			f"# Измеренные наблюдения\n\n"
			f"- Исходных строк: {quality['raw_rows']}; после очистки: {quality['clean_rows']}.\n"
			f"- Пустых текстов удалено: {quality['removed_empty_text']}; повторов: {quality['removed_duplicate_rows']}; конфликтующих строк: {quality['removed_conflicting_rows']}. Это причины очистки до split.\n"
			f"- Train: real={int(counts[0])}, fake={int(counts[1])}. Для сравнения используем macro-F1 и метрики каждого класса.\n"
			f"- Медианная длина: real={medians[0]:.0f}, fake={medians[1]:.0f} символов. Различие длин — повод проверить простые сигналы корпуса, не доказательство истинности новости.\n"
			"- Удалены точные повторы нормализованного текста; близкие перепечатки и зависимость от источников пока не исключены. Test-тексты здесь не анализировались.\n"
		)
		mlflow.log_text(notes, "eda/notes.md")
		print(notes)


if __name__ == "__main__":
	main()
