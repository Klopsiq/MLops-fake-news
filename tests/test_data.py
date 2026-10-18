import pandas as pd
import pytest

from ml.data import clean_data, prepare, read_split


def test_clean_removes_duplicates_conflicts_and_invalid_rows():
	raw = pd.DataFrame(
		{
			"text": [
				" News one ",
				"news  ONE",
				"Conflict",
				"CONFLICT",
				"",
				None,
				"Bad",
				"Second",
			],
			"label": [1, 1, 0, 1, 0, 1, 9, 0],
		}
	)
	frame, quality = clean_data(raw, fake_label=0)
	assert frame.text.tolist() == ["News one", "Second"]
	assert frame.label.tolist() == [0, 1]
	assert quality == {
		"raw_rows": 8,
		"removed_empty_text": 2,
		"removed_invalid_label": 1,
		"removed_conflicting_rows": 2,
		"removed_duplicate_rows": 1,
		"clean_rows": 2,
	}


def test_split_is_repeatable_disjoint_and_detects_tampering(tmp_path):
	raw = tmp_path / "source.csv"
	pd.DataFrame(
		{
			"text": [f"Unique article number {i}" for i in range(100)],
			"label": [i % 2 for i in range(100)],
		}
	).to_csv(raw, index=False)
	output = tmp_path / "processed"
	first = prepare(raw, output, fake_label=1)
	assert prepare(raw, output, fake_label=1) == first
	frames = [read_split(name, output)[0] for name in ("train", "validation", "test")]
	assert [len(frame) for frame in frames] == [70, 15, 15]
	assert len(set().union(*(set(frame.text_hash) for frame in frames))) == 100
	assert len(set().union(*(set(frame.row_id) for frame in frames))) == 100
	path = output / "train.csv"
	path.write_text(path.read_text() + "\n")
	with pytest.raises(ValueError, match="SHA-256"):
		read_split("train", output)
