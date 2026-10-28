import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
	root = Path("docs/experiments")
	fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
	metrics = [
		("ready", "Ready queue depth", "messages"),
		("rate", "Source accepted rate", "tasks / second"),
		("mean_task_seconds", "Mean completed task latency (cumulative)", "seconds"),
		("source_rss_mb", "Source + experiment controller RSS", "MiB"),
		("worker_rss_mb", "Workers: total Docker memory usage", "MiB"),
		("unacked", "Unacknowledged deliveries", "messages"),
	]
	for label in ["unbounded-1", "bounded-1", "bounded-2"]:
		with (root / f"{label}.csv").open() as stream:
			rows = list(csv.DictReader(stream))
		for ax, (key, title, unit) in zip(axes.flat, metrics, strict=True):
			ax.plot(
				[float(r["seconds"]) for r in rows],
				[float(r[key]) for r in rows],
				marker=".",
				label=label,
			)
			ax.set(title=title, xlabel="Elapsed seconds", ylabel=unit)
			ax.grid(alpha=0.25)
	axes[0, 0].axhline(20, linestyle="--", color="gray", label="bounded limit")
	axes[0, 0].legend()
	fig.suptitle(
		"HW4 / WELFake / MLflow champion: 200 tasks, 2 threads per worker, 0.2 s delay"
	)
	fig.savefig(root / "comparison.png", dpi=150)
	plt.close(fig)


if __name__ == "__main__":
	main()
