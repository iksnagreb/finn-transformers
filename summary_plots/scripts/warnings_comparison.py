"""Compare vision base throughput (30 W) with and without TensorRT build warnings.

with warnings:    ci-152127-602776  (trt.Logger.WARNING)
no warnings:      ci-152179-602967  (trt.Logger.INTERNAL_ERROR)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_for_plots"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "vision_base_warnings" / "throughput_warnings_comparison_30w.png"

RUNS = {
	"with warnings": DATA_ROOT / "base",
	"no warnings": DATA_ROOT / "base_no_warnings",
}
MODES = {"int8": "#2a78d6", "ort_int8": "#eb6834"}
LINESTYLES = {"with warnings": "-", "no warnings": "--"}
MARKERS = {"with warnings": "o", "no warnings": "s"}


def load_throughput(path):
	with open(path) as f:
		rows = json.load(f)
	return {row["batch_size"]: row["throughput_images_per_s"] for row in rows}


def main():
	data = {
		(mode, run): load_throughput(folder / mode / "throughput_30w.json")
		for mode in MODES
		for run, folder in RUNS.items()
	}

	fig, (ax_abs, ax_rel) = plt.subplots(
		2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
	)

	for mode, color in MODES.items():
		for run in RUNS:
			values = data[(mode, run)]
			batches = sorted(values)
			ax_abs.plot(
				batches,
				[values[b] for b in batches],
				color=color,
				linestyle=LINESTYLES[run],
				marker=MARKERS[run],
				markersize=7,
				linewidth=2,
				label=f"{mode.upper()} – {run}",
			)

	bar_width = 0.32
	for i, (mode, color) in enumerate(MODES.items()):
		warn = data[(mode, "with warnings")]
		no_warn = data[(mode, "no warnings")]
		batches = sorted(set(warn) & set(no_warn))
		diffs = [(no_warn[b] - warn[b]) / warn[b] * 100 for b in batches]
		offset = (i - 0.5) * bar_width
		positions = [b * 2 ** offset for b in batches]
		bars = ax_rel.bar(
			positions,
			diffs,
			width=[p * bar_width * 0.65 for p in positions],
			color=color,
			edgecolor="white",
			linewidth=1,
			label=mode.upper(),
		)
		ax_rel.bar_label(bars, labels=[f"{d:+.1f}%" for d in diffs], fontsize=8, padding=3, rotation=90)
		print(f"{mode}: " + ", ".join(f"bs{b}: {d:+.2f}%" for b, d in zip(batches, diffs)))

	ax_abs.set_xscale("log", base=2)
	ax_abs.set_ylabel("Throughput [images/s]")
	ax_abs.set_title("Vision base, 30 W: throughput with vs. without TensorRT warnings")
	ax_abs.grid(True, which="major", alpha=0.3)
	ax_abs.legend()

	ax_rel.axhline(0, color="black", linewidth=0.8)
	ax_rel.set_xlabel("Batch size")
	ax_rel.set_ylabel("Δ throughput no warnings vs. warnings [%]")
	ax_rel.grid(True, axis="y", alpha=0.3)
	limit = max(abs(v) for v in ax_rel.get_ylim())
	ax_rel.set_ylim(-max(limit, 5), max(limit, 5) * 1.35)
	ax_rel.legend()
	batches = sorted(data[("int8", "with warnings")])
	ax_rel.set_xticks(batches)
	ax_rel.set_xticklabels([str(b) for b in batches])

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(OUTPUT_PATH, dpi=200)
	print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
