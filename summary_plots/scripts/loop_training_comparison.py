"""Compare vision base2 FP32 trained with a repeated layer loop vs. individual layers.

loop (shared layers):  data_for_plots/base2/fp32             ci-150714-597259 / ci-150695-597208 / ci-149142-590753
individual layers:     data_for_plots/base2_individual/fp32  ci-160990-631006 / ci-160980-630979 / ci-159647-627017

All plots are written to summary_plots/outputs/loop_training_comparison/.
Relative differences are always (individual - loop) / loop.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_for_plots"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "loop_training_comparison"

POWER_MODES = ("15w", "30w", "50w")
RUNS = {
	"loop": DATA_ROOT / "base2",
	"individual": DATA_ROOT / "base2_individual",
}
LABELS = {"loop": "Loop (shared layers)", "individual": "Individual layers"}
COLORS = {"loop": "#2a78d6", "individual": "#eb6834"}
LINESTYLES = {"loop": "-", "individual": "--"}
MARKERS = {"loop": "o", "individual": "s"}
# Diverging: orange (individual lower) - neutral gray - blue (individual higher)
DIVERGING = LinearSegmentedColormap.from_list("loop_div", ["#eb6834", "#e6e6e3", "#2a78d6"])

# metric key -> (axis label, short name, unit hint for the Δ axis)
METRICS = {
	"throughput": ("Throughput [images/s]", "throughput", "higher is better"),
	"latency": ("Latency [ms]", "latency", "lower is better"),
	"efficiency": ("Energy efficiency [images/J]", "energy efficiency", "higher is better"),
	"energy": ("Energy per image [mJ]", "energy per image", "lower is better"),
	"power": ("Average power [W]", "power", "lower is better"),
}


def _load_json(path):
	with open(path) as f:
		return json.load(f)


def load_run(run_dir, power):
	"""Return {metric: {batch_size: value}} for one run and power mode."""
	fp32 = run_dir / "fp32"
	throughput = {r["batch_size"]: r["throughput_images_per_s"] for r in _load_json(fp32 / f"throughput_{power}.json")}

	latency = {}
	breakdown = {}
	for r in _load_json(fp32 / f"latency_{power}.json"):
		latency[r["batch_size"]] = latency.get(r["batch_size"], 0.0) + r["value"]
		breakdown.setdefault(r["type"], {})[r["batch_size"]] = r["value"]

	power_rows = _load_json(fp32 / f"power_throughput_{power}.json")
	power_w = {r["batch_size"]: r["power_total"] / 1000 for r in power_rows}
	efficiency = {r["batch_size"]: r["throughput_images_per_s"] / (r["power_total"] / 1000) for r in power_rows}
	energy = {bs: 1000 / value for bs, value in efficiency.items()}

	latency_throughput = {
		r["batch_size"]: (r["latency_total"], r["throughput_images_per_s"])
		for r in _load_json(fp32 / f"latency_throughput_{power}.json")
	}

	return {
		"throughput": throughput,
		"latency": latency,
		"breakdown": breakdown,
		"efficiency": efficiency,
		"energy": energy,
		"power": power_w,
		"latency_throughput": latency_throughput,
	}


def load_all():
	return {(run, power): load_run(run_dir, power) for run, run_dir in RUNS.items() for power in POWER_MODES}


def relative_diff(data, metric, power):
	loop = data[("loop", power)][metric]
	individual = data[("individual", power)][metric]
	batches = sorted(set(loop) & set(individual))
	return batches, [(individual[b] - loop[b]) / loop[b] * 100 for b in batches]


def _style_batch_axis(ax, batches):
	ax.set_xscale("log", base=2)
	ax.set_xticks(batches)
	ax.set_xticklabels([str(b) for b in batches])
	ax.minorticks_off()


def _save(fig, name):
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	path = OUTPUT_DIR / name
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	print(f"Saved {path}")


def metric_comparison_plot(data, metric, filename, log_y=False):
	"""Absolute values per power mode (top) and Δ individual vs. loop in % (bottom)."""
	label, name, hint = METRICS[metric]
	fig, axes = plt.subplots(
		2, len(POWER_MODES), figsize=(17, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
	)

	diff_limit = 5.0
	for col, power in enumerate(POWER_MODES):
		ax_abs, ax_rel = axes[0, col], axes[1, col]
		for run in RUNS:
			values = data[(run, power)][metric]
			batches = sorted(values)
			ax_abs.plot(
				batches,
				[values[b] for b in batches],
				color=COLORS[run],
				linestyle=LINESTYLES[run],
				marker=MARKERS[run],
				markersize=7,
				linewidth=2,
				label=LABELS[run],
			)
		if log_y:
			ax_abs.set_yscale("log")
		ax_abs.set_title(power.upper())
		ax_abs.grid(True, alpha=0.3)

		batches, diffs = relative_diff(data, metric, power)
		bars = ax_rel.bar(
			batches,
			diffs,
			width=[b * 0.45 for b in batches],
			color=["#6f6e69" if d >= 0 else "#a3a29b" for d in diffs],
			edgecolor="white",
			linewidth=1,
		)
		ax_rel.bar_label(bars, labels=[f"{d:+.1f}%" for d in diffs], fontsize=8, padding=3, rotation=90)
		ax_rel.axhline(0, color="black", linewidth=0.8)
		ax_rel.grid(True, axis="y", alpha=0.3)
		ax_rel.set_xlabel("Batch size")
		_style_batch_axis(ax_rel, batches)
		diff_limit = max(diff_limit, *(abs(d) for d in diffs))

	for ax_rel in axes[1]:
		ax_rel.set_ylim(-diff_limit * 1.6, diff_limit * 1.6)
	axes[0, 0].set_ylabel(label)
	axes[1, 0].set_ylabel(f"Δ {name}\nindividual vs. loop [%]")
	axes[0, 0].legend()
	fig.suptitle(f"Vision base2 FP32: {name}, loop vs. individual layers ({hint})", fontsize=14)
	_save(fig, filename)


def latency_throughput_plot(data):
	fig, axes = plt.subplots(1, len(POWER_MODES), figsize=(17, 5.5), sharey=True)
	for ax, power in zip(axes, POWER_MODES):
		for run in RUNS:
			points = data[(run, power)]["latency_throughput"]
			batches = sorted(points)
			ax.plot(
				[points[b][0] for b in batches],
				[points[b][1] for b in batches],
				color=COLORS[run],
				linestyle=LINESTYLES[run],
				marker=MARKERS[run],
				markersize=7,
				linewidth=2,
				label=LABELS[run],
			)
			if run == "loop":
				for b in batches[:: 2]:
					ax.annotate(f"bs {b}", points[b], textcoords="offset points", xytext=(6, -12), fontsize=7, color="#52514e")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_title(power.upper())
		ax.set_xlabel("Latency [ms]")
		ax.grid(True, which="both", alpha=0.3)
	axes[0].set_ylabel("Throughput [images/s]")
	axes[0].legend()
	fig.suptitle("Vision base2 FP32: latency–throughput trade-off, loop vs. individual layers", fontsize=14)
	_save(fig, "latency_throughput_comparison_logxy.png")


def latency_breakdown_plot(data):
	"""Grouped stacked bars: loop | individual per batch size, split into latency components."""
	types = ("inference", "datatransfer", "synchronize")
	alpha = {"inference": 1.0, "datatransfer": 0.55, "synchronize": 0.25}
	fig, axes = plt.subplots(1, len(POWER_MODES), figsize=(17, 5.5))
	for ax, power in zip(axes, POWER_MODES):
		batches = sorted(data[("loop", power)]["latency"])
		width = 0.38
		for offset, run in ((-width / 2 - 0.01, "loop"), (width / 2 + 0.01, "individual")):
			bottom = [0.0] * len(batches)
			for latency_type in types:
				values = [data[(run, power)]["breakdown"].get(latency_type, {}).get(b, 0.0) for b in batches]
				ax.bar(
					[i + offset for i in range(len(batches))],
					values,
					width=width,
					bottom=bottom,
					color=COLORS[run],
					alpha=alpha[latency_type],
					edgecolor="white",
					linewidth=1,
					label=f"{LABELS[run]} – {latency_type}",
				)
				bottom = [b + v for b, v in zip(bottom, values)]
		ax.set_yscale("log")
		ax.set_xticks(range(len(batches)))
		ax.set_xticklabels([str(b) for b in batches])
		ax.set_xlabel("Batch size")
		ax.set_title(power.upper())
		ax.grid(True, axis="y", which="both", alpha=0.3)
	axes[0].set_ylabel("Latency [ms] (log)")
	axes[0].legend(fontsize=7, ncol=2)
	fig.suptitle("Vision base2 FP32: latency breakdown, loop vs. individual layers", fontsize=14)
	_save(fig, "latency_breakdown_comparison.png")


def relative_difference_heatmap(data):
	metrics = ("throughput", "latency", "efficiency", "power")
	fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 12), gridspec_kw={"hspace": 0.55})
	matrices = {}
	for metric in metrics:
		rows = []
		for power in POWER_MODES:
			batches, diffs = relative_diff(data, metric, power)
			rows.append(diffs)
		matrices[metric] = (batches, rows)
	limit = max(abs(v) for _, rows in matrices.values() for row in rows for v in row)

	for ax, metric in zip(axes, metrics):
		batches, rows = matrices[metric]
		image = ax.imshow(rows, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
		for i, row in enumerate(rows):
			for j, value in enumerate(row):
				ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="#1a1a19")
		ax.set_xticks(range(len(batches)))
		ax.set_xticklabels([str(b) for b in batches])
		ax.set_yticks(range(len(POWER_MODES)))
		ax.set_yticklabels([p.upper() for p in POWER_MODES])
		label, name, hint = METRICS[metric]
		ax.set_title(f"Δ {name} [%] ({hint})", fontsize=11)
	axes[-1].set_xlabel("Batch size")
	fig.colorbar(image, ax=axes, shrink=0.6, label="individual vs. loop [%]")
	fig.suptitle("Vision base2 FP32: relative difference individual vs. loop", fontsize=14)
	fig.savefig(OUTPUT_DIR / "relative_difference_heatmap.png", dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {OUTPUT_DIR / 'relative_difference_heatmap.png'}")


def load_accuracy():
	accuracy = {}
	for run, run_dir in RUNS.items():
		with open(run_dir / "accuracy.yaml") as f:
			accuracy[run] = yaml.safe_load(f)
	return accuracy


def accuracy_plot(accuracy):
	keys = ("top-1", "top-5")
	fig, ax = plt.subplots(figsize=(7, 5.5))
	width = 0.36
	for offset, run in ((-width / 2 - 0.01, "loop"), (width / 2 + 0.01, "individual")):
		values = [accuracy[run][k] * 100 for k in keys]
		bars = ax.bar(
			[i + offset for i in range(len(keys))],
			values,
			width=width,
			color=COLORS[run],
			edgecolor="white",
			linewidth=1,
			label=LABELS[run],
		)
		ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], fontsize=9, padding=3)
	for i, k in enumerate(keys):
		delta = (accuracy["individual"][k] - accuracy["loop"][k]) * 100
		ax.annotate(f"Δ {delta:+.2f} pp", (i, 104), ha="center", fontsize=9, color="#52514e")
	ax.set_xticks(range(len(keys)))
	ax.set_xticklabels([k.capitalize() for k in keys])
	ax.set_ylabel("Accuracy (CIFAR-10 eval) [%]")
	ax.set_ylim(0, 110)
	ax.grid(True, axis="y", alpha=0.3)
	ax.legend(loc="lower right")
	ax.set_title("Vision base2: accuracy, loop vs. individual layers")
	_save(fig, "accuracy_comparison.png")


def load_loss():
	loss = {}
	for run, run_dir in RUNS.items():
		with open(run_dir / "loss.yaml") as f:
			loss[run] = yaml.safe_load(f)["loss"]
	return loss


def training_loss_plot(loss):
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	for ax, split in zip(axes, ("train", "valid")):
		for run in RUNS:
			values = [epoch[split] for epoch in loss[run]]
			ax.plot(
				range(1, len(values) + 1),
				values,
				color=COLORS[run],
				linestyle=LINESTYLES[run],
				linewidth=2,
				label=LABELS[run],
			)
			ax.annotate(
				f"{values[-1]:.1f}",
				(len(values), values[-1]),
				textcoords="offset points",
				xytext=(4, 0),
				fontsize=8,
				color="#52514e",
				va="center",
			)
		ax.set_xlim(right=len(values) + 15)
		ax.set_title(f"{split.capitalize()} loss")
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss (summed over batches)")
		ax.grid(True, alpha=0.3)
	axes[0].legend()
	fig.suptitle("Vision base2: training curves, loop vs. individual layers", fontsize=14)
	_save(fig, "training_loss_comparison.png")


def write_summary_csv(data, accuracy):
	path = OUTPUT_DIR / "summary.csv"
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	with open(path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["metric", "power", "batch_size", "loop", "individual", "diff_percent"])
		for key in ("top-1", "top-5"):
			loop, individual = accuracy["loop"][key], accuracy["individual"][key]
			writer.writerow([f"accuracy_{key}", "", "", loop, individual, f"{(individual - loop) / loop * 100:.3f}"])
		for metric in ("throughput", "latency", "efficiency", "energy", "power"):
			for power in POWER_MODES:
				batches, diffs = relative_diff(data, metric, power)
				for b, d in zip(batches, diffs):
					writer.writerow([
						metric,
						power,
						b,
						f"{data[('loop', power)][metric][b]:.4f}",
						f"{data[('individual', power)][metric][b]:.4f}",
						f"{d:.3f}",
					])
	print(f"Saved {path}")


def print_overview(data, accuracy):
	for key in ("top-1", "top-5"):
		print(f"accuracy {key}: loop {accuracy['loop'][key]:.4f}, individual {accuracy['individual'][key]:.4f}")
	for metric in ("throughput", "latency", "efficiency"):
		for power in POWER_MODES:
			_, diffs = relative_diff(data, metric, power)
			mean = sum(diffs) / len(diffs)
			print(f"{metric:10s} {power}: mean Δ {mean:+.2f}%, min {min(diffs):+.2f}%, max {max(diffs):+.2f}%")


def main():
	data = load_all()
	accuracy = load_accuracy()

	metric_comparison_plot(data, "throughput", "throughput_comparison.png")
	metric_comparison_plot(data, "latency", "latency_comparison.png", log_y=True)
	metric_comparison_plot(data, "efficiency", "energy_efficiency_comparison.png")
	metric_comparison_plot(data, "energy", "energy_per_image_comparison.png", log_y=True)
	metric_comparison_plot(data, "power", "power_comparison.png")
	latency_throughput_plot(data)
	latency_breakdown_plot(data)
	relative_difference_heatmap(data)
	accuracy_plot(accuracy)
	training_loss_plot(load_loss())
	write_summary_csv(data, accuracy)
	print_overview(data, accuracy)


if __name__ == "__main__":
	main()
