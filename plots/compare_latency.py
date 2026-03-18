import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
# todo: Pfade anpassen und alles für language und vision machen


MODEL_TYPE = os.environ.get("MODEL_TYPE", "radioml")
if MODEL_TYPE not in ("radioml", "vision", "language"):
    MODEL_TYPE = "radioml"
    print("Defaulting MODEL_TYPE to radioml.")

# Optional: comma-separated list of variants to include, e.g. "FP32,ORT_FP32"
# Leave unset (or empty) to include all available variants.
_variants_env = os.environ.get("VARIANTS", "").strip()
VARIANTS_FILTER = [v.strip() for v in _variants_env.split(",") if v.strip()] if _variants_env else []

PLOT_BASE = Path(f"outputs/{MODEL_TYPE}/plot")
# Output filename: latency_comparison.png  OR  latency_comparison_FP32_ORT_FP32.png
_suffix = "_" + "_".join(VARIANTS_FILTER) if VARIANTS_FILTER else ""
OUTPUT_PATH = PLOT_BASE / f"latency_comparison{_suffix}.png"

# Display labels / styles per precision variant.
# Convention mirrors measure.py (TRT) and measure_onnxruntime.py (ORT):
#   TensorRT : INT8  | FP16  | FP32
#   ORT CUDA : ORT_INT8 | ORT_FP16 | ORT_FP32

# "INT8":     {"label": "TensorRT INT8",  "marker": "o", "color": "#1f77b4"},
# "FP16":     {"label": "TensorRT FP16",  "marker": "^", "color": "#2ca02c"},
# "ORT_INT8": {"label": "ORT INT8",        "marker": "s", "color": "#ff7f0e"},
# "ORT_FP16": {"label": "ORT FP16",        "marker": "v", "color": "#9467bd"},
VARIANT_STYLES = {
    "FP32":     {"label": "TensorRT FP32",  "marker": "D", "color": "#d62728"},
    "ORT_FP32": {"label": "ORT FP32",        "marker": "P", "color": "#8c564b"},
}
HATCHES = ["", "//", "xx", "..", "||"]

LATENCY_TYPES  = ["inference", "synchronize", "datatransfer"]
LATENCY_LABELS = {
    "inference":    "Inference",
    "synchronize":  "Synchronize overhead",
    "datatransfer": "Data transfer overhead",
}
TYPE_COLORS = {
    "inference":    "#1f77b4",
    "synchronize":  "#ff7f0e",
    "datatransfer": "#2ca02c",
}


def load_totals(path):
    """Sum all latency types per batch_size → total latency per batch."""
    with open(path) as f:
        data = json.load(f)
    totals = defaultdict(float)
    for entry in data:
        totals[entry["batch_size"]] += entry["value"]
    batch_sizes = sorted(totals.keys())
    return batch_sizes, [totals[bs] for bs in batch_sizes]


def load_by_type(path):
    """Return {latency_type: {batch_size: value}}."""
    with open(path) as f:
        data = json.load(f)
    result = defaultdict(dict)
    for entry in data:
        # normalize TRT typo 'inteference' → 'inference'
        ltype = entry["type"].replace("inteference", "inference")
        result[ltype][entry["batch_size"]] = entry["value"]
    return result


# --- Auto-discover available precision variants (optionally filtered) ---
variants = {}  # name → Path
for variant_dir in sorted(PLOT_BASE.iterdir()):
    if not variant_dir.is_dir():
        continue
    if VARIANTS_FILTER and variant_dir.name not in VARIANTS_FILTER:
        continue
    lat_file = variant_dir / "latency_results_batch.json"
    if lat_file.exists():
        variants[variant_dir.name] = lat_file

if not variants:
    print(f"No latency data found under {PLOT_BASE}")
    raise SystemExit(0)

# Load all data
all_batch_sizes: set = set()
loaded = {}
for name, path in variants.items():
    bs, lat = load_totals(path)
    by_type = load_by_type(path)
    loaded[name] = {"bs": bs, "lat": lat, "by_type": by_type}
    all_batch_sizes.update(bs)

batch_sizes = sorted(all_batch_sizes)
n_variants  = len(loaded)
bar_width   = 0.8 / max(n_variants, 1)

fig, axes = plt.subplots(1, 2, figsize=(7 + 3 * n_variants, 5), sharey=False)

# --- Left: total latency per batch ---
ax = axes[0]
for name, d in loaded.items():
    style = VARIANT_STYLES.get(name, {"label": name, "marker": "o", "color": None})
    ax.plot(d["bs"], d["lat"], marker=style["marker"],
            label=style["label"], color=style.get("color"))
ax.set_xscale("log", base=2)
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Total Latency per Batch (ms)")
ax.set_title("Total Latency per Batch")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.5)

# --- Right: stacked bar breakdown ---
ax = axes[1]
x = range(len(batch_sizes))
for model_idx, (name, d) in enumerate(loaded.items()):
    style  = VARIANT_STYLES.get(name, {"label": name})
    hatch  = HATCHES[model_idx % len(HATCHES)]
    alpha  = 0.7 if model_idx % 2 == 0 else 1.0
    bottom = [0.0] * len(batch_sizes)
    for ltype in LATENCY_TYPES:
        values = [d["by_type"][ltype].get(bs, 0.0) for bs in batch_sizes]
        ax.bar(
            [xi + model_idx * bar_width for xi in x],
            values, bar_width,
            bottom=bottom,
            label=f"{style['label']} – {LATENCY_LABELS[ltype]}",
            color=TYPE_COLORS[ltype],
            alpha=alpha,
            hatch=hatch,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

offset = (n_variants - 1) * bar_width / 2
ax.set_xticks([xi + offset for xi in x])
ax.set_xticklabels(batch_sizes, rotation=45)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Latency (ms)")
ax.set_title("Latency Breakdown per Batch")
ax.legend(fontsize=7, ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.5)

variant_names = " vs. ".join(
    VARIANT_STYLES.get(name, {"label": name})["label"] for name in loaded
)
plt.suptitle(f"{MODEL_TYPE.capitalize()} – Latency per Batch: {variant_names}")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()
print(f"Saved → {OUTPUT_PATH}")


# python3 -m plots.compare_throughput
# python3 -m plots.compare_latency
# VARIANTS=FP16,ORT_FP16 python -m plots.compare_latency
# VARIANTS=FP16,ORT_FP16 python -m plots.compare_throughput
# VARIANTS=FP32,ORT_FP32 python -m plots.compare_latency
# VARIANTS=FP32,ORT_FP32 python -m plots.compare_throughput
# VARIANTS=INT8,ORT_INT8 python -m plots.compare_latency
# VARIANTS=INT8,ORT_INT8 python -m plots.compare_throughput