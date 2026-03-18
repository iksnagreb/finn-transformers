import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_TYPE = os.environ.get("MODEL_TYPE", "radioml")
if MODEL_TYPE not in ("radioml", "vision", "language"):
    MODEL_TYPE = "radioml"
    print("Defaulting MODEL_TYPE to radioml.")

# Optional: comma-separated list of variants to include, e.g. "FP32,ORT_FP32"
# Leave unset (or empty) to include all available variants.
_variants_env = os.environ.get("VARIANTS", "").strip()
VARIANTS_FILTER = [v.strip() for v in _variants_env.split(",") if v.strip()] if _variants_env else []

PLOT_BASE = Path(f"outputs/{MODEL_TYPE}/plot")
# Output filename: throughput_comparison.png  OR  throughput_comparison_FP32_ORT_FP32.png
_suffix = "_" + "_".join(VARIANTS_FILTER) if VARIANTS_FILTER else ""
OUTPUT_PATH = PLOT_BASE / f"throughput_comparison{_suffix}.png"

# Display labels / styles per precision variant.
# Convention mirrors measure.py (TRT) and measure_onnxruntime.py (ORT):
#   TensorRT : INT8  | FP16  | FP32
#   ORT CUDA : ORT_INT8 | ORT_FP16 | ORT_FP32
VARIANT_STYLES = {
    "INT8":     {"label": "TensorRT INT8",  "marker": "o", "color": "#1f77b4"},
    "FP16":     {"label": "TensorRT FP16",  "marker": "^", "color": "#2ca02c"},
    "FP32":     {"label": "TensorRT FP32",  "marker": "D", "color": "#d62728"},
    "ORT_INT8": {"label": "ORT INT8",        "marker": "s", "color": "#ff7f0e"},
    "ORT_FP16": {"label": "ORT FP16",        "marker": "v", "color": "#9467bd"},
    "ORT_FP32": {"label": "ORT FP32",        "marker": "P", "color": "#8c564b"},
}


def load(path):
    with open(path) as f:
        data = json.load(f)
    batch_sizes = [d["batch_size"] for d in data]
    throughput  = [d["throughput_images_per_s"] for d in data]
    return batch_sizes, throughput


# --- Auto-discover available precision variants (optionally filtered) ---
variants = {}  # name → Path
for variant_dir in sorted(PLOT_BASE.iterdir()):
    if not variant_dir.is_dir():
        continue
    if VARIANTS_FILTER and variant_dir.name not in VARIANTS_FILTER:
        continue
    tp_file = variant_dir / "throughput_results.json"
    if tp_file.exists():
        variants[variant_dir.name] = tp_file

if not variants:
    print(f"No throughput data found under {PLOT_BASE}")
    raise SystemExit(0)

# Load and plot
all_batch_sizes: set = set()
plt.figure(figsize=(9, 5))
for name, path in variants.items():
    bs, tp = load(path)
    style  = VARIANT_STYLES.get(name, {"label": name, "marker": "o", "color": None})
    plt.plot(bs, tp, marker=style["marker"], label=style["label"], color=style.get("color"))
    all_batch_sizes.update(bs)

all_batch_sizes = sorted(all_batch_sizes)
plt.xscale("log", base=2)
plt.xticks(all_batch_sizes, all_batch_sizes)
plt.xlabel("Batch Size")
plt.ylabel("Throughput (images / s)")
variant_names = " vs. ".join(
    VARIANT_STYLES.get(name, {"label": name})["label"] for name in variants
)
plt.title(f"{MODEL_TYPE.capitalize()} – Throughput: {variant_names}")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
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