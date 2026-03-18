import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_TYPE = os.environ.get("MODEL_TYPE", "radioml")
if MODEL_TYPE not in ("radioml", "vision", "language"):
    MODEL_TYPE = "radioml"
    print("Defaulting MODEL_TYPE to radioml.")

# Expected variants for this comparison are exactly two values,
# e.g. "INT8,ORT_INT8" or "FP32,ORT_FP32".
_variants_env = os.environ.get("VARIANTS", "").strip()
variants = [v.strip() for v in _variants_env.split(",") if v.strip()]
if len(variants) != 2:
    print("VARIANTS must contain exactly two entries, e.g. FP32,ORT_FP32")
    raise SystemExit(0)

trt_variant, ort_variant = variants

EVAL_BASE = Path(f"outputs/{MODEL_TYPE}/eval_results")
PLOT_BASE = Path(f"outputs/{MODEL_TYPE}/plot")

# Mapping from compare-variant name to eval filename pattern
# TensorRT: INT8/FP16/FP32 -> accuracy_INT8.json / accuracy_FP16.json / accuracy_FP32.json
# ORT:      ORT_INT8/ORT_FP16/ORT_FP32 -> accuracy_ORT_INT8.json / ...
def to_accuracy_filename(variant: str) -> str:
    if variant.startswith("ORT_"):
        suffix = variant.replace("ORT_", "", 1)
        return f"accuracy_ORT_{suffix}.json"
    return f"accuracy_{variant}.json"


def read_accuracy(path: Path) -> tuple[str, float]:
    with open(path, "r") as f:
        data = json.load(f)

    # Handle both dict and list-of-dict defensively
    if isinstance(data, list):
        if not data:
            raise ValueError(f"No accuracy entries in {path}")
        data = data[0]

    label = str(data.get("quantisation_type", path.stem))
    value = float(data.get("value", 0.0))
    return label, value


trt_file = EVAL_BASE / to_accuracy_filename(trt_variant)
ort_file = EVAL_BASE / to_accuracy_filename(ort_variant)

if not trt_file.exists() or not ort_file.exists():
    missing = [str(p) for p in (trt_file, ort_file) if not p.exists()]
    print(f"Missing accuracy files: {missing}")
    raise SystemExit(0)

trt_label, trt_acc = read_accuracy(trt_file)
ort_label, ort_acc = read_accuracy(ort_file)

x_labels = [trt_label, ort_label]
y_values = [trt_acc * 100.0, ort_acc * 100.0]  # show in %

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#1f77b4", "#ff7f0e"]
bars = ax.bar(x_labels, y_values, color=colors, width=0.55)

for b, y in zip(bars, y_values):
    ax.text(
        b.get_x() + b.get_width() / 2,
        y + 0.2,
        f"{y:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_ylabel("Accuracy (%)")
ax.set_title(f"{MODEL_TYPE.capitalize()} – Accuracy Comparison (Batch Size = 1)")
ax.set_ylim(0, max(100.0, max(y_values) * 1.1))
ax.grid(axis="y", linestyle="--", alpha=0.5)

suffix = f"{trt_variant}_{ort_variant}"
output_path = PLOT_BASE / f"accuracy_comparison_{suffix}.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close(fig)

print(f"Saved → {output_path}")
