import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def throughput_comparison_power_modes_plot(power_mode_files, output_path, value_key="throughput_images_per_s"):
    """
    Erstellt einen Throughput-Vergleich mehrerer Power-Modi als Liniendiagramm.

    Args:
        power_mode_files: Dict mit Label -> JSON-Pfad, z.B.
            {
                "15W": ".../throughput_15w.json",
                "30W": ".../throughput_30w.json",
                "50W": ".../throughput_50w.json",
            }
        output_path: Ausgabe-Pfad fuer den Plot.
        value_key: Zu plottender Throughput-Key,
            "throughput_images_per_s" oder "throughput_batches_per_s".
    """
    if not power_mode_files:
        print("WARNING: No power mode files provided")
        return

    mode_data = {}
    all_batch_sizes = set()

    for mode_label, json_path in power_mode_files.items():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: File not found for mode {mode_label}: {json_path}")
            continue
        except json.JSONDecodeError:
            print(f"WARNING: Invalid JSON for mode {mode_label}: {json_path}")
            continue

        if not data:
            print(f"WARNING: No data in {json_path} (mode {mode_label})")
            continue

        values_by_batch = {}
        for entry in data:
            bs = entry.get("batch_size")
            value = entry.get(value_key)
            if bs is None or value is None:
                continue
            values_by_batch[bs] = value
            all_batch_sizes.add(bs)

        if not values_by_batch:
            print(f"WARNING: No valid '{value_key}' values for mode {mode_label}")
            continue

        mode_data[mode_label] = values_by_batch

    if not mode_data or not all_batch_sizes:
        print("WARNING: No valid data available to create power mode comparison plot")
        return

    batch_sizes = sorted(all_batch_sizes)

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, (mode_label, values_by_batch) in enumerate(mode_data.items()):
        x_vals = []
        y_vals = []
        for bs in batch_sizes:
            if bs in values_by_batch:
                x_vals.append(bs)
                y_vals.append(values_by_batch[bs])

        if not x_vals:
            continue

        ax.plot(
            x_vals,
            y_vals,
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=mode_label,
        )

    ylabel = "Throughput (images/s)" if value_key == "throughput_images_per_s" else "Throughput (batches/s)"
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title("Throughput Comparison per Power Mode")
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Power Mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def parse_mode_args(mode_args):
    power_mode_files = {}
    for mode_arg in mode_args:
        if "=" not in mode_arg:
            raise ValueError(f"Invalid --mode value '{mode_arg}'. Expected format LABEL=/path/to/file.json")
        label, path = mode_arg.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --mode value '{mode_arg}'. Label or path is empty")
        power_mode_files[label] = path
    return power_mode_files


def discover_modes_from_input_dir(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("throughput_*w.json"))
    power_mode_files = {}

    for file_path in files:
        stem = file_path.stem.lower()
        label = stem.replace("throughput_", "").upper()
        power_mode_files[label] = str(file_path)

    return power_mode_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate throughput comparison plot per power mode (one line per mode)."
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Power mode mapping in the form LABEL=/path/to/file.json. Can be used multiple times.",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/hanna/git/finn-transformers/summary_plots/data/vision_base4_int8",
        help="Directory for auto-discovery of files named throughput_*w.json.",
    )
    parser.add_argument(
        "--output",
        default="/home/hanna/git/finn-transformers/summary_plots/outputs/vision_base4_int8/throughput_power_modes_comparison.png",
        help="Output path for the generated plot image.",
    )
    parser.add_argument(
        "--value-key",
        default="throughput_images_per_s",
        choices=["throughput_images_per_s", "throughput_batches_per_s"],
        help="Which throughput metric to plot.",
    )

    args = parser.parse_args()

    if args.mode:
        power_mode_files = parse_mode_args(args.mode)
    else:
        power_mode_files = discover_modes_from_input_dir(args.input_dir)

    throughput_comparison_power_modes_plot(
        power_mode_files=power_mode_files,
        output_path=args.output,
        value_key=args.value_key,
    )


if __name__ == "__main__":
    main()
