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


def throughput_comparison_model_families_plot(model_family_files, output_path, value_key="throughput_images_per_s"):
    """
    Erstellt einen kombinierten Throughput-Vergleich mehrerer Modellfamilien
    und Power-Modi in einem Diagramm.

    Args:
        model_family_files: Dict mit Modellfamilien-Label -> Dict von Power-Mode-Label
            zu JSON-Pfad, z.B.
            {
                "base2": {
                    "15W": ".../vision_base2_int8/throughput_15w.json",
                    "30W": ".../vision_base2_int8/throughput_30w.json",
                    "50W": ".../vision_base2_int8/throughput_50w.json",
                },
                "base4": {
                    "15W": ".../vision_base4_int8/throughput_15w.json",
                    "30W": ".../vision_base4_int8/throughput_30w.json",
                    "50W": ".../vision_base4_int8/throughput_50w.json",
                },
            }
        output_path: Ausgabe-Pfad fuer den Plot.
        value_key: Zu plottender Throughput-Key,
            "throughput_images_per_s" oder "throughput_batches_per_s".
    """
    if not model_family_files:
        print("WARNING: No model family files provided")
        return

    family_data = {}
    all_batch_sizes = set()

    for family_label, power_mode_files in model_family_files.items():
        if not power_mode_files:
            continue

        mode_data = {}

        for mode_label, json_path in power_mode_files.items():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"WARNING: File not found for {family_label} / {mode_label}: {json_path}")
                continue
            except json.JSONDecodeError:
                print(f"WARNING: Invalid JSON for {family_label} / {mode_label}: {json_path}")
                continue

            if not data:
                print(f"WARNING: No data in {json_path} ({family_label} / {mode_label})")
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
                print(f"WARNING: No valid '{value_key}' values for {family_label} / {mode_label}")
                continue

            mode_data[mode_label] = values_by_batch

        if mode_data:
            family_data[family_label] = mode_data

    if not family_data or not all_batch_sizes:
        print("WARNING: No valid data available to create combined throughput comparison plot")
        return

    batch_sizes = sorted(all_batch_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", ":", "-."]

    family_labels = list(family_data.keys())
    for family_idx, family_label in enumerate(family_labels):
        mode_items = list(family_data[family_label].items())
        for mode_idx, (mode_label, values_by_batch) in enumerate(mode_items):
            x_vals = []
            y_vals = []
            for bs in batch_sizes:
                if bs in values_by_batch:
                    x_vals.append(bs)
                    y_vals.append(values_by_batch[bs])

            if not x_vals:
                continue

            color = f"C{family_idx % 10}"
            linestyle = linestyles[mode_idx % len(linestyles)]
            ax.plot(
                x_vals,
                y_vals,
                marker=markers[(family_idx + mode_idx) % len(markers)],
                linewidth=2,
                linestyle=linestyle,
                color=color,
                label=f"{family_label} / {mode_label}",
            )

    ylabel = "Throughput (images/s)" if value_key == "throughput_images_per_s" else "Throughput (batches/s)"
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title("Throughput Comparison per Power Mode and Model Family")
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Model / Power Mode")

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
        "--combined-vision-models",
        action="store_true",
        help="Generate the combined base2/base4 vision comparison plot and ignore the other input arguments.",
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

    if args.combined_vision_models:
        main_combined_vision_models()
        return

    if args.mode:
        power_mode_files = parse_mode_args(args.mode)
    else:
        power_mode_files = discover_modes_from_input_dir(args.input_dir)

    throughput_comparison_power_modes_plot(
        power_mode_files=power_mode_files,
        output_path=args.output,
        value_key=args.value_key,
    )


def main_combined_vision_models():
    """Generate one plot that combines the base2 and base4 vision runs."""
    model_family_files = {
        "vision_base2_int8": {
            "15W": "/home/hanna/git/finn-transformers/summary_plots/data/vision_base2_int8/throughput_15w.json",
            "30W": "/home/hanna/git/finn-transformers/summary_plots/data/vision_base2_int8/throughput_30w.json",
            "50W": "/home/hanna/git/finn-transformers/summary_plots/data/vision_base2_int8/throughput_50w.json",
        },
        "vision_base4_int8": {
            "15W": "/home/hanna/git/finn-transformers/summary_plots/data/vision_base4_int8/throughput_15w.json",
            "30W": "/home/hanna/git/finn-transformers/summary_plots/data/vision_base4_int8/throughput_30w.json",
            "50W": "/home/hanna/git/finn-transformers/summary_plots/data/vision_base4_int8/throughput_50w.json",
        },
    }

    throughput_comparison_model_families_plot(
        model_family_files=model_family_files,
        output_path="/home/hanna/git/finn-transformers/summary_plots/outputs/vision_base2_int8/throughput_power_modes_base2_base4_comparison.png",
    )


if __name__ == "__main__":
    main()
