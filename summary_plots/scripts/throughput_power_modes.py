import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_for_plots"
DEFAULT_INPUT_DIR = DATA_ROOT / "base4" / "int8"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_COMPARISON_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "model_comparison"
POWER_MODES = ("15w", "30w", "50w")
MODELS = ("base", "base2", "base4")
QUANT_COMPARISON_MODES = ("int8", "ort_int8")


class RawDefaultsHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass


CLI_REFERENCE = """Plot modes
    Default throughput plot
        Reads throughput_*w.json files from --input-dir or explicit --mode mappings.
        Writes to summary_plots/outputs/<input-folder>/throughput_power_modes_comparison.png when --output is omitted.

    --latency-throughput
        Reads latency_throughput_*w.json files from --input-dir or explicit --mode mappings.
        Writes to summary_plots/outputs/<input-folder>/latency_throughput_power_modes_comparison_logxy.png when --output is omitted.

    --latency-throughput-summary
        Reads latency_throughput_*w.json files of every model that has data for the mode of --input-dir.
        Writes to summary_plots/outputs/model_comparison/latency_throughput_model_comparison_<mode>_logxy.png.

    --combined-vision-models
        Generates the throughput comparison across all models for the mode of --input-dir.
        Writes to summary_plots/outputs/model_comparison/throughput_model_comparison_<mode>.png.

    --latency-one-model
        Reads latency_*w.json files from --input-dir or explicit --mode mappings.
        Writes to summary_plots/outputs/<input-folder>/latency_power_modes_comparison.png when --output is omitted.

    --latency-summary
        Generates the latency comparison across all models for the mode of --input-dir.
        Writes to summary_plots/outputs/model_comparison/latency_model_comparison_<mode>.png.

    --quant-comparison
        Generates all three plots comparing int8 against ort_int8 for the model of --input-dir.
        Writes to summary_plots/outputs/vision_<model>_int8_ort_int8/*_quant_modes_comparison*.png.

Common options
    --mode LABEL=/path/to/file.json
        Repeatable explicit mode mapping. Skips auto-discovery.

    --input-dir DIR
        Directory used for auto-discovery of throughput_*w.json or latency_throughput_*w.json files.
        Expected layout is summary_plots/data_for_plots/<model>/<mode>, for example .../base4/fp16.

    --output PATH
        Output image path for the selected single-model plot. If omitted, the output folder is derived from --input-dir.

    --value-key {throughput_images_per_s,throughput_batches_per_s}
        Throughput metric used by the default throughput plot.

    --throughput-log-scale
        Use a logarithmic y-axis for throughput plots.

    --latency-log-scale
        Use a logarithmic x-axis for latency plots.
"""


def _output_folder_name(source_dir):
        """Flatten a <model>/<mode> data directory into a single output folder name."""
        source_dir = Path(source_dir).resolve()
        try:
                relative = source_dir.relative_to(DATA_ROOT)
        except ValueError:
                return source_dir.name
        return "vision_" + "_".join(relative.parts)


def _family_dir(input_dir, model):
        """Return the sibling data directory of another model for the same mode."""
        input_dir = Path(input_dir)
        return input_dir.parent.parent / model / input_dir.name


def _power_mode_files(model, mode, prefix):
        """Collect the power mode files that exist for one model, e.g. base has no 30W/50W int8 run."""
        files = {}
        for power in POWER_MODES:
                path = DATA_ROOT / model / mode / f"{prefix}_{power}.json"
                if path.is_file():
                        files[power.upper()] = str(path)
        return files


def _model_family_files(mode, prefix):
        """Map every model with data for this mode to its power mode files."""
        family_files = {}
        for model in MODELS:
                power_mode_files = _power_mode_files(model, mode, prefix)
                if power_mode_files:
                        family_files[_output_folder_name(DATA_ROOT / model / mode)] = power_mode_files
        return family_files


def _quant_family_files(model, prefix):
        """Map every quantization mode with data for this model to its power mode files."""
        family_files = {}
        for mode in QUANT_COMPARISON_MODES:
                power_mode_files = _power_mode_files(model, mode, prefix)
                if power_mode_files:
                        family_files[mode] = power_mode_files
        return family_files


def _default_output_path(source_dir, filename):
        return DEFAULT_OUTPUT_ROOT / _output_folder_name(source_dir) / filename


def _resolve_default_output_path(output_path, source_dir, filename):
        if output_path:
                return Path(output_path)
        return _default_output_path(source_dir, filename)


def _infer_source_dir_from_mode_files(mode_files, fallback_dir):
        if mode_files:
                first_path = Path(next(iter(mode_files.values())))
                return first_path.parent
        return Path(fallback_dir)


def throughput_comparison_power_modes_plot(
    power_mode_files,
    output_path,
    value_key="throughput_images_per_s",
    yscale="linear",
):
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
    title = "Throughput Comparison per Power Mode"
    if yscale == "log":
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Power Mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def throughput_comparison_model_families_plot(
    model_family_files,
    output_path,
    value_key="throughput_images_per_s",
    yscale="linear",
    family_kind="Model",
):
    """
    Erstellt einen kombinierten Throughput-Vergleich mehrerer Modellfamilien
    und Power-Modi in einem Diagramm.

    Args:
        model_family_files: Dict mit Modellfamilien-Label -> Dict von Power-Mode-Label
            zu JSON-Pfad, z.B.
            {
                "base2": {
                    "15W": ".../base2/int8/throughput_15w.json",
                    "30W": ".../base2/int8/throughput_30w.json",
                    "50W": ".../base2/int8/throughput_50w.json",
                },
                "base4": {
                    "15W": ".../base4/int8/throughput_15w.json",
                    "30W": ".../base4/int8/throughput_30w.json",
                    "50W": ".../base4/int8/throughput_50w.json",
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
    title = f"Throughput Comparison per Power Mode and {family_kind}"
    if yscale == "log":
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=f"{family_kind} / Power Mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def latency_throughput_comparison_power_modes_plot(power_mode_files, output_path, xscale="log", yscale="log"):
    """Create a latency-vs-throughput plot for multiple power modes."""
    if not power_mode_files:
        print("WARNING: No power mode files provided")
        return

    mode_data = {}

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

        points_by_batch = {}
        for entry in data:
            batch_size = entry.get("batch_size")
            latency_total = entry.get("latency_total")
            throughput_images_per_s = entry.get("throughput_images_per_s")
            if batch_size is None or latency_total is None or throughput_images_per_s is None:
                continue
            points_by_batch[batch_size] = (latency_total, throughput_images_per_s)

        if not points_by_batch:
            print(f"WARNING: No valid latency-throughput data for mode {mode_label}")
            continue

        mode_data[mode_label] = points_by_batch

    if not mode_data:
        print("WARNING: No valid data available to create latency-throughput comparison plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, (mode_label, points_by_batch) in enumerate(mode_data.items()):
        sorted_points = sorted(points_by_batch.items())
        latency_values = [point[0] for _, point in sorted_points]
        throughput_values = [point[1] for _, point in sorted_points]
        batch_sizes = [batch_size for batch_size, _ in sorted_points]

        ax.plot(
            latency_values,
            throughput_values,
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=mode_label,
        )

        for latency_value, throughput_value, batch_size in zip(latency_values, throughput_values, batch_sizes):
            ax.text(latency_value, throughput_value, str(batch_size), fontsize=9, ha="right", va="bottom")

    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Throughput (images/s)")
    title = "Throughput vs. Latency per Batch Size and Power Mode"
    if xscale == "log" or yscale == "log":
        title += " (log scale)"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Power Mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def latency_throughput_comparison_model_families_plot(
    model_family_files,
    output_path,
    xscale="log",
    yscale="log",
    family_kind="Model",
):
    """Create a latency-vs-throughput plot for multiple model families and power modes."""
    if not model_family_files:
        print("WARNING: No model family files provided")
        return

    family_data = {}

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

            points_by_batch = {}
            for entry in data:
                batch_size = entry.get("batch_size")
                latency_total = entry.get("latency_total")
                throughput_images_per_s = entry.get("throughput_images_per_s")
                if batch_size is None or latency_total is None or throughput_images_per_s is None:
                    continue
                points_by_batch[batch_size] = (latency_total, throughput_images_per_s)

            if not points_by_batch:
                print(f"WARNING: No valid latency-throughput data for {family_label} / {mode_label}")
                continue

            mode_data[mode_label] = points_by_batch

        if mode_data:
            family_data[family_label] = mode_data

    if not family_data:
        print("WARNING: No valid data available to create combined latency-throughput comparison plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", ":", "-."]

    family_labels = list(family_data.keys())
    for family_idx, family_label in enumerate(family_labels):
        mode_items = list(family_data[family_label].items())
        for mode_idx, (mode_label, points_by_batch) in enumerate(mode_items):
            sorted_points = sorted(points_by_batch.items())
            latency_values = [point[0] for _, point in sorted_points]
            throughput_values = [point[1] for _, point in sorted_points]
            batch_sizes = [batch_size for batch_size, _ in sorted_points]

            if not latency_values:
                continue

            color = f"C{family_idx % 10}"
            linestyle = linestyles[mode_idx % len(linestyles)]
            ax.plot(
                latency_values,
                throughput_values,
                marker=markers[(family_idx + mode_idx) % len(markers)],
                linewidth=2,
                linestyle=linestyle,
                color=color,
                label=f"{family_label} / {mode_label}",
            )

            for latency_value, throughput_value, batch_size in zip(latency_values, throughput_values, batch_sizes):
                ax.text(latency_value, throughput_value, str(batch_size), fontsize=9, ha="right", va="bottom")

    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Throughput (images/s)")
    title = f"Throughput vs. Latency per Batch Size, Power Mode, and {family_kind}"
    if xscale == "log" or yscale == "log":
        title += " (log scale)"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=f"{family_kind} / Power Mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def _load_latency_totals_by_batch(json_path):
    """Load latency entries and return per-batch totals plus per-type breakdown."""
    with open(json_path, "r") as f:
        data = json.load(f)

    totals_by_batch = {}
    breakdown_by_type = {}

    for entry in data:
        batch_size = entry.get("batch_size")
        latency_type = entry.get("type", "unknown")
        value = entry.get("value")

        if batch_size is None or value is None:
            continue

        totals_by_batch[batch_size] = totals_by_batch.get(batch_size, 0.0) + value
        if latency_type not in breakdown_by_type:
            breakdown_by_type[latency_type] = {}
        breakdown_by_type[latency_type][batch_size] = value

    return totals_by_batch, breakdown_by_type


def latency_comparison_power_modes_plot(power_mode_files, output_path, xscale="linear"):
    """
    Erstellt einen Latency-Vergleich eines Modells ueber mehrere Power-Modi.

    Args:
        power_mode_files: Dict mit Label -> JSON-Pfad, z.B.
            {
                "15W": ".../latency_15w.json",
                "30W": ".../latency_30w.json",
                "50W": ".../latency_50w.json",
            }
        output_path: Ausgabe-Pfad fuer den Plot.
    """
    if not power_mode_files:
        print("WARNING: No power mode files provided")
        return

    mode_data = {}
    all_batch_sizes = set()

    for mode_label, json_path in power_mode_files.items():
        try:
            totals_by_batch, _ = _load_latency_totals_by_batch(json_path)
        except FileNotFoundError:
            print(f"WARNING: File not found for mode {mode_label}: {json_path}")
            continue
        except json.JSONDecodeError:
            print(f"WARNING: Invalid JSON for mode {mode_label}: {json_path}")
            continue

        if not totals_by_batch:
            print(f"WARNING: No valid latency data for mode {mode_label}")
            continue

        mode_data[mode_label] = totals_by_batch
        all_batch_sizes.update(totals_by_batch.keys())

    if not mode_data or not all_batch_sizes:
        print("WARNING: No valid data available to create latency comparison plot")
        return

    batch_sizes = sorted(all_batch_sizes)

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, (mode_label, totals_by_batch) in enumerate(mode_data.items()):
        x_vals = []
        y_vals = []
        for batch_size in batch_sizes:
            if batch_size in totals_by_batch:
                x_vals.append(batch_size)
                y_vals.append(totals_by_batch[batch_size])

        if not x_vals:
            continue

        ax.plot(
            x_vals,
            y_vals,
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=mode_label,
        )

    if xscale == "log":
        ax.set_xscale("log")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    title = "Total Latency per Batch and Power Mode"
    if xscale == "log":
        title += " (log scale)"
    ax.set_title(title)
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Power Mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def latency_summary_model_families_plot(model_family_files, output_path, xscale="linear", family_kind="Model"):
    """
    Erstellt einen kombinierten Latency-Vergleich mehrerer Familien
    und Power-Modi in einem Diagramm.

    Args:
        model_family_files: Dict mit Familien-Label -> Dict von Power-Mode-Label
            zu JSON-Pfad. Eine Familie ist entweder ein Modell (base, base2, base4)
            oder ein Quantisierungs-Modus (int8, ort_int8), z.B.
            {
                "vision_base2_int8": {
                    "15W": ".../base2/int8/latency_15w.json",
                    "30W": ".../base2/int8/latency_30w.json",
                    "50W": ".../base2/int8/latency_50w.json",
                },
                "vision_base4_int8": {...},
            }
        output_path: Ausgabe-Pfad fuer den Plot.
        family_kind: Bezeichnung der Familien-Achse fuer Titel und Legende.
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
                totals_by_batch, _ = _load_latency_totals_by_batch(json_path)
            except FileNotFoundError:
                print(f"WARNING: File not found for {family_label} / {mode_label}: {json_path}")
                continue
            except json.JSONDecodeError:
                print(f"WARNING: Invalid JSON for {family_label} / {mode_label}: {json_path}")
                continue

            if not totals_by_batch:
                print(f"WARNING: No valid latency data in {json_path} ({family_label} / {mode_label})")
                continue

            mode_data[mode_label] = totals_by_batch
            all_batch_sizes.update(totals_by_batch.keys())

        if mode_data:
            family_data[family_label] = mode_data

    if not family_data or not all_batch_sizes:
        print("WARNING: No valid data available to create combined latency comparison plot")
        return

    batch_sizes = sorted(all_batch_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", ":", "-."]

    family_labels = list(family_data.keys())
    for family_idx, family_label in enumerate(family_labels):
        mode_items = list(family_data[family_label].items())
        for mode_idx, (mode_label, totals_by_batch) in enumerate(mode_items):
            x_vals = []
            y_vals = []
            for batch_size in batch_sizes:
                if batch_size in totals_by_batch:
                    x_vals.append(batch_size)
                    y_vals.append(totals_by_batch[batch_size])

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

    if xscale == "log":
        ax.set_xscale("log")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    title = f"Total Latency per Batch, Power Mode, and {family_kind}"
    if xscale == "log":
        title += " (log scale)"
    ax.set_title(title)
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=f"{family_kind} / Power Mode")

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


def discover_latency_from_input_dir(input_dir):
    input_dir = Path(input_dir)
    power_mode_files = {}

    for file_path in sorted(input_dir.glob("latency_*w.json")):
        stem = file_path.stem.lower()
        if stem.startswith("latency_throughput_"):
            continue
        label = stem.replace("latency_", "").upper()
        power_mode_files[label] = str(file_path)

    return power_mode_files


def discover_latency_throughput_from_input_dir(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("latency_throughput_*w.json"))
    power_mode_files = {}

    for file_path in files:
        stem = file_path.stem.lower()
        label = stem.replace("latency_throughput_", "").upper()
        power_mode_files[label] = str(file_path)

    return power_mode_files


def discover_latency_throughput_families_from_input_dir(input_dir):
    family_files = {}
    for model in MODELS:
        family_dir = _family_dir(input_dir, model)
        power_mode_files = discover_latency_throughput_from_input_dir(family_dir)
        if power_mode_files:
            family_files[_output_folder_name(family_dir)] = power_mode_files
    return family_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate throughput and latency comparison plots per power mode.",
        formatter_class=RawDefaultsHelpFormatter,
        epilog=CLI_REFERENCE,
    )
    parser.add_argument(
        "--combined-vision-models",
        action="store_true",
        help="Generate the combined base2/base4 vision comparison plot and ignore the other input arguments.",
    )
    parser.add_argument(
        "--latency-one-model",
        action="store_true",
        help="Generate the latency comparison plot for one model and ignore the throughput arguments.",
    )
    parser.add_argument(
        "--latency-summary",
        action="store_true",
        help="Generate the combined latency summary plot for the available model families.",
    )
    parser.add_argument(
        "--quant-comparison",
        action="store_true",
        help="Generate all three plots comparing int8 against ort_int8 for the model of --input-dir.",
    )
    parser.add_argument(
        "--latency-throughput",
        action="store_true",
        help="Generate the latency-throughput plot for the available power modes.",
    )
    parser.add_argument(
        "--latency-throughput-summary",
        action="store_true",
        help="Generate the combined latency-throughput plot for the available model families.",
    )
    parser.add_argument(
        "--throughput-log-scale",
        action="store_true",
        help="Use a logarithmic y-axis for throughput plots.",
    )
    parser.add_argument(
        "--latency-log-scale",
        action="store_true",
        help="Use a logarithmic x-axis for latency plots.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Power mode mapping in the form LABEL=/path/to/file.json. Can be used multiple times.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Data directory <model>/<mode> for auto-discovery of the *_*w.json files of the selected plot mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the generated plot image. If omitted, the folder is derived from --input-dir.",
    )
    parser.add_argument(
        "--value-key",
        default="throughput_images_per_s",
        choices=["throughput_images_per_s", "throughput_batches_per_s"],
        help="Which throughput metric to plot.",
    )

    args = parser.parse_args()

    if args.combined_vision_models:
        main_combined_vision_models(log_scale=args.throughput_log_scale, mode=Path(args.input_dir).name)
        return

    if args.latency_one_model:
        if args.mode:
            power_mode_files = parse_mode_args(args.mode)
            source_dir = _infer_source_dir_from_mode_files(power_mode_files, args.input_dir)
        else:
            power_mode_files = discover_latency_from_input_dir(args.input_dir)
            source_dir = Path(args.input_dir)

        latency_comparison_power_modes_plot(
            power_mode_files=power_mode_files,
            output_path=_resolve_default_output_path(
                args.output,
                source_dir,
                "latency_power_modes_comparison_logx.png"
                if args.latency_log_scale
                else "latency_power_modes_comparison.png",
            ),
            xscale="log" if args.latency_log_scale else "linear",
        )
        return

    if args.latency_summary:
        main_latency_summary(log_scale=args.latency_log_scale, mode=Path(args.input_dir).name)
        return

    if args.quant_comparison:
        main_quant_comparison(
            Path(args.input_dir).parent.name,
            throughput_log_scale=args.throughput_log_scale,
            latency_log_scale=args.latency_log_scale,
        )
        return

    if args.latency_throughput:
        if args.mode:
            power_mode_files = parse_mode_args(args.mode)
            source_dir = _infer_source_dir_from_mode_files(power_mode_files, args.input_dir)
        else:
            power_mode_files = discover_latency_throughput_from_input_dir(args.input_dir)
            source_dir = Path(args.input_dir)

        latency_throughput_comparison_power_modes_plot(
            power_mode_files=power_mode_files,
            output_path=_resolve_default_output_path(
                args.output,
                source_dir,
                "latency_throughput_power_modes_comparison_logxy.png",
            ),
            xscale="log",
            yscale="log",
        )
        return

    if args.latency_throughput_summary:
        model_family_files = discover_latency_throughput_families_from_input_dir(args.input_dir)
        latency_throughput_comparison_model_families_plot(
            model_family_files=model_family_files,
            output_path=MODEL_COMPARISON_OUTPUT_DIR
            / f"latency_throughput_model_comparison_{Path(args.input_dir).name}_logxy.png",
            xscale="log",
            yscale="log",
        )
        return

    if args.mode:
        power_mode_files = parse_mode_args(args.mode)
        source_dir = _infer_source_dir_from_mode_files(power_mode_files, args.input_dir)
    else:
        power_mode_files = discover_modes_from_input_dir(args.input_dir)
        source_dir = Path(args.input_dir)

    output_path = _resolve_default_output_path(
        args.output,
        source_dir,
        "throughput_power_modes_comparison.png",
    )

    throughput_comparison_power_modes_plot(
        power_mode_files=power_mode_files,
        output_path=output_path,
        value_key=args.value_key,
        yscale="log" if args.throughput_log_scale else "linear",
    )


def main_combined_vision_models(log_scale=False, mode="int8"):
    """Generate one plot that combines every vision model available for this mode."""
    throughput_comparison_model_families_plot(
        model_family_files=_model_family_files(mode, "throughput"),
        output_path=MODEL_COMPARISON_OUTPUT_DIR
        / (
            f"throughput_model_comparison_{mode}_logy.png"
            if log_scale
            else f"throughput_model_comparison_{mode}.png"
        ),
        yscale="log" if log_scale else "linear",
    )


def main_quant_comparison(model, throughput_log_scale=False, latency_log_scale=False):
    """Generate all three plots comparing the quantization modes of one model."""
    output_dir = DEFAULT_OUTPUT_ROOT / f"vision_{model}_{'_'.join(QUANT_COMPARISON_MODES)}"

    throughput_comparison_model_families_plot(
        model_family_files=_quant_family_files(model, "throughput"),
        output_path=output_dir
        / (
            "throughput_quant_modes_comparison_logy.png"
            if throughput_log_scale
            else "throughput_quant_modes_comparison.png"
        ),
        yscale="log" if throughput_log_scale else "linear",
        family_kind="Quantization",
    )

    latency_throughput_comparison_model_families_plot(
        model_family_files=_quant_family_files(model, "latency_throughput"),
        output_path=output_dir / "latency_throughput_quant_modes_comparison_logxy.png",
        family_kind="Quantization",
    )

    latency_summary_model_families_plot(
        model_family_files=_quant_family_files(model, "latency"),
        output_path=output_dir
        / ("latency_quant_modes_comparison_logx.png" if latency_log_scale else "latency_quant_modes_comparison.png"),
        xscale="log" if latency_log_scale else "linear",
        family_kind="Quantization",
    )


def main_latency_summary(log_scale=False, mode="int8"):
    """Generate one plot that compares every available model's latency across power modes."""
    latency_summary_model_families_plot(
        model_family_files=_model_family_files(mode, "latency"),
        output_path=MODEL_COMPARISON_OUTPUT_DIR
        / (
            f"latency_model_comparison_{mode}_logx.png"
            if log_scale
            else f"latency_model_comparison_{mode}.png"
        ),
        xscale="log" if log_scale else "linear",
    )


if __name__ == "__main__":
    main()
