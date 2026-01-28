import json
import numpy as np
import matplotlib.pyplot as plt#
from collections import defaultdict
from datetime import datetime
# Legende mit Balkenart und Voltage Type
from matplotlib.patches import Patch
from pathlib import Path
from dvclive import Live
import os
import yaml

MODEL_TYPE = os.environ.get("MODEL_TYPE", "vision")
if MODEL_TYPE != "radioml" and MODEL_TYPE != "language" and MODEL_TYPE != "vision":
    MODEL_TYPE = "vision"
    print("Defaulting Model Type to vision model.")

# look up quantisation type from params
with open(f"{MODEL_TYPE}/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

bits = cfg["model"]["embedding"].get("bits", 0)
INT8 = (bits == 8)

def throughput_batch_plot(json_path, output_path):
    with open(json_path, "r") as f:  # Name deiner Datei anpassen
        data = json.load(f)

    # Batch-Größen und Werte extrahieren
    batch_sizes = [d["batch_size"] for d in data]
    throughput_batches = [d["throughput_batches_per_s"] for d in data]

    # x-Positionen für die Balken (gleichmäßig)
    x = np.arange(len(batch_sizes))
    bar_width = 0.6  # Lücken zwischen den Balken

    # Plot erstellen
    fig, ax = plt.subplots()

    ax.bar(x, throughput_batches, width=bar_width, color='skyblue')

    # x-Achse beschriften
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput pro Batch Size")

    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def throughput_images_plot(json_path, output_path):
    with open(json_path, "r") as f:  # Name deiner Datei anpassen
        data = json.load(f)

    # Batch-Größen und Werte extrahieren
    batch_sizes = [d["batch_size"] for d in data]
    throughput_images = [d["throughput_images_per_s"] for d in data]

    # x-Positionen für die Balken (gleichmäßig)
    x = np.arange(len(batch_sizes))
    bar_width = 0.6  # Lücken zwischen den Balken

    # Plot erstellen
    fig, ax = plt.subplots()

    ax.bar(x, throughput_images, width=bar_width, color='skyblue')

    # x-Achse beschriften
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_title("Throughput pro Batch Size")

    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def latency_plot(json_path, output_path):
    with open(json_path, "r") as f:
        latency_results = json.load(f)

    # 2. Alle Batch-Größen sammeln (ohne Duplikate)
    batch_sizes = sorted(list(set(d["batch_size"] for d in latency_results)))

    # 3. Separate Listen für jeden Typ
    inteference = []
    synchronize = []
    datatransfer = []

    for bs in batch_sizes:
        # filtern nach Batch-Größe
        entries = [d for d in latency_results if d["batch_size"] == bs]
        
        # Werte den richtigen Listen zuordnen
        inteference.append(next(d["value"] for d in entries if d["type"] == "inteference"))
        synchronize.append(next(d["value"] for d in entries if d["type"] == "synchronize"))
        datatransfer.append(next(d["value"] for d in entries if d["type"] == "datatransfer"))

    fig, ax = plt.subplots()

    bar_width = 0.4
    x = np.arange(len(batch_sizes))
    ax.bar(x, inteference, width=bar_width, edgecolor="white", linewidth=0.7, label='inteference')
    ax.bar(x, synchronize, width=bar_width, bottom=inteference, edgecolor="white", linewidth=0.7, label='synchronize')
    ax.bar(x, datatransfer, width=bar_width, bottom=np.array(inteference)+np.array(synchronize), edgecolor="white", linewidth=0.7, label='datatransfer')


    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)

    ax.set_ylabel("Latenz (ms)")
    ax.set_xlabel("Batch Size")

    ax.legend()  # Legende anzeigen
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def throughput_per_power_plot(json_path, output_path):
    with open(json_path, "r") as f:  # Name deiner Datei anpassen
        data = json.load(f)

    # Batch-Größen und Werte extrahieren
    batch_sizes = [d["batch_size"] for d in data]
    throughput_per_power = [d["throughput_per_power"] for d in data]

    # x-Positionen für die Balken (gleichmäßig)
    x = np.arange(len(batch_sizes))
    bar_width = 0.6  # Lücken zwischen den Balken

    # Plot erstellen
    fig, ax = plt.subplots()

    ax.bar(x, throughput_per_power, width=bar_width, color='skyblue')

    # x-Achse beschriften
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput per power (sample/Joule)")
    ax.set_title("Throughput per power")

    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
def latency_per_throughput_plot(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    batch_sizes = [d["batch_size"] for d in data]
    latency_values = [d["latency_total"] for d in data]
    throughput_values = [d["throughput_images_per_s"] for d in data]

    # x: Latency in sec
    # y: Throughput in samples/s
    # neben den punkten die batch size als text anzeigen
    fig, ax = plt.subplots(figsize=(8,6))

    # Linie mit Punkten
    ax.plot(latency_values, throughput_values, marker='o', linestyle='-', color='red')

    # Batch Size neben jedem Punkt anzeigen
    for x, y, bs in zip(latency_values, throughput_values, batch_sizes):
        ax.text(x, y, str(bs), fontsize=9, ha='right', va='bottom')

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_title("Throughput vs. Latency per Batch Size")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot speichern
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    
def accuracies_plot(json_path, output_path):
    with open(json_path, "r") as f:  # Name deiner Datei anpassen
        data = json.load(f)

    # Batch-Größen und Werte extrahieren
    quantisation_type = [d["quantisation_type"] for d in data]
    accuracy = [d["value"] for d in data]

    # x-Positionen für die Balken (gleichmäßig)
    x = np.arange(len(quantisation_type))
    bar_width = 0.6  # Lücken zwischen den Balken

    # Plot erstellen
    fig, ax = plt.subplots()

    ax.bar(x, accuracy, width=bar_width, color='skyblue')

    # x-Achse beschriften
    ax.set_xticks(x)
    ax.set_xticklabels(quantisation_type)
    ax.set_xlabel("Quantisation Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Quantisation Type")

    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
def energy_consumption_plot(json_path, output_path):
        # JSON laden
    with open(json_path, "r") as f:
        data = json.load(f)

    data = [d for d in data if d.get("batch_size") == 256]
    if not data:
        print("Keine Daten für batch_size = 256 gefunden.")
        return
    # Zeitstempel sammeln (einzigartige, sortierte)
    timestamps = sorted(list(set(d["timestamp"] for d in data)))
    timestamps_dt = [datetime.fromisoformat(ts) for ts in timestamps]  # für schöne X-Achse

    # Alle Stromtypen sammeln
    current_types = sorted(list(set(d["type"] for d in data)))

    # Werte für jedes type pro timestamp vorbereiten
    values_per_type = {t: [] for t in current_types}
    bar_start_flags = []

    for ts in timestamps:
        entries = [d for d in data if d["timestamp"] == ts]
        total_per_type = defaultdict(float)
        bar_start = False
        for e in entries:
            total_per_type[e["type"]] += e["value"]
            if e.get("bar_start_end", False):
                bar_start = True
        for t in current_types:
            values_per_type[t].append(total_per_type[t])
        bar_start_flags.append(bar_start)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(timestamps))
    bottom = np.zeros(len(timestamps))

    colors = plt.cm.tab10.colors  # Farbschema für bis zu 10 Typen

    for i, t in enumerate(current_types):
        ax.bar(x, values_per_type[t], bottom=bottom, color=colors[i % 10], label=t)
        bottom += np.array(values_per_type[t])

    # horizontale Linie für bar_start_end
    for xi, flag in enumerate(bar_start_flags):
        if flag:
            ax.axvline(x=xi, color='black', linestyle='--', linewidth=1)

    # Achsen und Labels
    ax.set_xticks(x)
    ax.set_xticklabels([dt.strftime("%H:%M:%S") for dt in timestamps_dt], rotation=45, ha='right')
    ax.set_ylabel("Current (mA)")
    ax.set_xlabel("Timestamp")
    ax.set_title("Energy Consumption per Timestamp")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def power_bar_plot(json_path, output_path):
    # JSON laden
    with open(json_path, "r") as f:
        data = json.load(f)

    # Alle Batch-Größen sammeln
    batch_sizes = sorted(list(set(d["batch_size"] for d in data)))

    # Alle Typen sammeln
    types = sorted(list(set(d["type"] for d in data)))

    # Werte vorbereiten: dict[batch_size][type] = value
    idle_values = {bs: [] for bs in batch_sizes}
    inference_values = {bs: [] for bs in batch_sizes}

    for bs in batch_sizes:
        for t in types:
            idle = sum(d["idle_value"] for d in data if d["batch_size"]==bs and d["type"]==t)
            inf = sum(d["value"] for d in data if d["batch_size"]==bs and d["type"]==t)
            idle_values[bs].append(idle)
            inference_values[bs].append(inf)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(batch_sizes))
    bar_width = 0.35

    colors = plt.cm.tab10.colors

    # Idle Bars (links)
    bottom_idle = np.zeros(len(batch_sizes))
    for i, t in enumerate(types):
        values = [idle_values[bs][i] for bs in batch_sizes]
        ax.bar(x - bar_width/2, values, bar_width, bottom=bottom_idle, color=colors[i%10], label=t if i==0 else "")
        bottom_idle += np.array(values)

    # Inference Bars (rechts)
    bottom_inf = np.zeros(len(batch_sizes))
    for i, t in enumerate(types):
        values = [inference_values[bs][i] for bs in batch_sizes]
        ax.bar(x + bar_width/2, values, bar_width, bottom=bottom_inf, color=colors[i%10])
        bottom_inf += np.array(values)

    # Achsen beschriften
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Power (mA)")
    ax.set_title("Idle vs Inference Power per Batch Size")

    
    legend_handles = [Patch(facecolor='white', edgecolor='black', label='Left = Idle, Right = Inference')]
    legend_handles += [Patch(facecolor=colors[i%10], label=t) for i, t in enumerate(types)]
    ax.legend(handles=legend_handles, title="Legend")

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

# echte pfade ergänzen (erstmal FP32)
# outputs/*MODEL*/plot/*QUANT_TYPE*/*NAME*  --> passende dateien mit passenden namen müssen noch erstellt werden


if __name__ == "__main__":
    if INT8 == True:
        quant_type = "INT8"
    else:
        quant_type = "FP32"

    base_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / quant_type 

    latency_throughput_path = base_path / "latency_throughput.json"
    latency_throughput_output = base_path / "latency_per_throughput_plot.png"

    latency_throughput_output.parent.mkdir(parents=True, exist_ok=True)
    latency_per_throughput_plot(latency_throughput_path, latency_throughput_output)


    throughput_results_path = base_path / "throughput_results.json"
    # throughput_results_output_batch = "images/throughput_batch_plot.png"
    throughput_results_output_batch =  base_path / "throughput_batch_plot.png"
    throughput_batch_plot(throughput_results_path, throughput_results_output_batch)

    throughput_results_output_images =  base_path / "throughput_images_plot.png"
    throughput_images_plot(throughput_results_path, throughput_results_output_images)

    latency_results_path = base_path / "latency_results_batch.json"
    # latency_results_output = "images/latency_plot.png"
    latency_results_output = base_path / "latency_plot.png"
    latency_plot(latency_results_path, latency_results_output)

    power_throughput_path = base_path / "power_throughput.json"
    power_throughput_output = base_path / "throughput_per_power_plot.png"
    throughput_per_power_plot(power_throughput_path, power_throughput_output)

    energy_consumption_path = base_path / "energy_consumption.json" 
    energy_consumption_output = base_path / "energy_consumption_plot.png"
    energy_consumption_plot(energy_consumption_path, energy_consumption_output)

    power_bar_path = base_path / "power_averages_baseline_inference.json"
    power_bar_output = base_path / "power_bar_plot.png"
    power_bar_plot(power_bar_path, power_bar_output)

    # eigentlich nicht bei jeder Quantisierung plotten, sondern am ende
    # accuracy_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE /"plot" / "accuracy.json"
    # accuracy_output = base_path / "accuracies_plot.png"
    # accuracies_plot(accuracy_path, accuracy_output)

    # mit dvc exp hochladen
    # log image
    with Live(save_dvc_exp=True, cache_images = false, report="md") as live:
        print("Starte DVC Live Bericht....", flush=True)

        live.log_image("latency_throughput_plot.png", latency_throughput_output)
        live.log_image("throughput_batch_plot.png", throughput_results_output_batch)
        live.log_image("throughput_images_plot.png", throughput_results_output_images)
        live.log_image("latency_plot.png", latency_results_output)
        live.log_image("throughput_per_power_plot.png", power_throughput_output)
        # live.log_image("accuracies_plot.png", accuracy_output)
        live.log_image("energy_consumption_plot.png", energy_consumption_output)
        live.log_image("power_bar_plot.png", power_bar_output)


        live.next_step()

    print("DVC Live Bericht fertig!")