def power_averages(batch_sizes, power_averages_file, energy_consumption_file, quant_type):
    # input logs besteht aus tegrastats_log, batch_size tuples
    # in jedem eintrag der output dateien soll als zusätzlicher key der batch_size wert stehen
    import re
    import json
    from datetime import datetime
    from pathlib import Path
    

    power_averages = []

    with open(energy_consumption_file, "r") as f:
            energy_consumption = json.load(f)

    for batch_size in batch_sizes:
        start_end_time_file = Path(__file__).resolve().parent.parent / "outputs" / "radioml" /"energy_metrics" / quant_type / f"timestamps_{batch_size}.json"
        with open(start_end_time_file, "r") as f:
            timestamps = json.load(f)
        start_iso = timestamps["start_time"]
        end_iso = timestamps["end_time"]
        start = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%S.%f")
        end = datetime.strptime(end_iso, "%Y-%m-%dT%H:%M:%S.%f")

        vdd_gpu = 0
        vdd_cpu = 0
        vin_sys = 0
        count_vdd_gpu = 0
        count_vdd_cpu = 0
        count_vin_sys = 0

        for entry in energy_consumption:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S")
            if entry_time >= start and entry_time <= end and batch_size == entry["batch_size"] and entry["type"]=="vdd_gpu_soc_current":
                vdd_gpu += entry["value"]
                count_vdd_gpu += 1

            if entry_time >= start and entry_time <= end and batch_size == entry["batch_size"] and entry["type"]=="vdd_cpu_cv_current":
                vdd_cpu += entry["value"]
                count_vdd_cpu += 1

            if entry_time >= start and entry_time <= end and batch_size == entry["batch_size"] and entry["type"]=="vin_sys_5v0_current":
                vin_sys += entry["value"]
                count_vin_sys += 1

        if count_vdd_gpu==0 or count_vdd_cpu==0 or count_vin_sys==0:
            print(f"⚠️ Warnung: Keine Daten für batch_size {batch_size} gefunden.")

        vdd_gpu_avg = vdd_gpu/count_vdd_gpu if count_vdd_gpu > 0 else 0
        vdd_cpu_avg = vdd_cpu/count_vdd_cpu if count_vdd_cpu > 0 else 0
        vin_sys_avg = vin_sys/count_vin_sys if count_vin_sys > 0 else 0

        power_averages.append({
            "batch_size": batch_size,
            "type": "vdd_gpu_avg",
            "value": vdd_gpu_avg,
        })
        power_averages.append({
            "batch_size": batch_size,
            "type": "vdd_cpu_avg",
            "value": vdd_cpu_avg,
        })
        power_averages.append({
            "batch_size": batch_size,
            "type": "vin_sys_avg",
            "value": vin_sys_avg,
        })

    with open(power_averages_file, "w") as f:
        json.dump(power_averages, f, indent=2)

    print(f"{len(power_averages)} Einträge in '{power_averages_file.name}' gespeichert (Durchschnittswerte).")



def power_averages_baseline(batch_sizes, power_averages_file, energy_consumption_file, quant_type):
    # input logs besteht aus tegrastats_log, batch_size tuples
    # in jedem eintrag der output dateien soll als zusätzlicher key der batch_size wert stehen
    import re
    import json
    from datetime import datetime
    from pathlib import Path

    power_averages = []

    with open(energy_consumption_file, "r") as f:
            energy_consumption = json.load(f)

    for batch_size in batch_sizes:
        start_end_time_file = Path(__file__).resolve().parent.parent / "outputs" / "radioml" /"energy_metrics" / quant_type / f"timestamps_{batch_size}.json"
        with open(start_end_time_file, "r") as f:
            timestamps = json.load(f)
        start_iso = timestamps["start_time"]
        end_iso = timestamps["end_time"]
        start = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%S.%f")
        end = datetime.strptime(end_iso, "%Y-%m-%dT%H:%M:%S.%f")

        vdd_gpu = 0
        vdd_cpu = 0
        vin_sys = 0
        count_vdd_gpu = 0
        count_vdd_cpu = 0
        count_vin_sys = 0

        for entry in energy_consumption:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S")
            if (entry_time < start or entry_time > end) and batch_size == entry["batch_size"] and entry["type"]=="vdd_gpu_soc_current":
                vdd_gpu += entry["value"]
                count_vdd_gpu += 1

            if (entry_time < start or entry_time > end) and batch_size == entry["batch_size"] and entry["type"]=="vdd_cpu_cv_current":
                vdd_cpu += entry["value"]
                count_vdd_cpu += 1

            if (entry_time < start or entry_time > end) and batch_size == entry["batch_size"] and entry["type"]=="vin_sys_5v0_current":
                vin_sys += entry["value"]
                count_vin_sys += 1

        
        vdd_gpu_avg = vdd_gpu/count_vdd_gpu if count_vdd_gpu > 0 else 0
        vdd_cpu_avg = vdd_cpu/count_vdd_cpu if count_vdd_cpu > 0 else 0
        vin_sys_avg = vin_sys/count_vin_sys if count_vin_sys > 0 else 0

        power_averages.append({
            "batch_size": batch_size,
            "type": "vdd_gpu_avg",
            "value": vdd_gpu_avg,
        })
        power_averages.append({
            "batch_size": batch_size,
            "type": "vdd_cpu_avg",
            "value": vdd_cpu_avg,
        })
        power_averages.append({
            "batch_size": batch_size,
            "type": "vin_sys_avg",
            "value": vin_sys_avg,
        })

    with open(power_averages_file, "w") as f:
        json.dump(power_averages, f, indent=2)

    print(f"{len(power_averages)} Einträge in '{power_averages_file.name}' gespeichert (Durchschnittswerte).")

def power_averages_difference(batch_sizes, power_averages_file, power_averages_baseline_file, power_difference_file, quant_type):
    # input logs besteht aus tegrastats_log, batch_size tuples
    import re
    import json
    from datetime import datetime
    from pathlib import Path
    # in jedem eintrag der output dateien soll als zusätzlicher key der batch_size wert stehen
        # JSON-Dateien einlesen
    with open(power_averages_file, 'r') as f:
        inference_data = json.load(f)

    with open(power_averages_baseline_file, 'r') as f:
        baseline_data = json.load(f)

    # Baseline-Werte in ein Dictionary für schnellen Zugriff umwandeln
    baseline_dict = {
        (entry['batch_size'], entry['type']): entry['value']
        for entry in baseline_data
    }

    # Differenzen berechnen
    difference_data = []
    for entry in inference_data:
        batch_size = entry['batch_size']
        type_ = entry['type']
        inference_value = entry['value']

        # Nur berechnen, wenn batch_size in der gewünschten Liste ist
        if batch_size in batch_sizes:
            start_end_time_file = Path(__file__).resolve().parent.parent / "outputs" / "radioml" /"energy_metrics" / quant_type / f"timestamps_{batch_size}.json"
            baseline_value = baseline_dict.get((batch_size, type_))

            if baseline_value is not None:
                difference = inference_value - baseline_value
                difference_data.append({
                    "batch_size": batch_size,
                    "type": type_,
                    "value": difference
                })
            else:
                print(f"⚠️ Kein Baseline-Wert für batch_size {batch_size} und type {type_} gefunden.")

    # In die Ausgabedatei schreiben
    with open(power_difference_file, 'w') as f:
        json.dump(difference_data, f, indent=2)




if __name__ == "__main__":
    import re
    import json
    from datetime import datetime
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent / "outputs" / "radioml" /"energy_metrics"
    energy_consumption_file = base_path / "energy_consumption.json" 
    power_averages_file = base_path / "power_averages.json"
    power_averages_baseline_file = base_path / "power_averages_baseline.json"
    power_difference_file = base_path / "power_averages_difference.json"


    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    difference_baseline_inference(batch_sizes, power_averages_file, power_averages_baseline_file, power_difference_file)


    # power_averages(batch_sizes, power_averages_file, energy_consumption_file)

