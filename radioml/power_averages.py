def power_averages(batch_sizes):
    # input logs besteht aus tegrastats_log, batch_size tuples
    # in jedem eintrag der output dateien soll als zusätzlicher key der batch_size wert stehen
    import re
    import json
    from datetime import datetime
    from pathlib import Path

    base_path = Path(__file__).resolve().parent.parent / "outputs" / "radioml" /"energy_metrics"
    energy_consumption_file = base_path / "energy_consumption.json" 
    power_averages_file = base_path / "power_averages.json"

    power_averages = []

    with open(energy_consumption_file, "r") as f:
            energy_consumption = json.load(f)

    for batch_size in batch_sizes:



        # begin with entry after start time
        # end with entry before end time

        # read start and endtime from files:
        start_end_time_file = base_path / f"timestamps_{batch_size}.json"
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




if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    power_averages(batch_sizes)

