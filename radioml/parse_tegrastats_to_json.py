def parse_tegrastats(input_logs):
    # input logs besteht aus tegrastats_log, batch_size tuples
    # in jedem eintrag der output dateien soll als zusätzlicher key der batch_size wert stehen
    import re
    import json
    from datetime import datetime
    from pathlib import Path

    base_path = Path(__file__).resolve().parent.parent / "outputs" / "radioml" /"energy_metrics"
    input_log = base_path / f"tegrastats.log"
    output_json_full = base_path /"energy_metrics.json"
    output_json_simple = base_path / "ram_metrics.json"
    output_json_simple2 = base_path / "ram_metrics_2.json"
    output_json_energy = base_path / "energy_consumption.json" 
    output_json_energy2 = base_path / "energy_consumption_2.json"

    def parse_tegrastats_line(line):
        try:
            data = {}

            # Zeitstempel
            ts_match = re.match(r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if not ts_match:
                return None
            timestamp = datetime.strptime(ts_match.group(1), "%m-%d-%Y %H:%M:%S").isoformat()
            data["timestamp"] = timestamp

            # RAM
            ram_match = re.search(r"RAM (\d+)/(\d+)MB", line)
            if ram_match:
                data["ram_used"] = int(ram_match.group(1))
                data["ram_total"] = int(ram_match.group(2))

            # SWAP
            swap_match = re.search(r"SWAP (\d+)/(\d+)MB", line)
            if swap_match:
                data["swap_used"] = int(swap_match.group(1))
                data["swap_total"] = int(swap_match.group(2))

            # CPU Nutzung
            cpu_match = re.search(r"CPU \[(.*?)\]", line)
            if cpu_match:
                cpu_usages = [int(re.search(r"(\d+)%@", core).group(1)) for core in cpu_match.group(1).split(',') if re.search(r"\d+%@\d+", core)]
                data["cpu"] = cpu_usages

            # GPU-Auslastung
            gr3d_match = re.search(r"GR3D_FREQ (\d+)%", line)
            if gr3d_match:
                data["gpu_usage"] = int(gr3d_match.group(1))

            # Temperaturen
            temps = {}
            for sensor in ["cpu", "gpu", "soc0", "soc1", "soc2", "tj"]:
                t_match = re.search(rf"{sensor}@([\d\.]+)C", line)
                if t_match:
                    temps[sensor] = float(t_match.group(1))
            if temps:
                data["temperature"] = temps

            # Energieverbrauch (Leistung in mW)
            for name in ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]:
                match = re.search(rf"{name} (\d+)mW/(\d+)mW", line)
                if match:
                    data[f"{name.lower()}_current"] = int(match.group(1))
                    data[f"{name.lower()}_average"] = int(match.group(2))

            return data
        except Exception as e:
            print(f"Fehler beim Parsen: {e}")
            return None

    parsed_data = []
    simple_data = []
    energy_data = []

    for tegrastats_log, batch_size in input_logs:
        with open(tegrastats_log, "r") as f:
            for line in f:
                parsed = parse_tegrastats_line(line)
                if parsed:
                    parsed_data.append(parsed)

                    # Start und Endzeit in Json eintragen
                    print(parsed["timestamp"])
                    fmt = "%Y-%m-%dT%H:%M:%S"
                    t1 = datetime.strptime(parsed["timestamp"], "%Y-%m-%dT%H:%M:%S")


                    # aus json auslesen
                    timestamps_file = Path(__file__).resolve().parent.parent / "outputs" / "radioml" / "energy_metrics" / f"timestamps_{batch_size}.json"
                    with open(timestamps_file, "r") as f:
                        timestamps = json.load(f)
                    start_iso = timestamps["start_time"]
                    end_iso = timestamps["end_time"]

                    start = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%S.%f")
                    end = datetime.strptime(end_iso, "%Y-%m-%dT%H:%M:%S.%f")

                    
                    diff1 = abs((t1 - start).total_seconds())
                    diff2 = abs((t1 - end).total_seconds())

                    if (diff1 <= 0.5) or (diff2 <= 0.5):
                        bar = True
                    else:
                        bar = False

                    # RAM vereinfachte Daten
                    if "timestamp" in parsed and "ram_used" in parsed and "ram_total" in parsed:
                        simple_data.append({
                            "timestamp": parsed["timestamp"],
                            "ram_used": parsed["ram_used"],
                            "ram_total": parsed["ram_total"],
                            "batch_size": batch_size,
                            "bar_start_end": bar
                        })

                    # Energy-Daten in einzelne Objekte splitten
                    if all(k in parsed for k in ["vdd_gpu_soc_current", "vdd_cpu_cv_current", "vin_sys_5v0_current"]):
                        print("all keys present")
                        energy_data.extend([
                            {
                                "timestamp": parsed["timestamp"],
                                "type": "vdd_gpu_soc_current",
                                "value": parsed["vdd_gpu_soc_current"],
                                "batch_size": batch_size,
                                "bar_start_end": bar
                            },
                            {
                                "timestamp": parsed["timestamp"],
                                "type": "vdd_cpu_cv_current",
                                "value": parsed["vdd_cpu_cv_current"],
                                "batch_size": batch_size,
                                "bar_start_end": bar
                            },
                            {
                                "timestamp": parsed["timestamp"],
                                "type": "vin_sys_5v0_current",
                                "value": parsed["vin_sys_5v0_current"],
                                "batch_size": batch_size,
                                "bar_start_end": bar
                            }
                        ])

    # Speichern
    with open(output_json_full, "w") as f:
        json.dump(parsed_data, f, indent=2)
    print(f"{len(parsed_data)} Einträge in '{output_json_full.name}' gespeichert .")

    with open(output_json_simple, "w") as f:
        json.dump(simple_data, f, indent=2)
    with open(output_json_simple2, "w") as f:
        json.dump(simple_data, f, indent=2)
    print(f"{len(simple_data)} Einträge in '{output_json_simple.name}' gespeichert (vereinfacht).")

    with open(output_json_energy, "w") as f:
        json.dump(energy_data, f, indent=2)
    with open(output_json_energy2, "w") as f:
        json.dump(energy_data, f, indent=2)
        
    print(f"{len(energy_data)} Einträge in '{output_json_energy.name}' gespeichert  (Energieverbrauch).")



if __name__ == "__main__":
    parse_tegrastats()



#current oder gemittelt?