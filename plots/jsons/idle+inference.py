import json

def merge_idle_and_inference(idle_path, inference_path, output_path):
    # JSON-Dateien laden
    with open(idle_path, "r") as f:
        idle_data = json.load(f)

    with open(inference_path, "r") as f:
        inference_data = json.load(f)

    # idle_data als Dictionary mit (batch_size, type) als Schlüssel
    idle_dict = {(d["batch_size"], d["type"]): d["idle_value"] for d in idle_data}

    # Zusammenführen
    merged_data = []
    for inf in inference_data:
        key = (inf["batch_size"], inf["type"])
        idle_value = idle_dict.get(key)
        if idle_value is not None:
            merged_entry = inf.copy()
            merged_entry["idle_value"] = idle_value
            merged_data.append(merged_entry)
        else:
            print(f"Warnung: Kein Idle-Wert gefunden für {key}")

    # Ergebnis speichern
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)

# Beispielaufruf
merge_idle_and_inference("idle.json", "average_power_per_inference.json", "average_power_per_batch_size.json")
