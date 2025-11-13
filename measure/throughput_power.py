import json
from pathlib import Path
from collections import defaultdict

# POWER_PATH = Path("/home/hanna/git/measure-radioml/outputs/radioml/energy_metrics/FP32/power_averages.json")    
# THROUGHPUT_PATH = Path("/home/hanna/git/measure-radioml/outputs/radioml/throughput/FP32/throughput_results.json")
# OUT_PATH = Path("/home/hanna/git/measure-radioml/outputs/radioml/throughput/FP32/power_throughput.json")

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def power_throughput(POWER_PATH, THROUGHPUT_PATH, OUT_PATH):
    OUT_PATH = Path(OUT_PATH)

    power = load_json(POWER_PATH)
    thr = load_json(THROUGHPUT_PATH)

    # sum power parts per batch_size
    power_sum = defaultdict(float)
    for e in power:
        bs = int(e["batch_size"])
        power_sum[bs] += float(e.get("value", 0.0))

    # build summary
    thr_map = {int(e["batch_size"]): e for e in thr}
    summary = []
    for bs, t in sorted(thr_map.items()):
        power_total = power_sum.get(bs, None)
        throughput = float(t.get("throughput_images_per_s", 0.0))

        # calculate throughput per power (throughput / power)
        if power_total is None or power_total == 0:
            throughput_per_power = None
        else:
            throughput_per_power = throughput / float(power_total)

        entry = {
            "batch_size": bs,
            "power_total": power_total,
            "throughput_images_per_s": throughput,
            "throughput_per_power": throughput_per_power
        }
        summary.append(entry)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", OUT_PATH, "entries:", len(summary))

if __name__ == "__main__":
    power_throughput(POWER_PATH, THROUGHPUT_PATH, OUT_PATH)