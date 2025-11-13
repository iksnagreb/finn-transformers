import json
from pathlib import Path
from collections import defaultdict

LATENCY_PATH = Path("/home/hanna/git/measure-radioml/outputs/radioml/throughput/FP32/latency_results_batch.json")     
THROUGHPUT_PATH = Path("/home/hanna/git/measure-radioml/outputs/radioml/throughput/FP32/throughput_results.json")
OUT_PATH = Path("/home/hanna/git/measure-radioml/outputs/radioml/throughput/FP32/latency_throughput.json")

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def latency_throughput(LATENCY_PATH, THROUGHPUT_PATH, OUT_PATH):
    lat = load_json(LATENCY_PATH)
    thr = load_json(THROUGHPUT_PATH)

    # sum latency parts per batch_size
    lat_sum = defaultdict(float)
    for e in lat:
        bs = int(e["batch_size"])
        lat_sum[bs] += float(e.get("value", 0.0))

    # build summary
    thr_map = {int(e["batch_size"]): e for e in thr}
    summary = []
    for bs, t in sorted(thr_map.items()):
        latency_total = lat_sum.get(bs, None)
        entry = {
            "batch_size": bs,
            "latency_total": latency_total,
            "throughput_images_per_s": float(t.get("throughput_images_per_s", 0.0))
        }
        summary.append(entry)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", OUT_PATH, "entries:", len(summary))

if __name__ == "__main__":
    latency_throughput(LATENCY_PATH, THROUGHPUT_PATH, OUT_PATH)