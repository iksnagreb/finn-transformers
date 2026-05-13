import os
import time
import json
import gc
import subprocess
import signal
from pathlib import Path
import yaml
from dvclive import Live
from datetime import datetime

# Ensure ORT can lazily load CUDA libs (same as measure_onnxruntime)
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from measure.parse_tegrastats_to_json import parse_tegrastats
from measure.power_averages_log import (
    power_averages,
    power_averages_baseline,
    power_averages_difference,
    power_averages_baseline_inference,
)
from measure.throughput_power import power_throughput

# Re-use ONNX Runtime measurement helpers
from measure.measure_onnxruntime import (
    run_accuracy_eval,
    get_model_io_info,
    create_ort_session,
    create_test_dataloader,
    run_inference_ort,
)

from onnxconverter_common import float16


FP16 = os.environ.get("FP16", "0") == "1"

GPU_MEM_LIMIT_GB = float(os.environ.get("GPU_MEM_LIMIT_GB", "2.0"))
GPU_MEM_LIMIT_BYTES = int(GPU_MEM_LIMIT_GB * 1024 * 1024 * 1024)
print(f"GPU memory budget: {GPU_MEM_LIMIT_GB:.2f} GB ({GPU_MEM_LIMIT_BYTES} bytes)")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "vision")
if MODEL_TYPE not in ("radioml", "language", "vision"):
    print("Defaulting Model Type to vision model.")
    MODEL_TYPE = "vision"

with open(f"{MODEL_TYPE}/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

bits = cfg["model"]["embedding"].get("bits", 0)
INT8 = (bits == 8)

if INT8:
    print("INT8 enabled")
elif FP16:
    print("FP16 enabled")
else:
    print("FP32")


RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"
CIFAR10_PATH_NPZ = R"/data/gitlab/cifar-10-batches-py/cifar10.npz"
LANG_PATH_NPZ = R"/data/gitlab/language.npz"

DATA_PATH_NPZ = {
    "radioml": RADIOML_PATH_NPZ,
    "vision": CIFAR10_PATH_NPZ,
    "language": LANG_PATH_NPZ,
}[MODEL_TYPE]


def start_tegrastats(logfile_path: Path):
    logfile_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=open(logfile_path, "w"), preexec_fn=os.setsid)
    return proc


def stop_tegrastats(proc: subprocess.Popen):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print("killed tegrastats process")


def run_accuracy_sweep(batch_size, onnx_model_path, tegrastats_log, timestamps_file):
    # get model IO info (may be dynamic)
    input_info, output_info = get_model_io_info(onnx_model_path)

    # create ORT session explicitly and verify CUDA EP is active
    session = create_ort_session(onnx_model_path)
    providers = session.get_providers()
    print("ORT active providers:", providers)
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider not active in ORT session. Ensure onnxruntime-gpu is installed and CUDA available.")

    # start tegrastats
    tegra_proc = start_tegrastats(tegrastats_log)

    time.sleep(10)

    start_ts = time.time()
    start_iso = datetime.fromtimestamp(start_ts).isoformat(timespec='milliseconds')

    # build test loader and run inference loop using existing ORT helpers
    try:
        test_loader = create_test_dataloader(DATA_PATH_NPZ, batch_size, onnx_model_path)
        num_executions = 5 if batch_size > 32 else 1
        for i in range(num_executions):
            _, _, _, accuracy = run_inference_ort(
                session=session,
                test_loader=test_loader,
                batch_size=batch_size,
                input_info=input_info,
                output_info=output_info,
                accuracy_flag=True,
            )
    finally:
        end_ts = time.time()
        end_iso = datetime.fromtimestamp(end_ts).isoformat(timespec='milliseconds')
        with open(timestamps_file, "w") as f:
            json.dump({"start_time": start_iso, "end_time": end_iso}, f, indent=2)
        stop_tegrastats(tegra_proc)
        try:
            del session
        except Exception:
            pass
        gc.collect()

    return accuracy


if __name__ == "__main__":
    if (MODEL_TYPE == "language") or (MODEL_TYPE == "vision"):
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    onnx_model_path = f"outputs/{MODEL_TYPE}/model_dynamic_batchsize.onnx"

    if INT8:
        quant_type = "ORT_INT8"
    elif FP16:
        quant_type = "ORT_FP16"
    else:
        quant_type = "ORT_FP32"

    energy_base_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "energy_metrics" / quant_type
    throughput_base_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / quant_type
    energy_base_path.mkdir(parents=True, exist_ok=True)
    throughput_base_path.mkdir(parents=True, exist_ok=True)

    tegrastats_logs = []

    for batch_size in batch_sizes:
        print("Batch size:", batch_size)
        if INT8:
            # Use batch-specific simple model (may include TopK wrapper or not)
            current_onnx = f"outputs/{MODEL_TYPE}/model_brevitas_{batch_size}_simple.onnx"
        else:
            current_onnx = onnx_model_path

        input_info, output_info = get_model_io_info(current_onnx)
        tegrastats_log = energy_base_path / f"tegrastats_{batch_size}.log"
        timestamps = energy_base_path / f"timestamps_{batch_size}.json"

        accuracy = run_accuracy_sweep(batch_size, current_onnx, tegrastats_log, timestamps)
        print(f"Accuracy for batch size {batch_size}: {accuracy:.4f}")

        tegrastats_logs.append((tegrastats_log, batch_size))

    # parse tegrastats and compute power metrics
    parse_tegrastats(tegrastats_logs, energy_base_path, throughput_base_path)

    energy_consumption_file = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / quant_type / "energy_consumption.json"
    power_averages_file = energy_base_path / "power_averages.json"
    power_averages_file_baseline = energy_base_path / "power_averages_baseline.json"
    power_averages_difference_file = energy_base_path / "power_averages_difference.json"
    power_averages_file_baseline_inference = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / quant_type / "power_averages_baseline_inference.json"

    power_averages(batch_sizes, power_averages_file, energy_consumption_file, quant_type, MODEL_TYPE)
    power_averages_baseline(batch_sizes, power_averages_file_baseline, energy_consumption_file, quant_type, MODEL_TYPE)
    power_averages_difference(batch_sizes, power_averages_file, power_averages_file_baseline, power_averages_difference_file, quant_type, MODEL_TYPE)

    power_averages_baseline_inference(power_averages_file_baseline, power_averages_file, power_averages_file_baseline_inference)

    power_throughput_path = throughput_base_path / "power_throughput.json"
    throughput_path = throughput_base_path / "throughput_results.json"
    power_path = energy_base_path / "power_averages.json"

    power_throughput(power_path, throughput_path, power_throughput_path)

    _prev_dvc_loglevel = os.environ.get("DVC_LOGLEVEL")
    os.environ["DVC_LOGLEVEL"] = "ERROR"
    try:
        with Live(save_dvc_exp=True, report="md") as live:
            print("Start DVC Live report....", flush=True)

            live.log_artifact(energy_consumption_file, name="energy_consumption")
            live.log_artifact(power_averages_file, name=f"power_averages_{quant_type}_{MODEL_TYPE}")
            live.log_artifact(power_averages_file_baseline, name=f"power_averages_baseline_{quant_type}_{MODEL_TYPE}")
            live.log_artifact(power_averages_difference_file, name=f"power_averages_difference_{quant_type}_{MODEL_TYPE}")
            live.log_artifact(power_throughput_path, name=f"power_throughput_{quant_type}_{MODEL_TYPE}")
            live.log_artifact(power_path, name=f"power_averages_{quant_type}_{MODEL_TYPE}")

            live.next_step()
    finally:
        if _prev_dvc_loglevel is None:
            os.environ.pop("DVC_LOGLEVEL", None)
        else:
            os.environ["DVC_LOGLEVEL"] = _prev_dvc_loglevel

    print("DVC Live report ready!")

    gc.collect()
    os._exit(0)
