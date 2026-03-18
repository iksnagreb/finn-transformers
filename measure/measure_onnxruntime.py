"""
measure_onnxruntime.py

Throughput / latency benchmark using ONNX Runtime with CUDAExecutionProvider.
Mirrors the structure of measure.py (TensorRT) so results can be directly compared.

Timing breakdown (matches measure.py conventions):
  latency_inference   – session.run() wall-clock time only
  latency_synchronize – session.run() + explicit torch.cuda.synchronize()
  latency_datatransfer – full round-trip: numpy prep → session.run() → output retrieval

Note: ORT's session.run() already blocks until all GPU work is finished and the
result has been copied back to CPU, so the synchronize and datatransfer deltas
will typically be very small.  This mirrors the TRT version in measure.py where
the deltas represent the overhead on top of raw kernel execution.
"""

import time
import json
import os
import sys
import gc
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from pathlib import Path
from dvclive import Live

# Prevent PyTorch from initialising a CUDA context before ORT gets a chance
# to allocate its cuBLAS handle.  We import torch lazily (after ORT sessions
# are created) to avoid CUBLAS_STATUS_ALLOC_FAILED on the Jetson.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from measure.latency_throughput_log import latency_throughput

# ---------------------------------------------------------------------------
# Global configuration (mirrors measure.py)
# ---------------------------------------------------------------------------

FP16 = os.environ.get("FP16", "0") == "1"

MODEL_TYPE = os.environ.get("MODEL_TYPE", "vision")
if MODEL_TYPE not in ("radioml", "language", "vision"):
    MODEL_TYPE = "vision"
    print("Defaulting Model Type to vision model.")

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
LANG_PATH_NPZ    = R"/data/gitlab/language.npz"

DATA_PATH_NPZ = {
    "radioml":  RADIOML_PATH_NPZ,
    "vision":   CIFAR10_PATH_NPZ,
    "language": LANG_PATH_NPZ,
}[MODEL_TYPE]

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

ONNX_TO_NP_DTYPE = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
    "tensor(uint8)":   np.uint8,
    "tensor(int8)":    np.int8,
    "tensor(bool)":    np.bool_,
}

def onnx_dtype_to_numpy(onnx_dtype_str: str) -> np.dtype:
    return ONNX_TO_NP_DTYPE.get(onnx_dtype_str, np.float32)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def save_json(log, filepath):
    filepath = Path(filepath)
    filepath.parent.parent.mkdir(parents=True, exist_ok=True)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)


def parse_shape(shape, batch_value):
    """Replace symbolic ONNX dimensions with concrete values (same as measure.py)."""
    resolved = []
    for i, d in enumerate(shape):
        if isinstance(d, str):
            if d == "batch_size" or i == 0:
                resolved.append(batch_value)
            elif d == "sequence_length":
                resolved.append(128)
            elif d == "Muloutput_dim_2":
                resolved.append(64)
            elif d == "channels":
                resolved.append(3)
            else:
                resolved.append(1)
        elif d is None:
            resolved.append(batch_value if i == 0 else 1)
        else:
            resolved.append(int(d))
    return tuple(resolved)


def print_latency(latency_ms, latency_synchronize, latency_datatransfer,
                  end_time, start_time, num_batches,
                  throughput_batches, throughput_images, batch_size):
    print("For Batch Size: ", batch_size)
    print(f"Gemessene durchschnittliche Latenz für Inferenz        : {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation : {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer   : {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time - start_time:.4f} s")
    print("num_batches", num_batches)
    print(f"Throughput: {throughput_batches:.4f} Batches/Sekunde")
    print(f"Throughput: {throughput_images:.4f} Bilder/Sekunde")


# ---------------------------------------------------------------------------
# Model / session helpers
# ---------------------------------------------------------------------------

def get_model_io_info(model_path: str):
    """
    Read input / output metadata from an ONNX model via a plain CPU ORT session.
    Returns lists of dicts with keys 'name', 'shape', 'dtype' (same schema as measure.py).
    """
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 8
    session = ort.InferenceSession(model_path, sess_opts)
    input_info = [
        {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
        for inp in session.get_inputs()
    ]
    output_info = [
        {"name": out.name, "shape": out.shape, "dtype": out.type}
        for out in session.get_outputs()
    ]
    return input_info, output_info


def create_ort_session(onnx_model_path: str) -> ort.InferenceSession:
    """
    Create an ORT InferenceSession backed by CUDAExecutionProvider.
    Falls back to CPU if CUDA is not available.
    Graph optimisations are disabled so UINT QONNX graphs reach the CUDA provider
    unchanged (same rationale as onnxruntime_inf.py).
    """
    available = ort.get_available_providers()
    print(f"Available ORT providers: {available}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    if "CUDAExecutionProvider" in available:
        cuda_providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": int(4 * 1024 * 1024 * 1024),  # 4 GB
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                },
            ),
            ("CPUExecutionProvider", {}),
        ]
        try:
            session = ort.InferenceSession(
                onnx_model_path, sess_options=sess_opts, providers=cuda_providers
            )
            active = session.get_providers()
            print(f"ORT session active providers: {active}")
            if active[0] != "CUDAExecutionProvider":
                print("WARNING: Session is NOT using the GPU – check your onnxruntime-gpu install.")
            return session
        except Exception as e:
            print(f"WARNING: CUDAExecutionProvider failed ({e}), falling back to CPU.")

    print("WARNING: CUDAExecutionProvider not available or failed – using CPU.")
    session = ort.InferenceSession(
        onnx_model_path, sess_options=sess_opts, providers=[("CPUExecutionProvider", {})]
    )
    print(f"ORT session active providers: {session.get_providers()}")
    return session


# ---------------------------------------------------------------------------
# Data loading (same logic as measure.py)
# ---------------------------------------------------------------------------

def create_test_dataloader(data_path_npz: str, batch_size: int, onnx_model_path: str) -> list:
    """Build a DataLoader from the NPZ test split."""
    data = np.load(data_path_npz)
    input_info, output_info = get_model_io_info(onnx_model_path)
    key_list = list(data.keys())
    print("Keys in NPZ file:", key_list)

    if len(input_info) >= 2:
        input_key        = key_list[0]
        attention_mask_key = key_list[1]
        output_key       = key_list[2]
    else:
        input_key          = key_list[0]
        attention_mask_key = None
        output_key         = key_list[1]

    input_ids      = data[input_key]
    attention_mask = data[attention_mask_key] if attention_mask_key else None
    labels         = data[output_key]

    # Build a simple list-of-numpy-arrays DataLoader without importing torch
    n = len(input_ids)
    n = (n // batch_size) * batch_size  # drop_last=True equivalent
    batches = []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        if attention_mask is not None:
            batches.append((input_ids[sl], attention_mask[sl], labels[sl]))
        else:
            batches.append((input_ids[sl], labels[sl]))
    return batches


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference_ort(
    session: ort.InferenceSession,
    test_loader: list,
    batch_size: int,
    input_info: list,
    output_info: list,
    accuracy_flag: bool = False,
):
    """
    Run ORT CUDA-EP inference over all batches in test_loader.

    Timing mirrors the TRT version in measure.py:
      - latency_inference   = session.run() wall time
      - latency_synchronize = session.run() + torch.cuda.synchronize()
      - latency_datatransfer = numpy conversion + session.run() + output retrieval

    Returns
    -------
    (avg_latency_ms, avg_latency_sync_ms, avg_latency_dt_ms, accuracy)
    """
    total_time          = 0.0
    total_time_sync     = 0.0
    total_time_dt       = 0.0
    iterations          = 0

    total_predictions   = 0
    correct_predictions = 0
    do_prints           = True

    output_names = [out["name"] for out in output_info]

    for batch in test_loader:
        if len(batch) == 2:
            xb, yb   = batch
            att_mask = None
            token_type = None
        elif len(batch) == 3:
            xb, att_mask, yb = batch
            token_type = None
        elif len(batch) == 4:
            xb, att_mask, yb, token_type = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        # ── datatransfer timer: numpy conversion + session.run() + output ──
        start_time_dt = time.time()

        inp_dtype = onnx_dtype_to_numpy(input_info[0]["dtype"])
        feed = {input_info[0]["name"]: np.asarray(xb).astype(inp_dtype)}

        if att_mask is not None and len(input_info) > 1:
            att_dtype = onnx_dtype_to_numpy(input_info[1]["dtype"])
            feed[input_info[1]["name"]] = np.asarray(att_mask).astype(att_dtype)

        if token_type is not None and len(input_info) > 2:
            tt_dtype = onnx_dtype_to_numpy(input_info[2]["dtype"])
            feed[input_info[2]["name"]] = np.asarray(token_type).astype(tt_dtype)

        # ── synchronize timer: session.run() (ORT already synchronises internally) ──
        start_time_sync = time.time()

        # ── inference timer: session.run() only ──
        start_time_inf = time.time()
        outputs = session.run(output_names, feed)
        end_time = time.time()
        # ORT's session.run() blocks until GPU work is done and results are
        # copied back to CPU, so no explicit CUDA sync is needed here.
        end_time_sync = time.time()

        # ORT already returns numpy arrays; no extra D2H copy needed
        output = outputs[0]
        end_time_dt = time.time()

        total_time      += end_time      - start_time_inf
        total_time_sync += end_time_sync - start_time_sync
        total_time_dt   += end_time_dt   - start_time_dt
        iterations      += 1

        if accuracy_flag:
            pred    = output.argmax(axis=-1)
            correct = (pred == yb).sum()
            total   = int(np.prod(yb.shape)) if MODEL_TYPE == "language" else yb.shape[0]
            correct_predictions += correct
            total_predictions   += total

        if accuracy_flag and do_prints:
            print("Prediction (Raw):", output[0])
            print("Label:           ", yb[0])
            print("Predicted label: ", pred[0])
            do_prints = False

    accuracy = (
        correct_predictions / total_predictions
        if (accuracy_flag and total_predictions > 0)
        else 0.0
    )

    avg_latency_ms   = (total_time      / iterations) * 1000
    avg_latency_sync = (total_time_sync / iterations) * 1000
    avg_latency_dt   = (total_time_dt   / iterations) * 1000

    return avg_latency_ms, avg_latency_sync, avg_latency_dt, accuracy


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def run_accuracy_eval(batch_size, input_info, output_info, data_path_npz, onnx_model_path):
    print("batch_size           :", batch_size)
    print("input_info           :", input_info)
    print("output_info          :", output_info)
    print("data_path_npz        :", data_path_npz)
    print("onnx_model_path      :", onnx_model_path)

    input_info, output_info = get_model_io_info(onnx_model_path)
    session     = create_ort_session(onnx_model_path)
    test_loader = create_test_dataloader(data_path_npz, 1, onnx_model_path)

    _, _, _, accuracy = run_inference_ort(
        session=session,
        test_loader=test_loader,
        batch_size=batch_size,
        input_info=input_info,
        output_info=output_info,
        accuracy_flag=True,
    )
    del session
    gc.collect()
    return accuracy


# ---------------------------------------------------------------------------
# Latency / throughput sweep
# ---------------------------------------------------------------------------

def calculate_latency_and_throughput(batch_sizes, onnx_model_path, input_info, output_info):
    """
    Sweep over batch sizes and record latency / throughput for ORT CUDA EP.

    ORT supports dynamic batch sizes natively, so a single session is reused
    across all batch sizes (unless the model file changes, e.g. INT8 batch-
    specific ONNX files, in which case a new session is created).
    """
    throughput_log  = []
    latency_log     = []
    latency_log_batch = []

    current_session_path = None
    session = None

    for batch_size in batch_sizes:
        print("Measuring for batch size:", batch_size)

        current_onnx_path = onnx_model_path
        if INT8:
            current_onnx_path = f"outputs/{MODEL_TYPE}/model_brevitas_{batch_size}_simple.onnx"
            print(f"Using INT8 ONNX model: {current_onnx_path}")

        # Re-create session only when the model file changes
        if current_onnx_path != current_session_path:
            if session is not None:
                del session
                gc.collect()
            input_info, output_info = get_model_io_info(current_onnx_path)
            session = create_ort_session(current_onnx_path)
            current_session_path = current_onnx_path

        test_loader = create_test_dataloader(DATA_PATH_NPZ, batch_size, current_onnx_path)

        # Average over num_executions runs (set > 1 for more stable estimates)
        num_executions    = 1
        latency_ms_sum    = 0.0
        latency_sync_sum  = 0.0
        latency_dt_sum    = 0.0
        total_time_sum    = 0.0

        for _ in range(num_executions):
            start_time = time.time()
            latency_ms, latency_sync, latency_dt, _ = run_inference_ort(
                session=session,
                test_loader=test_loader,
                batch_size=batch_size,
                input_info=input_info,
                output_info=output_info,
            )
            end_time = time.time()

            latency_ms_sum   += latency_ms
            # store *deltas* (same convention as measure.py)
            latency_sync_sum += (latency_sync - latency_ms)
            latency_dt_sum   += (latency_dt   - latency_sync)
            total_time_sum   += (end_time - start_time)

        latency_avg      = latency_ms_sum   / num_executions
        latency_sync_avg = latency_sync_sum / num_executions
        latency_dt_avg   = latency_dt_sum   / num_executions
        total_time_avg   = total_time_sum   / num_executions

        num_batches        = int(7600 / batch_size)
        throughput_batches = num_batches / total_time_avg
        throughput_images  = (num_batches * batch_size) / total_time_avg

        # per-sample logs
        log_lat_inf  = {"batch_size": batch_size, "type": "inference",    "value": latency_avg      / batch_size}
        log_lat_sync = {"batch_size": batch_size, "type": "synchronize",  "value": latency_sync_avg / batch_size}
        log_lat_dt   = {"batch_size": batch_size, "type": "datatransfer", "value": latency_dt_avg   / batch_size}
        # per-batch logs
        log_lat_inf_b  = {"batch_size": batch_size, "type": "inference",    "value": latency_avg}
        log_lat_sync_b = {"batch_size": batch_size, "type": "synchronize",  "value": latency_sync_avg}
        log_lat_dt_b   = {"batch_size": batch_size, "type": "datatransfer", "value": latency_dt_avg}
        throughput_entry = {
            "batch_size": batch_size,
            "throughput_images_per_s": throughput_images,
            "throughput_batches_per_s": throughput_batches,
        }

        throughput_log.append(throughput_entry)
        latency_log.extend([log_lat_inf, log_lat_sync, log_lat_dt])
        latency_log_batch.extend([log_lat_inf_b, log_lat_sync_b, log_lat_dt_b])

        # reconstruct cumulative values expected by print_latency (same as measure.py)
        print_latency(
            latency_avg,
            latency_sync_avg + latency_avg,
            latency_dt_avg + latency_sync_avg + latency_avg,
            end_time, start_time,
            num_batches, throughput_batches, throughput_images,
            batch_size,
        )

    if session is not None:
        del session
        gc.collect()

    return throughput_log, latency_log, latency_log_batch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # ORT handles dynamic batch sizes natively → one model file for all batch sizes
    onnx_model_path = f"outputs/{MODEL_TYPE}/model_dynamic_batchsize.onnx"
    if INT8:
        # For INT8 accuracy eval use batch-size-1 QCDQ model
        onnx_model_path = f"outputs/{MODEL_TYPE}/model_brevitas_1_simple.onnx"

    model = onnx.load(onnx_model_path)
    input_info, output_info = get_model_io_info(onnx_model_path)

    # ── Accuracy evaluation ────────────────────────────────────────────────
    accuracy = run_accuracy_eval(1, input_info, output_info, DATA_PATH_NPZ, onnx_model_path)
    print(f"Accuracy (ORT CUDA): {accuracy:.2%}")

    if FP16:
        quantisation_type = "FP16"
        accuracy_path = (
            Path(__file__).resolve().parent.parent
            / "outputs" / MODEL_TYPE / "eval_results" / "accuracy_ORT_FP16.json"
        )
    elif INT8:
        quantisation_type = "INT8"
        accuracy_path = (
            Path(__file__).resolve().parent.parent
            / "outputs" / MODEL_TYPE / "eval_results" / "accuracy_ORT_INT8.json"
        )
    else:
        quantisation_type = "FP32"
        accuracy_path = (
            Path(__file__).resolve().parent.parent
            / "outputs" / MODEL_TYPE / "eval_results" / "accuracy_ORT_FP32.json"
        )

    save_json(
        {"quantisation_type": f"ORT_CUDA_{quantisation_type}", "value": accuracy},
        accuracy_path,
    )

    # ── Latency / throughput sweep ─────────────────────────────────────────
    throughput_log, latency_log, latency_log_batch = calculate_latency_and_throughput(
        batch_sizes, onnx_model_path, input_info=input_info, output_info=output_info
    )

    base_dir = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot"
    if FP16:
        subdir = base_dir / "ORT_FP16"
    elif INT8:
        subdir = base_dir / "ORT_INT8"
    else:
        subdir = base_dir / "ORT_FP32"

    os.makedirs(subdir, exist_ok=True)
    throughput_results      = subdir / "throughput_results.json"
    latency_results_batch   = subdir / "latency_results_batch.json"
    latency_throughput_path = subdir / "latency_throughput.json"

    save_json(throughput_log,    throughput_results)
    save_json(latency_log_batch, latency_results_batch)
    latency_throughput(latency_results_batch, throughput_results, latency_throughput_path)

    with Live(save_dvc_exp=True, report="md") as live:
        print("Starte DVC Live Bericht (ORT CUDA)...", flush=True)
        live.log_artifact(
            throughput_results,
            name=f"ort_throughput_results_{quantisation_type}_{MODEL_TYPE}",
        )
        live.log_artifact(
            latency_results_batch,
            name=f"ort_latency_results_batch_{quantisation_type}_{MODEL_TYPE}",
        )
        live.log_artifact(
            latency_throughput_path,
            name=f"ort_latency_throughput_{quantisation_type}_{MODEL_TYPE}",
        )
        live.next_step()

    print("DVC Live Bericht (ORT CUDA) fertig!")

    # Explicit exit to avoid free(): invalid pointer crash during Python
    # interpreter shutdown.  ORT's CUDA-EP C++ destructors are called in
    # undefined order when the interpreter tears down module globals, which
    # causes a double-free.  At this point all data is written and DVC Live
    # is closed, so a hard exit is safe.
    gc.collect()
    os._exit(0)
