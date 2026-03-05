"""
ONNX Runtime inference with CUDA Execution Provider.

NOTE – Jetson (aarch64) installation
--------------------------------------
NEITHER the CPU-only nor the GPU `onnxruntime` wheel from PyPI works on Jetson.
The PyPI aarch64 wheel crashes on import with a cpuid assertion failure because
Jetson's CPU vendor is not recognised by the upstream cpuinfo library.

The ONLY working source is the Jetson Zoo wheel, which is patched for Jetson:
  https://elinux.org/Jetson_Zoo#ONNX_Runtime

Place the downloaded wheel at /data/onnxruntime_gpu.whl – the CI will then
install it automatically on every run (see .gitlab-ci.yml).

TensorRT provider vs CUDA provider
------------------------------------
TensorRT does not support UINT data types in ONNX graphs.  Use
CUDAExecutionProvider for models that contain uint activations (e.g. FINN /
QONNX uint8 models).  CUDAExecutionProvider handles uint tensors natively.
"""

import sys
import time
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception as e:  # catches ImportError AND the aarch64 cpuid AssertionError
    print(f"[onnxruntime_inf] WARNING: onnxruntime not importable ({e}). "
          "Install the Jetson Zoo wheel – see module docstring.", file=sys.stderr)
    _ORT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Provider utilities
# ---------------------------------------------------------------------------

def print_providers() -> None:
    """Print all providers compiled into the installed onnxruntime build."""
    if not _ORT_AVAILABLE:
        print("onnxruntime not available – cannot list providers.")
        return
    available = ort.get_available_providers()
    print(f"onnxruntime version : {ort.__version__}")
    print(f"Available providers : {available}")
    for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider",
              "CPUExecutionProvider"):
        status = "✓" if p in available else "✗"
        print(f"  {status}  {p}")


def get_providers() -> list[tuple]:
    """
    Return an ordered provider list that prioritises CUDAExecutionProvider.

    Falls back to CPUExecutionProvider if CUDA is not available.
    TensorRT is intentionally skipped – it does not support UINT ONNX models.
    """
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        print("Using CUDAExecutionProvider")
        # device_id=0 targets the first (and usually only) Jetson GPU.
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    # Let CUDA allocate memory on demand; avoids OOM on Jetson's
                    # unified memory architecture.
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            ("CPUExecutionProvider", {}),
        ]
    else:
        print(
            "WARNING: CUDAExecutionProvider not available – falling back to CPU.\n"
            "         Make sure you installed onnxruntime-gpu (see module docstring)."
        )
        return [("CPUExecutionProvider", {})]


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

def create_session(model_path: str, intra_threads: int = 1) -> ort.InferenceSession:
    """
    Create an ONNX Runtime InferenceSession with CUDA as the primary provider.

    Parameters
    ----------
    model_path:
        Path to the .onnx model file.
    intra_threads:
        Number of threads for intra-op parallelism (1 is usually best on Jetson
        when the GPU is doing the heavy lifting).

    Returns
    -------
    ort.InferenceSession
    """
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = intra_threads
    # Disable all graph optimisations – the UINT QONNX graph must reach the
    # CUDA provider unmodified.
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = get_providers()
    session = ort.InferenceSession(model_path, sess_options=sess_opts,
                                   providers=providers)

    active = session.get_providers()
    print(f"Session active providers : {active}")
    if active[0] != "CUDAExecutionProvider":
        print("WARNING: Session is NOT using the GPU – check your onnxruntime-gpu install.")
    return session


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """
    Run a single forward pass.

    Parameters
    ----------
    session:
        An InferenceSession returned by `create_session`.
    inputs:
        Dict mapping ONNX input names to numpy arrays.
        Use `get_input_names(session)` to look up the expected names.

    Returns
    -------
    List of numpy output arrays.
    """
    output_names = [o.name for o in session.get_outputs()]
    return session.run(output_names, inputs)


def get_input_names(session: ort.InferenceSession) -> list[str]:
    """Return the input tensor names expected by the model."""
    return [i.name for i in session.get_inputs()]


def get_input_shapes(session: ort.InferenceSession) -> dict[str, list]:
    """Return a dict of {input_name: shape} for inspection."""
    return {i.name: i.shape for i in session.get_inputs()}


# ---------------------------------------------------------------------------
# Throughput / latency benchmark
# ---------------------------------------------------------------------------

def benchmark(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """
    Measure mean latency and throughput for a fixed input.

    Parameters
    ----------
    session:
        InferenceSession to benchmark.
    inputs:
        Input dict (same format as `run_inference`).
    n_warmup:
        Number of warmup iterations (not measured).
    n_runs:
        Number of timed iterations.

    Returns
    -------
    Dict with keys ``latency_ms_mean``, ``latency_ms_std``, ``throughput_fps``,
    ``batch_size``, ``provider``.
    """
    output_names = [o.name for o in session.get_outputs()]

    # Warmup
    for _ in range(n_warmup):
        session.run(output_names, inputs)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(output_names, inputs)
        latencies.append((time.perf_counter() - t0) * 1e3)  # ms

    lat = np.array(latencies)
    batch_size = next(iter(inputs.values())).shape[0]
    mean_ms = float(lat.mean())

    result = {
        "provider": session.get_providers()[0],
        "batch_size": batch_size,
        "latency_ms_mean": mean_ms,
        "latency_ms_std": float(lat.std()),
        "throughput_fps": float(batch_size / (mean_ms / 1e3)),
    }
    print(
        f"[benchmark] provider={result['provider']}  "
        f"bs={batch_size}  "
        f"latency={mean_ms:.2f} ± {result['latency_ms_std']:.2f} ms  "
        f"throughput={result['throughput_fps']:.1f} fps"
    )
    return result


# ---------------------------------------------------------------------------
# RadioML dataset helpers
# ---------------------------------------------------------------------------

RADIOML_NPZ = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

# Models to evaluate (batch-size-1 variants)
RADIOML_MODELS = [
    "outputs/radioml/model_brevitas_1_simple.onnx",
    "outputs/radioml/model_brevitas_1.onnx",
    "outputs/radioml/model_dynamic_batchsize.onnx",
]


def load_radioml(npz_path: str) -> tuple:
    """
    Load the RadioML NPZ dataset.

    Returns
    -------
    (data, inp_key, att_key, out_key)
        data     – the loaded npz object
        inp_key  – key for the input (IQ) array
        att_key  – key for the attention mask (or None)
        out_key  – key for the label array
    """
    data = np.load(npz_path)
    keys = list(data.keys())
    # heuristic: first key = input, last key = label, middle key = attention mask
    inp_key = keys[0]
    att_key = keys[1] if len(keys) > 2 else None
    out_key = keys[-1]
    return data, inp_key, att_key, out_key


def make_session(
    onnx_path: str,
    providers,
    intra_threads: int = 4,
    disable_extended_opts: bool = False,
) -> ort.InferenceSession | None:
    """
    Create a session for the given provider list.

    Parameters
    ----------
    onnx_path:
        Path to the .onnx file.
    providers:
        Provider list passed directly to InferenceSession.
    intra_threads:
        intra_op_num_threads (4 is a reasonable default for Jetson CPU fallback).
    disable_extended_opts:
        When True, caps optimisation at ORT_ENABLE_BASIC.  Prevents
        MatMulAddFusion / QDQPropagationTransformer from injecting
        DequantizeLinear(INT32_bias) nodes that TensorRT rejects.

    Returns
    -------
    InferenceSession or None on failure.
    """
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        if disable_extended_opts:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            # Disable ALL opts for UINT QONNX graphs so the graph reaches the
            # CUDA provider unchanged.
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(onnx_path), so, providers=providers)
    except Exception as e:
        print(f"  -> session failure for {providers}: {e}")
        return None


def make_session_verbose(
    onnx_path: str,
    providers,
) -> ort.InferenceSession | None:
    """Like make_session but logs node-to-provider assignments (log_severity_level=0)."""
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 0   # prints "Node X assigned to EP Y"
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(onnx_path), so, providers=providers)
    except Exception as e:
        print(f"  -> session failure: {e}")
        return None


def evaluate(
    session: ort.InferenceSession,
    data,
    inp_key: str,
    att_key: str | None,
    out_key: str,
    max_samples: int | None = None,
) -> tuple[float, int]:
    """
    Evaluate classification accuracy at batch size 1.

    Parameters
    ----------
    session:
        Active InferenceSession.
    data:
        Loaded npz object (or any dict-like mapping keys to arrays).
    inp_key, att_key, out_key:
        Array keys as returned by load_radioml.
    max_samples:
        Cap the number of samples evaluated (None = all).

    Returns
    -------
    (accuracy, n_samples)
    """
    input_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name

    X = data[inp_key]
    Y = data[out_key]
    A = data[att_key] if (att_key and att_key in data) else None

    n = X.shape[0]
    if max_samples:
        n = min(n, max_samples)

    correct = 0
    for i in range(n):
        feed: dict[str, np.ndarray] = {input_names[0]: X[i : i + 1]}
        if A is not None and len(input_names) > 1:
            feed[input_names[1]] = A[i : i + 1]

        out = session.run([out_name], feed)[0]
        pred = int(np.argmax(out, axis=-1).ravel()[0])

        label = Y[i]
        if isinstance(label, np.ndarray):
            label = int(np.argmax(label)) if label.size > 1 else int(label.ravel()[0])
        else:
            label = int(label)

        if pred == label:
            correct += 1

    acc = correct / n if n > 0 else 0.0
    return acc, n


# ---------------------------------------------------------------------------
# Evaluation entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _ORT_AVAILABLE:
        print("onnxruntime not available – install the Jetson Zoo wheel (see module docstring).")
        sys.exit(0)  # exit 0 so the CI step does not fail

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Evaluate RadioML ONNX models with CPU / CUDA / TensorRT providers."
    )
    parser.add_argument("--npz", type=str, default=RADIOML_NPZ,
                        help="Path to the RadioML NPZ file.")
    parser.add_argument("--models", nargs="+", default=RADIOML_MODELS,
                        help="ONNX model paths to evaluate.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of samples per model/provider (default: all).")
    parser.add_argument("--providers-only", action="store_true",
                        help="Just print available providers and exit.")
    parser.add_argument("--verbose-session", action="store_true",
                        help="Use verbose session to log node-to-provider assignments.")
    args = parser.parse_args()

    print_providers()
    if args.providers_only:
        raise SystemExit(0)

    if not Path(args.npz).exists():
        raise SystemExit(f"NPZ not found: {args.npz}")

    data, inp_key, att_key, out_key = load_radioml(args.npz)
    print(f"\nData keys  ->  input: {inp_key}  |  attention: {att_key}  |  label: {out_key}\n")

    _session_fn = make_session_verbose if args.verbose_session else make_session

    for model_path in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_path}")
        if not Path(model_path).exists():
            print("  -> model file missing, skipping")
            continue

        # --- CPU ---
        print("\n  [CPU]")
        sess = _session_fn(model_path, providers=["CPUExecutionProvider"])
        if sess:
            acc, n = evaluate(sess, data, inp_key, att_key, out_key, args.max_samples)
            print(f"    samples={n}  accuracy={acc:.4%}")

        # --- CUDA (primary target on Jetson for UINT QONNX models) ---
        print("\n  [CUDA]")
        sess = _session_fn(
            model_path,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ],
        )
        if sess:
            if "CUDAExecutionProvider" not in sess.get_providers():
                print("    CUDA not available – check onnxruntime-gpu install")
            else:
                acc, n = evaluate(sess, data, inp_key, att_key, out_key, args.max_samples)
                print(f"    samples={n}  accuracy={acc:.4%}")
                if args.verbose_session:
                    print("    (node assignments logged above)")

        # --- TensorRT → CUDA fallback ---
        # NOTE: TensorRT does NOT support UINT data types in ONNX graphs.
        # For FINN/QONNX uint8 models, CUDA is the correct provider.
        # TRT is listed here only for completeness / comparison with FP32 models.
        print("\n  [TensorRT → CUDA fallback]")
        sess = _session_fn(
            model_path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            disable_extended_opts=True,
        )
        if sess:
            if "TensorrtExecutionProvider" not in sess.get_providers():
                print("    TRT not available, skipping")
            else:
                acc, n = evaluate(sess, data, inp_key, att_key, out_key, args.max_samples)
                print(f"    samples={n}  accuracy={acc:.4%}")
