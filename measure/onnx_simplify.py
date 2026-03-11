"""
onnx_simplify.py

Runs onnxsim's simplify() in an isolated subprocess so that heap corruption
in the native C++ code (which manifests as 'free(): invalid pointer' or
'free(): invalid next size') cannot crash the parent process.

Each call spawns a fresh Python interpreter, which gets its own clean heap.
If simplify crashes or returns check=False, the original unsimplified model
is copied to the output path instead, and a warning is printed.
"""

import shutil
import subprocess
import sys
import textwrap


def simplify_onnx(export_path: str, simplified_path: str, timeout: int = 600) -> bool:
    """
    Simplify an ONNX model using onnxsim in a subprocess.

    Parameters
    ----------
    export_path:
        Path to the input .onnx file.
    simplified_path:
        Path where the simplified (or fallback unsimplified) .onnx is written.
    timeout:
        Maximum seconds to wait for the subprocess (default 10 min).

    Returns
    -------
    True  – simplification succeeded.
    False – simplification failed or crashed; unsimplified copy written instead.
    """
    script = textwrap.dedent(f"""
        import onnx
        from onnxsim import simplify
        model = onnx.load({export_path!r})
        model_simplified, check = simplify(model)
        if check:
            onnx.save(model_simplified, {simplified_path!r})
            print("SIMPLIFY_OK")
        else:
            print("SIMPLIFY_FAIL")
    """)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            print(f"[onnxsim] Subprocess crashed (returncode={result.returncode}) "
                  f"for {export_path}")
            if stderr:
                # Print last few lines to avoid flooding the log
                for line in stderr.splitlines()[-5:]:
                    print(f"  stderr: {line}")
            _copy_fallback(export_path, simplified_path)
            return False

        if "SIMPLIFY_OK" in stdout:
            print(f"Simplified gespeichert: {simplified_path}")
            return True
        else:
            print(f"[onnxsim] check=False für {export_path}")
            if stderr:
                for line in stderr.splitlines()[-5:]:
                    print(f"  stderr: {line}")
            _copy_fallback(export_path, simplified_path)
            return False

    except subprocess.TimeoutExpired:
        print(f"[onnxsim] Timeout nach {timeout}s für {export_path}")
        _copy_fallback(export_path, simplified_path)
        return False


def _copy_fallback(export_path: str, simplified_path: str) -> None:
    shutil.copy(export_path, simplified_path)
    print(f"[onnxsim] Fallback: unsimplified Kopie gespeichert → {simplified_path}")
