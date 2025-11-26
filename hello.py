import torch
from pathlib import Path
import subprocess
import time
import random
import pandas as pd

print("PyTorch verfügbar:", torch.__version__, flush=True)
print("CUDA verfügbar:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0), flush=True)
    print("Anzahl GPUs:", torch.cuda.device_count(), flush=True)
else:
    print("Keine GPU oder CUDA nicht verfügbar", flush=True)



def start_tegrastats(logfile_path: Path):
    # tegrastats im Hintergrund starten, Ausgabe in Logdatei
    proc = subprocess.Popen(['sudo', 'tegrastats', '--interval', '1000'], stdout=open(logfile_path, 'w'))
    return proc

def stop_tegrastats(proc: subprocess.Popen):
    print("trying t stop tegrastats")
    proc.terminate()  # schickt SIGTERM an tegrastats
    try:
        print("try")
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        print("exce pt")
        proc.kill()


tegrastats_log =  "tegrastats_runner.log"

tegra_proc = start_tegrastats(tegrastats_log)
time.sleep(10)
print("Startzeit :  ", time.time())
stop_tegrastats(tegra_proc)
print("stopped tegrasta ts!")


torch.cuda.empty_cache()
