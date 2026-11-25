# Exports the RadioML *evaluation* split (exactly the split used by
# radioml/export.py and radioml/eval.py) into an NPZ file for the measure
# scripts on the Jetson.
#
# Usage (from repo root):
#   RADIOML_PATH=/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5 \
#   python -m measure.radioml_eval_to_npz [--out PATH] [--num-samples N | --all]
#       [--hdf5-out PATH]
import argparse
import os

import h5py
import numpy as np
import yaml

from radioml.dataset import get_datasets

DEFAULT_IN = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
DEFAULT_OUT = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024_eval.npz"

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="radioml/params.yaml")
parser.add_argument("--out", default=DEFAULT_OUT)
parser.add_argument("--num-samples", type=int, default=10000,
                    help="Random subset size of the eval split (seeded)")
parser.add_argument("--all", action="store_true",
                    help="Export the complete eval split (~1.5 GB)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hdf5-out", default=None,
                    help="Additionally save the same samples as HDF5 in the "
                         "original RadioML layout (X, one-hot Y, Z)")
args = parser.parse_args()

with open(args.params, "r") as f:
    params = yaml.safe_load(f)

path = os.environ.get("RADIOML_PATH", DEFAULT_IN)
# Same call as in radioml/export.py -> identical train/valid/eval split
_, _, eval_data = get_datasets(path=path, **params["dataset"])
eval_indices = np.asarray(eval_data.indices)
print(f"Eval split size: {len(eval_indices)} samples")

if args.all or args.num_samples >= len(eval_indices):
    selected = eval_indices
else:
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(eval_indices, size=args.num_samples, replace=False)

# HDF5 fancy indexing requires increasing indices; read in chunks to limit RAM
order = np.argsort(selected)
sorted_idx = selected[order]
reshape = params["dataset"].get("reshape")
X = np.empty((len(sorted_idx), *reshape), dtype=np.float32)
chunk = 4096
for start in range(0, len(sorted_idx), chunk):
    idx = sorted_idx[start:start + chunk]
    X[start:start + len(idx)] = eval_data.file["X"][idx].reshape(len(idx), *reshape)
    print(f"  read {start + len(idx)}/{len(sorted_idx)}", end="\r")
print()
Y = eval_data.cls[sorted_idx].astype(np.int64)
Z = eval_data.snr[sorted_idx]

# Restore the random order (measure scripts use shuffle=False)
inverse = np.argsort(order)
X, Y, Z = X[inverse], Y[inverse], Z[inverse]

# Key order matters: measure scripts take key_list[0] as input and
# key_list[1] as label. Z (SNR) is appended last and ignored by them.
np.savez(args.out, X=X, Y=Y, Z=Z)
print(f"Saved {len(X)} eval samples to {args.out}")
print(f"X {X.shape} {X.dtype}, Y {Y.shape}, SNR range {Z.min()}..{Z.max()}")

if args.hdf5_out:
    num_classes = eval_data.file["Y"].shape[1]
    with h5py.File(args.hdf5_out, "w") as f:
        # Same layout as GOLD_XYZ_OSC.0001_1024.hdf5, readable by RadioMLDataset
        f.create_dataset("X", data=X.reshape(len(X), *eval_data.file["X"].shape[1:]))
        f.create_dataset("Y", data=np.eye(num_classes, dtype=np.int64)[Y])
        f.create_dataset("Z", data=Z.reshape(-1, 1).astype(np.int64))
    print(f"Saved {len(X)} eval samples to {args.hdf5_out}")
