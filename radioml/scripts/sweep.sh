#!/usr/bin/env bash
set -euo pipefail


lrs=(0.001 0.0008 0.0005)
embs=(48 96 192)
nls=(1 2)
nhs=(3 4)

# lrs=(0.0023 0.000323)
# embs=(48)
# nls=(1)
# nhs=(3)

echo "Queueing experiments..."
for lr in "${lrs[@]}"; do
  for emb in "${embs[@]}"; do
    expdim=$((4 * emb))
    for nl in "${nls[@]}"; do
      for nh in "${nhs[@]}"; do
        if (( emb % nh != 0 )); then
          echo "Skipping emb=${emb} nh=${nh} (not divisible)"
          continue
        fi
        dvc exp run --queue \
          --set-param train.optimizer.lr="${lr}" \
          --set-param model.emb_dim="${emb}" \
          --set-param model.expansion_dim="${expdim}" \
          --set-param model.num_layers="${nl}" \
          --set-param model.num_heads="${nh}"\
          --set-param train.epochs=60
      done
    done
  done
done
dvc exp show --only-changed

echo "Running all queued experiments..."
dvc exp run --run-all

# dvc exp run --run-all --jobs 3

#!/usr/bin/env bash
set -euo pipefail


# start the sweep


# Test to not fail the eval:

# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8

# source .venv/bin/activate
# cd radioml
# export RADIOML_PATH="/scratch/hpc-prf-ekiapp/haka/finn-transformers/data/GOLD_XYZ_OSC.0001_1024.hdf5"
# export RADIOML_PATH_NPZ="/scratch/hpc-prf-ekiapp/haka/finn-transformers/data/GOLD_XYZ_OSC.0001_1024.npz"
# RUN_VIA_SLURM=1 bash scripts/sweep.sh




#  dvc queue remove --all