#!/usr/bin/env bash
set -euo pipefail


export RADIOML_PATH="/scratch/hpc-prf-ekiapp/haka/finn-transformers/data/GOLD_XYZ_OSC.0001_1024.hdf5"
export RADIOML_PATH_NPZ="/scratch/hpc-prf-ekiapp/haka/finn-transformers/data/GOLD_XYZ_OSC.0001_1024.npz"

export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"


lrs=(0.001)
embs=(192)
nls=(1)
nhs=(4)


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
          --set-param train.epochs=100
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

# ── eb33001 [aglow-cool]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 8ed28f6 [ heigh-wort]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 0f48256 [coaly-help]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── c1ce6ba [pavid-pyre]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── bb3a9bd [beaky-lame]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 214843e [coxal-acts]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── afc6df2 [hammy-wool]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 3bb8784 [farci-leno]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 5d52496 [erect-tune]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 1894aa6 [wired-craw]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 57b38ea [heapy-cony]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 2817332 [misty-ford]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── a58234b [telic-ados]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 24e0b36 [heady-dees]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
#   ├── 41e0d8f [basal-ally]   02:16 PM   Queued    Dvc-task                                     -                                     -   >
# /tmp/tmpouqj4en5/pydoc.out



# 4f6f113 [baser-vega]   02:03 AM       Success   Dvc-task                               0.75236
# bdc2eb8 [typic-damn]   12:34 AM       Success   Dvc-task                               0.71712 

# 4f6f113 [baser-vega]   02:03 AM       Success   Dvc-task                               0.75236                               0.93785                            
# dvc exp diff main 4f6f113

# dvc exp branch 4f6f113 best-radioml-model