#!/usr/bin/env bash
set -euo pipefail



export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"
export LC_CTYPE=en_US.UTF-8
export PYTHONUTF8=1

# activations=(relu gelu)
# nls=(2 3 4)
# nhs=(3, 4)
# embs=(192, 256, 384)
# lrs=(0.001)

activations=(relu)
nls=(2)
nhs=(3)
embs=(192)
lrs=(0.001)

echo "Queueing Round 1..."
for act in "${activations[@]}"; do
  for nl in "${nls[@]}"; do
    for emb in "${embs[@]}"; do
      expdim=$((4 * emb))
      for nh in "${nhs[@]}"; do
        if (( emb % nh != 0 )); then
          echo "Skipping emb=${emb} nh=${nh}"
          continue
        fi
        dvc exp run --queue \
          --set-param model.activation="${act}" \
          --set-param model.num_layers="${nl}" \
          --set-param model.emb_dim="${emb}" \
          --set-param model.expansion_dim="${expdim}" \
          --set-param model.num_heads="${nh}" \
          --set-param train.epochs=1
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


## Round 2: 
# lrs=(0.001 0.0005 0.0003)
# embs=(192 256 384)
# weight_decays=(0.0 1e-4)
# + bestes norm/activation aus Runde 1


# scluster