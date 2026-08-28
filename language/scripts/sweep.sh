#!/usr/bin/env bash
set -euo pipefail


# export RADIOML_PATH="/scratch/hpc-prf-ekiapp/haka/finn-transformers/data/GOLD_XYZ_OSC.0001_1024.hdf5"
# export RADIOML_PATH_NPZ="/scratch/hpc-prf-ekiapp/haka/finn-transformers/data/GOLD_XYZ_OSC.0001_1024.npz"

export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

## Round 2: 

norms=(batch-norm)
activations=(relu)
nls=(2)
nhs=(3)
embs=(384)
weight_decays=(0.0 1e-4)    # weight_decay: 0.0
lrs=(0.001 0.0005 0.0003) # lr: 0.001



# mirky snip:
# emb dim: 384
# same activation, same norm, same layers, same lr

echo "Queueing Round 2..."
for norm in "${norms[@]}"; do
  for act in "${activations[@]}"; do
    for nl in "${nls[@]}"; do
      for emb in "${embs[@]}"; do
        expdim=$((4 * emb))
        for nh in "${nhs[@]}"; do
          if (( emb % nh != 0 )); then
            echo "Skipping emb=${emb} nh=${nh}"
            continue
          fi
          for wd in "${weight_decays[@]}"; do
            for lr in "${lrs[@]}"; do
              dvc exp run --queue \
                --set-param model.norm="${norm}" \
                --set-param model.activation="${act}" \
                --set-param model.num_layers="${nl}" \
                --set-param model.emb_dim="${emb}" \
                --set-param model.expansion_dim="${expdim}" \
                --set-param model.num_heads="${nh}" \
                --set-param train.optimizer.weight_decay="${wd}" \
                --set-param train.optimizer.lr="${lr}" \
                --set-param train.epochs=50
            done
          done
        done
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


