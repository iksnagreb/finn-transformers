#!/usr/bin/env bash
set -euo pipefail



export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"
export LC_CTYPE=en_US.UTF-8
export PYTHONUTF8=1


# activations=(relu relu relu relu)
# nls=(3 12 6 3)
# nhs=(4 12 6 3)
# embs=(384 768 384 192)
# lrs=(0.001 0.001 0.001 0.001)


# ev. layer norm instead of batch norm


# Der originale Vision Transformer verwendet:

# LayerNorm -> AssertionError: Unsupported norm: layer-norm
# GELU
# MLP mit Expansion 4×
# keine BatchNorm

activations=(gelu) # swap to gelu, or silu
nls=(12)
nhs=(12)
embs=(768)
lrs=(0.0005)  # 0.00025
norm=(batch-norm)

echo "Queueing experiments..."

for i in "${!activations[@]}"; do
    act="${activations[$i]}"
    nl="${nls[$i]}"
    nh="${nhs[$i]}"
    emb="${embs[$i]}"
    lr="${lrs[$i]}"
    norm="${norm[$i]}"

    expdim=$((4 * emb))

    if (( emb % nh != 0 )); then
        echo "Skipping experiment $((i+1)): emb=${emb}, nh=${nh}"
        continue
    fi

    echo "Queueing experiment $((i+1)): nl=${nl}, nh=${nh}, emb=${emb}, lr=${lr}"

    dvc exp run --queue \
        --set-param model.activation="${act}" \
        --set-param model.num_layers="${nl}" \
        --set-param model.emb_dim="${emb}" \
        --set-param model.expansion_dim="${expdim}" \
        --set-param model.num_heads="${nh}" \
        --set-param model.norm="${norm}" \
        --set-param train.optimizer.lr="${lr}" \
        --set-param train.epochs=150
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



# dvc im training live metrik einbauen/ tensorboard
# training ohne quantisierung mit vision
# language trainieren
# dropout: 0.25 verringern erhöhen