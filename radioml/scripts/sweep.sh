#!/usr/bin/env bash
set -euo pipefail

# Simple DVC experiment sweep script.
# - enqueues parameter combinations (with validation)
# - runs all queued experiments
# - finds the experiment with the best `accuracy` metric and applies it

lrs=(0.001 0.0008 0.0005)
embs=(48 96 192)
nls=(1 2)
nhs=(3 4)

# lrs=(0.001 0.0005)
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

echo "Selecting best experiment by metric 'accuracy'..."
best=$(dvc exp show --json | python3 - <<'PY'
import sys, json
data=json.load(sys.stdin)
best=None
best_key=None
def search(d):
    if isinstance(d, dict):
        if 'accuracy' in d and isinstance(d['accuracy'], (int,float)):
            return d['accuracy']
        for v in d.values():
            res = search(v)
            if res is not None:
                return res
    return None

for k,v in data.items():
    metrics = v.get('metrics', {})
    acc = search(metrics)
    if acc is None:
        continue
    if best is None or acc > best:
        best = acc
        best_key = k
if best_key:
    print(best_key)
PY
)

if [ -z "$best" ]; then
  echo "No experiment with 'accuracy' metric found."
  exit 0
fi

echo "Best experiment: $best"
dvc exp apply "$best"
echo "Applied experiment $best to working tree. To create a git branch from it run:"
echo "  dvc exp branch best-$best $best"
#!/usr/bin/env bash
set -euo pipefail

# (.venv) [haka@login1 radioml]$ RUN_VIA_SLURM=1 bash scripts/sweep.sh
# Queueing experiments...
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'mirky-tugs' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'muted-toms' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'aulic-pams' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'diazo-poss' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'abuzz-yell' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'awash-kern' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'soled-pans' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'hooly-epha' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'oleic-jato' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'amber-nibs' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'varus-suds' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.001', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'piled-gaur' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'loved-phon' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'ahull-divs' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'pavid-weir' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'empty-suds' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'puffy-inks' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'mesic-sway' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'rural-tarp' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'vocal-aged' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'tidal-snib' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'bovid-hose' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'scald-amie' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0008', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'ethic-stay' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'taped-lulu' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'unlet-rins' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'tinct-sled' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=48', 'model.expansion_dim=192', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'niffy-cusp' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'dural-aqua' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'elfin-merk' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'prone-coof' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=96', 'model.expansion_dim=384', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'bluer-yapp' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=1', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'shoed-jaws' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=1', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'amber-hour' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=2', 'model.num_heads=3', 'train.epochs=60']}'.
# Queued experiment 'rathe-tach' for future execution.
# Queueing with overrides '{'params.yaml': ['train.optimizer.lr=0.0005', 'model.emb_dim=192', 'model.expansion_dim=768', 'model.num_layers=2', 'model.num_heads=4', 'train.epochs=60']}'.
# Queued experiment 'tight-tosh' for future execution.