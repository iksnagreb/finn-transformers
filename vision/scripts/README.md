# Sweep script

Usage:

starten aus

cd radioml

RUN_VIA_SLURM=1 bash scripts/sweep.sh

1. Make the script executable (optional):

```bash
chmod +x scripts/sweep.sh
```

2. Run the sweep (will queue experiments then run them):

```bash
bash scripts/sweep.sh
```

What it does:
- Queues combinations for `train.optimizer.lr`, `model.emb_dim`, `model.num_layers`, `model.num_heads`.
- Skips combinations where `model.emb_dim % model.num_heads != 0`.
- Computes `model.expansion_dim = 4 * model.emb_dim` for each experiment.
- Runs all queued experiments with `dvc exp run --run-all`.
- Selects the experiment with the highest `accuracy` metric and applies it to the working tree with `dvc exp apply`.

Notes:
- The script looks for an `accuracy` metric inside the experiment metrics JSON. Adjust the metric name/path if your training writes a different metric file or key.
- To keep experiment history, use `dvc exp branch` or `dvc exp commit` as shown after the script runs.

