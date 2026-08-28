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

 Experiment                 Created    State    Executor   outputs/radioml/accuracy.yaml:top-1   outputs/radioml/accuracy.yaml:top-5   outputs/vision/accuracy.yaml:top-1   outputs/vision/accuracy.yaml:top-5   outp>
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>
  workspace                  -          -        -                                      0.76008                               0.93898                               0.6585                               0.9745       >
  train-measure              02:47 PM   -        -                                      0.76008                               0.93898                               0.6585                               0.9745       >
  ├── 40964e1 [leafy-hull]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 263bd28 [waspy-wire]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 92de52e [nowed-arcs]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 9ea2544 [robed-berk]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 0d3792e [store-obit]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── b6c5aeb [kinky-cham]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 960c6ad [crisp-raja]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 2b14944 [nasty-umbo]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 6bdb7aa [rooky-meed]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── e475686 [conic-sink]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  ├── 9759f83 [pupal-vase]   02:48 PM   Queued   Dvc-task                                     -                                     -                                    -                                    -       >
  └── c7c8a4c [rathe-dole]   02:48 PM   Queued   Dvc-task                                