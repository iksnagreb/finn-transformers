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



 Experiment                 Created    State    Executor   outputs/vision/accuracy.yaml:top-1   outputs/vision/accuracy.yaml:top-5   outputs/radioml/accuracy.yaml:top-1   outputs/radioml/accuracy.yaml:top-5   outputs/language/>
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>
  workspace                  -          -        -                                      0.6585                               0.9745                               0.76008                               0.93898                    >
  train-measure              02:57 PM   -        -                                      0.6585                               0.9745                               0.76008                               0.93898                    >
  ├── 0c58707 [store-glee]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 08a6307 [plump-slew]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 8d14235 [snowy-user]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 1408cfc [veiny-weld]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── cc45a5e [every-sand]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 0071989 [agaze-buck]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 6994c85 [hooly-bite]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── dd4d9cf [faery-dook]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── e682ac3 [swept-corm]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 2b3c74d [wired-gude]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── afe1d0b [moire-jean]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 7f1b047 [loose-nock]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 28c84a9 [small-shoe]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 5669867 [zonal-mash]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 8de6f0f [mushy-lame]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── cec0455 [color-kibe]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── a2a58d1 [beery-gore]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── a0e3afa [coaly-sine]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── dadd77d [spumy-maul]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── f7e5388 [tacit-wire]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 29dd9a9 [goosy-okra]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 43b8ec1 [duple-snob]   03:05 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 68ef7de [elect-brig]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 0a2191e [japan-ords]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── cdef42a [cheap-oner]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── aa1b848 [goofy-tics]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── ce794cc [jammy-molt]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── 15caf51 [boxed-ribs]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── c274052 [major-hope]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  ├── ee3f2ed [filmy-sida]   03:04 PM   Queued   Dvc-task                                    -                                    -                                     -                                     -                    >
  └── 4c35271 [blond-teds]   02:58 PM   Failed   -                                      0.6585                               0.9745                               0.76008                               0.93898                    >
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>
/tmp/tmpx380qj2w/pydoc.out (END)