# Sweep script

Usage:

starten aus

cd radioml



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



# Changes, 75% Accuracy in 4f6f113

(.venv) [haka@login1 finn-transformers]$ dvc exp diff main 4f6f113
Path                            Metric    main     4f6f113    Change
outputs/radioml/accuracy.yaml   top-1     0.7076   0.75236    0.044764
outputs/radioml/accuracy.yaml   top-5     0.93531  0.93785    0.0025378
dvclive/metrics.json            step      -        0          diff not supported
outputs/language/accuracy.yaml  top-1     0.6699   0.66991    2.855e-06
outputs/language/accuracy.yaml  top-5     0.85738  0.85718    -0.00020127

model_emb_dim: 192
expansion_dim: 768
num_heads: 4
epochs: 60
learning rate: the same as main




git show 4f6f113:params.yaml

Path                  Param                                main                                                                                                                                                                                                                                                                                                                                                                                                                                         4f6f113    Change
radioml/params.yaml   export.batch_size                    1                                                                                                                                                                                                                                                                                                                                                                                                                                            -          diff not supported
radioml/params.yaml   export.do_constant_folding           True                                                                                                                                                                                                                                                                                                                                                                                                                                         -          diff not supported
radioml/params.yaml   export.dynamo                        False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/params.yaml   export.external_data                 False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/params.yaml   export.format                        qonnx                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/params.yaml   export.opset_version                 19                                                                                                                                                                                                                                                                                                                                                                                                                                           -          diff not supported
radioml/params.yaml   export.optimize                      False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/params.yaml   export.split_heads                   True                                                                                                                                                                                                                                                                                                                                                                                                                                         -          diff not supported
radioml/params.yaml   model.emb_dim                        96                                                                                                                                                                                                                                                                                                                                                                                                                                           192        96
radioml/params.yaml   model.expansion_dim                  384                                                                                                                                                                                                                                                                                                                                                                                                                                          768        384
radioml/params.yaml   model.num_heads                      3                                                                                                                                                                                                                                                                                                                                                                                                                                            4          1
radioml/params.yaml   train.epochs                         100                                                                                                                                                                                                                                                                                                                                                                                                                                          60         -40
language/params.yaml  eval.loader.num_workers              32                                                                                                                                                                                                                                                                                                                                                                                                                                           8          -24
radioml/passes.yaml   imports                              ['adhoc_passes']                                                                                                                                                                                                                                                                                                                                                                                                                             -          diff not supported
radioml/passes.yaml   logging.checkpoint                   False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/passes.yaml   logging.keep_intermediates           False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/passes.yaml   logging.verbose                      False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/passes.yaml   model_checker.full_check             True                                                                                                                                                                                                                                                                                                                                                                                                                                         -          diff not supported
radioml/passes.yaml   onnxruntime.full_context_dump        False                                                                                                                                                                                                                                                                                                                                                                                                                                        -          diff not supported
radioml/passes.yaml   onnxruntime.providers                [['CPUExecutionProvider', {}]]                                                                                                                                                                                                                                                                                                                                                                                                               -          diff not supported
radioml/passes.yaml   passes                               ['import-qonnx', 'convert-layouts', 'shape-inference', 'checker', 'verify', 'inline-qonnx', 'inline-batchnorm', 'inline-gemm', 'lower-conv', 'lower-pooling', 'shape-inference', 'checker', 'verify', 'streamline-thresholds', 'streamline', 'checker', 'verify', 'absorb-layouts', 'decompose-thresholds', 'tidyup', 'MoveMulPastTranspose', 'tidyup', 'MoveSplitPastElementwise', 'tidyup', 'unbroadcast', 'tidyup', 'checker', 'verify']  -          diff not supported
radioml/passes.yaml   reference.inp                        ['outputs/radioml/inp.npy']                                                                                                                                                                                                                                                                                                                                                                                                                  -          diff not supported
radioml/passes.yaml   reference.out                        ['outputs/radioml/out.npy']                                                                                                                                                                                                                                                                                                                                                                                                                  -          diff not supported
radioml/passes.yaml   verify.metrics.count_mispredictions  [0, 0]                                                                                                                                                                                                                                                                                                                                                                                                                                       -          diff not supported
radioml/passes.yaml   verify.metrics.max_abs_error         [0.0, 0.5]                                                                                                                                                                                                                                                                                                                                                                                                                                   -          diff not supported
radioml/passes.yaml   verify.tolerance.atol                0.5                                                                                                                                                                                                                                                   



.venv) [haka@login1 finn-transformers]$ git show-ref | grep exp
0560fbd532b263f772735142f90153552d108318 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/armed-sima
4f6f113ca91184082bda278b2cb76e8b9582a0ab refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/baser-vega
0d45d46ae5b345ee19070328f73cb71ca278c662 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/brood-cook
0a6dba25008774b24c7250ed845319fcc4236604 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/bushy-jura
5b239abcce786a7f44d7b82d0b50561a4478bf15 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/curly-bate
e78648be73409a9d370f6298587a44b4e5fcfb67 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/dicey-dees
faba900bf75e02423e37f56510d2d53a8f5c45b0 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/dormy-ankh
6dd1a2c71cd7b9241799fa9b29ed7974a514ac00 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/elfin-rubs
32e04502566642bfedb0a8128612f2b7c12336f8 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/erect-loof
6a086a8365a81e20267db120e6388abc7a8306d5 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/farci-sous
8dba400dd081677132b8dcb6ce34c6c6206896cf refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/finny-weal
9f31a425706b9bd0c17ffcddae26c136e8eac8b4 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/goofy-mene
17b8cf7b22b8876314f93fcecccb785c9a80253d refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/hexed-bots
af0f56d6ff5c552bf09a64912bd36c4e8bd31040 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/kaput-prof
096d73e79f38a6971b1ae7a735110256749f9439 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/level-puku
8692cf815a37c06c1f63cb695899e63f8da902cd refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/magic-soke
e35703c96c99a9aeb6f10877bffecbbb8e4236c2 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/miffy-lift
d61e1c04f9ca89620da661a52d8f98c89ee972f3 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/ovoid-mold
2c726b8696a54cb7f61de0097e8d9ff4a37af201 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/prosy-lats
2fc771929862e0b8bc2515af8fc9ed14318c8be8 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/quack-girl
713f053a57d5eed1cd64236a24c466684b81c2a9 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/ridgy-bael
9e55ea187c8896a0528b5e2eaa1723557a9cee03 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/risen-rand
cb29be6545ceef9d04a714c69431c1fc0bf0b63c refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/roily-mela
7376f07b2c0d834658f405bdf845220b037457bf refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/shoed-lamb
dce4ba035c536fb61805e55476417068e0ba27f9 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/staid-pita
0d01320c3e387cda6cb02d925230144f2fd678d9 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/tarry-raps
675648e81c0561cd79489489be2127aa713dfd9c refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/tasty-play
bdc2eb87be74c4e9dad459d7b9fcad0b386acbf3 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/typic-damn
e00fbbb58bf4e94edbc224bcf6a1f39de478484c refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/veiny-knit
f27d4299928e1c9e12a97cf7153582f5c4e4ebf5 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/waxen-skin
8261e1d66a42dbddbf39213b9fe73cdad37eb726 refs/exps/a9/70adf430e22f97b024c3fe2b83cbbd40da6e68/white-flew
eb33001c2b3ba9436bfc97b40b4eea47afd68bc5 refs/exps/celery/stash


train-measure Jun 17, 2026 - - 0.69136 0.93952 0.6585 > 
├── df8cb6c [quack-girl] - Running Dvc-task 0.37318 0.79833 0.6585 > 
├── e78648b [dicey-dees] 07:48 AM Success Dvc-task 0.62271 0.91949 0.6585 > 
├── 0d01320 [tarry-raps] 06:55 AM Success Dvc-task 0.19636 0.47343 0.6585 > 
├── 17b8cf7 [hexed-bots] 04:34 AM Success Dvc-task 0.18384 0.59234 0.6585 > 
├── 4f6f113 [baser-vega] 02:03 AM Success Dvc-task 0.75236 0.93785 0.6585 > 
├── bdc2eb8 [typic-damn] 12:34 AM Success Dvc-task 0.71712 0.93392 0.6585 > 
├── 6dd1a2c [elfin-rubs] Jun 17, 2026 Success Dvc-task 0.073403 0.35402 0.6585 > 
├── dce4ba0 [staid-pita] Jun 17, 2026 Success Dvc-task 0.13609 0.35546 0.6585 > 
├── 096d73e [level-puku] Jun 17, 2026 Success Dvc-task 0.69976 0.93401 0.6585 > 
├── d61e1c0 [ovoid-mold] Jun 17, 2026 Success Dvc-task 0.67888 0.93681 0.6585 > 
├── f27d429 [waxen-skin] Jun 17, 2026 Success Dvc-task 0.11563 0.42908 0.6585 > 
├── cb29be6 [roily-mela] Jun 17, 2026 Success Dvc-task 0.049005 0.24019 0.6585 > 
├── 0a6dba2 [bushy-jura] Jun 17, 2026 Success Dvc-task 0.59521 0.92688 0.6585 > 
├── 0d45d46 [brood-cook] Jun 17, 2026 Success Dvc-task 0.63358 0.92065 0.6585 > 
├── 675648e [tasty-play] Jun 17, 2026 Success Dvc-task 0.27591 0.57545 0.6585 > 
── 8dba400 [finny-weal] Jun 17, 2026 Success Dvc-task 0.35906 0.82329 0.6585 > 
├── e00fbbb [veiny-knit] Jun 17, 2026 Success Dvc-task 0.26779 0.56441 0.6585 > 
├── 8692cf8 [magic-soke] Jun 17, 2026 Success Dvc-task 0.3697 0.82339 0.6585 > 
├── 2c726b8 [prosy-lats] Jun 17, 2026 - - 0.26937 0.56606 0.6585 > 
├── 9e55ea1 [risen-rand] Jun 17, 2026 - - 0.35392 0.82097 0.6585 > 
├── 0560fbd [armed-sima] Jun 17, 2026 - - 0.59521 0.92688 0.6585 > 
├── 8261e1d [white-flew] Jun 17, 2026 - - 0.63358 0.92065 0.6585 > 
├── 713f053 [ridgy-bael] Jun 17, 2026 - - 0.26937 0.56606 0.6585 > 
├── 32e0450 [erect-loof] Jun 17, 2026 - - 0.35392 0.82097 0.6585 > 
├── af0f56d [kaput-prof] Jun 17, 2026 - - 0.26937 0.56606 0.6585 > 
├── faba900 [dormy-ankh] Jun 17, 2026 - - 0.35392 0.82097 0.6585 > 
├── 5b239ab [curly-bate] Jun 17, 2026 - - 0.26668 0.56454 0.6585 > 
├── 6a086a8 [farci-sous] Jun 17, 2026 - - 0.39725 0.82592 0.6585 > 
├── 7376f07 [shoed-lamb] Jun 17, 2026 - - 0.26668 0.56454 0.6585 > 
├── 9f31a42 [goofy-mene] Jun 17, 2026 - - 0.39725 0.82592 0.6585 > 
├── eb33001 [aglow-cool] Jun 17, 2026 Queued Dvc-task - - - > :

dvc exp diff eb33001

Path                 Param                eb33001    workspace    Change
radioml/params.yaml  model.emb_dim        192        96           -96
radioml/params.yaml  model.expansion_dim  768        384          -384
radioml/params.yaml  model.num_heads      4          3            -1
radioml/params.yaml  model.num_layers     2          1            -1
radioml/params.yaml  train.epochs         60         1            -59
radioml/params.yaml  train.optimizer.lr   0.0005     0.001        0.0005
(.venv) [haka@login1 finn-transformers]$ 

--> noch abwarten bis das letzte exp gelaufen ist

aglow-cool -> schlecht
quack-girl -> schlecht

beide anschauen