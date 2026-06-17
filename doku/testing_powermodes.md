Powermodus ändern:

    2: 30 Watt
    1: 15 Watt
    3: 50 Watt
    0: no constraints (kann zu überhitzung kommen)


Change power mode:
```
sudo nvpmodel -q    # check power mode
sudo nvpmodel -m 2   # set mode (e.g., 15W)
sudo jetson_clocks

```

test power modes with **normal radioml**:

    - 30 Watt
        - Commit: 07ebb266c6acb9355426bb96bbc02ff279a87cc6, ci-131471-529090
    - 15 Watt
        - Commit: 67f9e9f61b14ebc1e4fabf87453d5cfc82f590ac  NOT FOUND IN DVC [ci-127073-517969] , after committing and push its there again
    - 50 Watt
        - Commit: 68025e326d6ab5dd75d9883d2becf38d1320d316, ci-131437-528975

test power modes with **normal vision**:

    - 30 Watt
        - Commit: a6be9bbd0c4f0e4232924989f39cae98da8bca12 NOT FOUND IN DVC [logs are full, cant see exp name]
    - 15 Watt (tag: git tag baseline-vision-15w)
        - Commit: 00ef7b8b0cb13f353944fba2f6870b6ba9f1ec9c NOT FOUND IN DVC [ci-127084-518006]
    - 50 Watt
        - Commit: 1b0914ea098ff533f1ab9b146a9c3022c4b0d2d3, ci-131447-529004


test power modes with **normal language**:

    - 30 Watt
        - Commit: b4ca7ae3dab21e9979f3d49f3b347e8a6381dbd6 NOT FOUND IN DVC [ci-127069-517964] NOW ITS IN DVC, after updating dvc datachain its not there anymore..., after commit and push its there (before the gitlab mirror has pulled the changes), after the pipeline in the gitlab mirror has begun: the commit is not there anymore!

        or [ci-128109-520914] (all batch sizes)
    - 15 Watt
        - Commit: c48b2ba3e62eead9dcd3ae07f31a9fecd6b3f706 NOT FOUND IN DVC [ci-127092-518035] NOW ITS IN DVC, not anymore now, after committing and push its there again, not there now
    - 50 Watt
        - Commit: 931c0e49e3d68322e37c1592a8e541b12a518562, ci-131455-529028

Radioml (change model size, always with 30 Watt):

    - **normal**
        - num_layers:1
        - emb_dim: 96
        - num_heads: 3
        - expansion_dim: 512
        - Commit: 34754810c11a7fdd3508db3884dfc6abacb2a442 [ci-127059-517947]

    - bigger:
        - num_layers: 3
        - emb_dims: 128
        - num_heads: 4
        - expansion_dim: 512
        - Commit (30 Watt): 4dd0b68a4fc33ef228095d1fcff6a2c64ffe3981 [ci-128063-520725]
        - Commmit (50 Watt): e6da147df1da6788092118d9b2b241dcbd8b3ca3 [ci-127244-518854]
        - Commit (15 Watt): 0b21f8e27ca1bb6fd73031adb61ad2b9ffdf8c3d [ci-127253-518883]




    - bigger: train it again bc of bad accuracy, changed model.py:
        Patch Model to instantiate fresh block instances per layer (fix the list-multiplication bug) 

    - even bigger:
        - num_layers: 6
        - emb_dims: 128
        - num_heads: 4
        - expansion_dim: 512
        - Commit: 






https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_5.0.2/Performance_Tuning/Maximizing_Performance
sudo jetson_clocks --show
sudo jetson_clocks: fixes clocks to max. frequenz

# Todo:

## Cluster
- Radioml trainieren mit alten parametern (Vergleichswert/überprüfen ob das training klappt)
- größerere Modelle und lernrate verändern
    - more layers
    - more heads

# Jetson
- alle tests nochmal ausführen, mit sudo jetson clocks

# Aufräumen:
- alte pngs löschen - fertig

# Experimente anschauen:
1. Alle experimente pullen (geht genauso langsam wie wenn ich nur eins pulle):
```
dvc exp pull origin -A -r upload
```
2. Experiment apply
```
dvc exp apply ci-....
```
3. Plots sind in 
```
dvclive/report.md
```
4. Jsons sind dort wo sie erstellt wurden, eg.
```
outputs/radioml/plot/INT8/energy_consumption.json
```