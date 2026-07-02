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
        - Commit: fb5101f6193cbf0bf5e8019e0121bd00971f27c1, ci-131492-529136
    - 50 Watt
        - Commit: 68025e326d6ab5dd75d9883d2becf38d1320d316, ci-131437-528975

test power modes with **normal vision**:

    - 30 Watt
        - Commit: 45c1469e0c8c9e57a09dfd50139ba40bf5d7cc58, ci-131472-529092
    - 15 Watt (tag: git tag baseline-vision-15w)
        - Commit: eada526496667733420247f557c6bce64a0fe07b, ci-131497-529142
    - 50 Watt
        - Commit: 1b0914ea098ff533f1ab9b146a9c3022c4b0d2d3, ci-131447-529004


test power modes with **normal language**:

    - 30 Watt
        - Commit: d88996f13a432fa471d01e039ac3a3691366d40f, ci-131484-529114
    - 15 Watt
        - Commit: fac8cd0d931b4adab8540591858f93fcbe22d22e, ci-131506-529170
    - 50 Watt
        - Commit: 931c0e49e3d68322e37c1592a8e541b12a518562, ci-131455-529028

Radioml (change model size):

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

- best radioml:
    - 15 watt: 282e0914885035581075a855c9600abdd0ea61c9, ci-131731-529888
    - 30 Watt: a1566cbd0001aad275a9f8b6c3aba5ae0bd0db1c, ci-131742-529917
    - 50 Watt: 784a3c801ec9f33f2981272286b41f4e00a1dd24, ci-131752-529947


271596b [atrip-fish]
best radioml mit 100 epochen


https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_5.0.2/Performance_Tuning/Maximizing_Performance
sudo jetson_clocks --show
sudo jetson_clocks: fixes clocks to max. frequenz


bigger vision (emb dim 384, num heads 4, num layers 3, accuracy: 72%)
- 50 Watt: ci-134012-537991, 298743c4c1e02ec4ca494b25da375a531da578ec
- 30 Watt: ci-134059-538121, 00c79f71083e10826eacb13a0c53f0fa9724f37b
- 15 Watt:

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
dvc exp apply ci-131471-529090
```
3. Plots sind in 
```
dvclive/report.md
```
4. Jsons sind dort wo sie erstellt wurden, eg.
```
outputs/radioml/plot/INT8/energy_consumption.json
```