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

- best radioml:
    - 15 watt: 282e0914885035581075a855c9600abdd0ea61c9, ci-131731-529888
    - 30 Watt: a1566cbd0001aad275a9f8b6c3aba5ae0bd0db1c, ci-131742-529917
    - 50 Watt: 


- vision base/4: 
    - 15 watt: 13b83500c0f90eabb05b2abc3bc21dc087b52d89, ci-137765-550751
    - 30 watt: b1cdbc2c68a2d3eea87a8ae415dc2267dc313d9f, ci-137717-550529
    - 50 watt: d25373d7a30ceaeaf3a4c2e09922e13d27b2d98e, ci-137709-550489
    - FP32: is trained already

- vision base/2: jazzy-cogs (0.74 accuracy)
    - 15 watt: 25c6dacfbebf10d86940d15f4debd1ddd3e367d7, ci-136567-547139
    - 30 watt: e80fd4fd95a21db469060ba84d8485202ea9ef4a, ci-136553-547096
    - 50 watt: 0c7c2b58a0e8b9a92557c1221bf25af24c9e489b, ci-136537-547026
    - FP32: is trained already

- vision base FP32 -> 0.8286 accuracy, eval dataset
    - 15 watt: 
    - 30 watt: df99dc5b64dcb59f35c5c81b27754cd2164b052f
    - 50 watt: 5fab971f1f673020c12e7eb9cce4dc73ca632c05, ci-139320-555878
- vision base FP16 -> 0.8286 accuracy
    - 15 watt: 
    - 30 watt:
    - 50 watt: 


https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_5.0.2/Performance_Tuning/Maximizing_Performance
sudo jetson_clocks --show
sudo jetson_clocks: fixes clocks to max. frequenz



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


- train base model without quantisation -> works 
- experiments for base/4 model fp32 and fp16 in different power modes


- training with layer-norm - done -> layernorm and noquant and 0.0001 LR is working, as well as 0.00001 LR
- change dropout parameter (try doing that with the base/2 model, without or with quantisation)?

- train the language model (on cluster if available)

dropout sweep:
-> base/2 vision without quantisation, with layer norm
-> only very little changes in dopout 0 ... 0.35, then the accuracy gets bad


-> todo use evaluation data for evaluation -> fertig

Freitag:
-> power modi in throughput comparison kombinieren -> neuer branch, für dvc
-> power modi & Modelle (extra dimension) in throughput comparison kombinieren -> seaborn
-> language model trainieren