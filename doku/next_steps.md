Powermodus ändern:

    2: 30 Watt
    1: 15 Watt
    3: 50 Watt
    0: no constraints (kann zu überhitzung kommen)


Change power mode:
```
sudo nvpmodel -q    # check power mode
sudo nvpmodel -m 2   # set mode (e.g., 15W)
sudo jetson_clocks   # lock clocks - test that
```
Restart is needed to change power mode. What happens if I restart the jetson?
![alt text](image-3.png)

test power modes with **normal radioml**:

    - 30 Watt
        - Commit: 34754810c11a7fdd3508db3884dfc6abacb2a442
    - 15 Watt
        - Commit: 67f9e9f61b14ebc1e4fabf87453d5cfc82f590ac
    - 50 Watt
        - Commit: TODO (failed because of DVC)

test power modes with **normal vision**:

    - 30 Watt
        - Commit: a6be9bbd0c4f0e4232924989f39cae98da8bca12
    - 15 Watt (tag: git tag baseline-vision-15w)
        - Commit: 00ef7b8b0cb13f353944fba2f6870b6ba9f1ec9c
    - 50 Watt
        - Commit: 15c4feb163bd84b6eda0e39c7e3999b36b938684

test power modes with **normal language**:

    - 30 Watt
        - Commit: b4ca7ae3dab21e9979f3d49f3b347e8a6381dbd6
    - 15 Watt
        - Commit: c48b2ba3e62eead9dcd3ae07f31a9fecd6b3f706
    - 50 Watt
        - Commit: d84b7381edf822ac7ae092b24a9a26756684c3c4


Radioml (change model size, always with 30 Watt):

    - **normal**
        - num_layers:1
        - emb_dim: 96
        - num_heads: 3
        - expansion_dim: 512
        - Commit: 34754810c11a7fdd3508db3884dfc6abacb2a442 (tags nutzen)

    - bigger:
        - num_layers: 3
        - emb_dims: 128
        - num_heads: 4
        - expansion_dim: 512
        - Commit (30 Watt): 4dd0b68a4fc33ef228095d1fcff6a2c64ffe3981
        - Commmit (50 Watt): e6da147df1da6788092118d9b2b241dcbd8b3ca3
        - Commit (15 Watt): 0b21f8e27ca1bb6fd73031adb61ad2b9ffdf8c3d

    - bigger: train it again bc of bad accuracy, changed model.py:
        Patch Model to instantiate fresh block instances per layer (fix the list-multiplication bug) 

    - even bigger:
        - num_layers: 6
        - emb_dims: 128
        - num_heads: 4
        - expansion_dim: 512
        - Commit: 


        mehr heads testen




        - Vergleiche zusammenfügen (Dashboards selber erstellen, da ddatachain nicht zuverlässig alle commits hat!)
        - größere batch sizes language und vision
        - rätsel LPFAI lösen
        - Quiz VL machen


### How to get the plots of a specific experiment:

1. git fetch origin 'refs/exps/*:refs/exps/*'
2. dvc exp apply ci-128068-520744
3. dvc pull -r upload

the files from the experiment should be in the outputs folder now.