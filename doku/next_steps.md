Powermodus ändern:

    2: 30 Watt
    1: 15 Watt
    3: 50 Watt
    0: no constraints (kann zu überhitzung kommen)


Change power mode:
```
sudo nvpmodel -q    # check power mode
sudo nvpmodel -m 3   # set mode (e.g., 15W)
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
        - Commit: TODO (failed because of DVC)

test power modes with **normal language**:

    - 30 Watt
        - Commit: b4ca7ae3dab21e9979f3d49f3b347e8a6381dbd6
    - 15 Watt
        - Commit: c48b2ba3e62eead9dcd3ae07f31a9fecd6b3f706
    - 50 Watt
        - Commit: TODO (failed because of DVC)


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
        - Commit (30 Watt): 93de53e4601604953a3151b3032c0fc4cc92e68c
        - Commmit (50 Watt): e6da147df1da6788092118d9b2b241dcbd8b3ca3
        - Commit (15 Watt): 0b21f8e27ca1bb6fd73031adb61ad2b9ffdf8c3d

    - even bigger:
        - num_layers: 6
        - emb_dims: 128
        - num_heads: 4
        - expansion_dim: 512
        - Commit: 


        mehr heads testen