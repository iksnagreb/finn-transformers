Powermodus ändern:
    1: 15 Watt
    2: 30 Watt
    3: 50 Watt
    0: no constraints (kann zu überhitzung kommen)


Change power mode:
```
sudo nvpmodel -q    # check power mode
sudo nvpmodel -m 2   # set mode (e.g., 15W)
sudo jetson_clocks

```

# Experimente anschauen:
1. Alle experimente pullen (geht genauso langsam wie wenn ich nur eins pulle):
```
dvc exp pull origin -A -r upload
```
2. Experiment apply
```
dvc exp apply TODO
```
3. Plots sind in 
```
dvclive/report.md
```
4. Jsons sind dort wo sie erstellt wurden, eg.
```
outputs/radioml/plot/INT8/energy_consumption.json
```

# Language Configs

## Modellübersicht

| Konfiguration | `num_layers` | `num_heads` | `emb_dim` |
|---|---:|---:|---:|
| Config 1 (~base/4) | 1 | 3 | 96 |
| Config 2 (base/2)| 3 | 3 | 192 |
| Config 3 (base)| 6 | 6 | 384 |


## Config 1

### FP16

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |

### FP32

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |

### INT8

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |

## Config 2 

### FP16  (trained CEG)

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `ci-165241-647646` |
| 30 W | `TODO` |
| 50 W | `TODO` |

### FP32  (trained CEG)

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `next` |
| 30 W | `TODO` |
| 50 W | `TODO` |

### INT8 

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |

## Config 3

### FP16

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |

### FP32 

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |

### INT8 

| Power Mode | Experiment-ID |
|---|---|
| 15 W | `TODO` |
| 30 W | `TODO` |
| 50 W | `TODO` |
