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


## Evaluation with eval dataset, vision:

- vision base/2 INT8 & ORT_INT8
    - 15 watt: 668f15e5165607d7ade2c78aa286a079bd451fc0, ci-149099-590654
    - 30 watt: 4c0628268a4aaaa0fb01d68366eca6357ecd2987, ci-149017-590270
    - 50 watt: 2be7e4e522b6aeefa7c80b52a36f463c544594f2, ci-149075-590584
- vision base/2 FP16
    - 15 watt: 6426f0d8e24ce26bb427e9be45e5f725c6183b91, ci-150711-597254
    - 30 watt: 02dc34f238ff6eda1b0c5408f637d7a2e753827f, ci-150703-597238
    - 50 watt: ac521abd6b996db1ed571d10e104062a018c6536, ci-149139-590748
- vision base/2 FP32
    - 15 watt: b8f1ed6ba2cbcf7bade8773080d53781443e1deb, ci-150714-597259
    - 30 watt: c24f80b4acac6e39ca1a7d57e4c1cc49818f2502, ci-150695-597208
    - 50 watt: 3b17c7a03fdde11260fb515fab2a66d6c73cd3e0, ci-149142-590753
- vision base/2 INT8 & ORT_INT8 (signed instead of unsigned)
    - 15 watt: 5ad3ec4945c6d07f781bc23006a63b95ce5bf870, ci-154419-608791
    - 30 watt: 9214c5ce64a8e0c7eeac3209ee146544c3f95c62, ci-154428-608813
    - 50 watt: 0395c0e4cf8a3e4151fcfc295cef9cc2e250d90b, ci-154435-608828
- individual layers, fp32
    - 15 watt: e55e787b919e01d6e66562c5e884b384bfa55f99, ci-160990-631006
    - 30 watt: 7571db495a905f224c9dc543ce33bccbc92999ab, ci-160980-630979
    - 50 watt: e83a2325b14318eb5a27fe4e8f4e67417f7bdc2b, ci-159647-627017


- vision base/4 INT8 & ORT_INT8
    - 15 watt: bc5cef0c29396a2b695f0780565850f0a9750002, ci-148567-588762
    - 30 Watt: f3bce167791c7d993968072cbf63d0e87024c5ce, ci-149008-590221
    - 50 watt: 0e5d4254bc8cccc5f8ccc56530bfd4e4d5253fce, ci-148580-588796
- vision base/4 FP16
    - 15 watt: c26ea69c6d7d4e9ba6422ebecb48008ae81ddf6c, ci-150803-597585
    - 30 Watt: 7c2956a3026cc15d5f3a6e155897cbbab44fce2b, ci-150768-597490
    - 50 watt: dfb50cc5686022611c98ab7de7e770328cab6103, ci-150761-597470
- vision base/4 FP32
    - 15 watt: 460a40ad13773ddbcf1801f5b78c558f98a72c1d, ci-150799-597574
    - 30 Watt: fdb625037280c62b0d349d833ef5144f1198e5cf, ci-150771-597495
    - 50 watt: bcb7be87bc334093950b2463366dc3536e9fa093, ci-150747-597418
- vision base/4 INT8 & ORT_INT8 (signed instead of unsigned)
    - 15 watt: 67e2a67bc825256518b83963d4adcae5b059e528, ci-153406-606107
    - 30 Watt: 2b23914bb9379ceb7cd1828b80de34decb9c7861, ci-153280-605765
    - 50 watt: f2e40c2f29e3bcccb212cd2692caf025100262da, ci-153396-606064
- vision base/4 INT8 & ORT_INT8 (signed instead of unsigned, inkl. softmax)
    - 15 watt: facfb8934ca45dcd593fc0cc7a98d9521767d0cf, ci-154683-610018
    - 30 Watt: c5974dc3e9cfccbd780267b544ca00657780ea43, ci-154650-609906
    - 50 watt: 236bc7abd2cb51d9ec9f8edb1e31472bd8223f32, ci-154618-609786

- vision base INT8 & ORT_INT8
    - 15 watt: eeb964837daa5929f98c827aec52e1d24664ccd3, ci-151271-600411
    - 30 watt: aeb5397f748916ab32f6f5bdf2de1d914c02a21a, ci-152127-602776
    - 50 watt: b313741311a3fecc977ec0d1477372d903f101fc, ci-152114-602751
- vision base INT8 & ORT_INT8, signed
    - 15 watt: 8373a0f099f5fcf506e6e3d14da322ead7f4ee1f, ci-156465-617015
    - 30 watt: 849843396831235eabe1cc85ac02ae023d68b4f1, ci-156301-616434
    - 50 watt: bfd77d1b12fd2dc229f9ef4aff459e2e47f3b377, ci-156411-616805
- vision base FP16 -> 0.8286 accuracy
    - 15 watt: aa6ee6bcd671184c9072ab7510e40fc380bb4a46, ci-144941-573891
    - 30 watt: d9947a69695e9323d7f60a70ded169a8140a434d, ci-147874-586629
    - 50 watt: f2475416de98772b79c2433d8c792ece16c250c1, ci-144854-573492
- vision base FP32 -> 0.8286 accuracy, eval dataset
    - 15 watt: f3bd756f25e0577ef12ec5d1b53157bcf794a231, ci-147861-586679
    - 30 watt: 81dac9dfb47b952629d409b9ac81693e2b9ee7b4, ci-144377-571533
    - 50 watt: 5fab971f1f673020c12e7eb9cce4dc73ca632c05, ci-139320-555878




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


