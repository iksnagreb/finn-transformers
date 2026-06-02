# Normal Language Modell, different Power Modes

Diese Seite zeigt die gleichen Plots für drei Commits (verschiedene Power-Modes) nebeneinander, damit Du die Auswirkungen vergleichen kannst.

## Commits

- `c48b2ba3e62eead9dcd3ae07f31a9fecd6b3f706` — 15 Watt
- `b4ca7ae3dab21e9979f3d49f3b347e8a6381dbd6` — 30 Watt
- `d84b7381edf822ac7ae092b24a9a26756684c3c4` — 50 Watt (tag: [ci-127257-518890])

## Schritte zum Exportieren der Plots (einmalig)

Die Idee: für jeden Commit eine temporäre Worktree-Checkout anlegen, die relevanten Plot-Bilder aus `dvclive/plots/images` in `doku/plots/<shortsha>/` kopieren und dann die Worktree entfernen.

Führe im Repository-Root aus: FUNKTIONIERT NICHT

```bash
set -e
mkdir -p doku/plots
REVS=(
  "c48b2ba3e62eead9dcd3ae07f31a9fecd6b3f706"
  "b4ca7ae3dab21e9979f3d49f3b347e8a6381dbd6"
  "d84b7381edf822ac7ae092b24a9a26756684c3c4"
)
IMGS=(
  "accuracy_comparison_INT8_ORT_INT8_language.png"
  "throughput_comparison_INT8_ORT_INT8_language.png"
  "power_bar_plot_ORT_INT8_language.png"
  "power_bar_plot_INT8_language.png"
)

for rev in "${REVS[@]}"; do
  short=$(echo "$rev" | cut -c1-7)
  echo "Processing $rev -> $short"
  git worktree add .worktree/$short $rev
  mkdir -p doku/plots/$short
  for img in "${IMGS[@]}"; do
    src=.worktree/$short/dvclive/plots/images/$img
    dst=doku/plots/$short/$(basename "$img")
    if [ -f "$src" ]; then
      cp "$src" "$dst"
    else
      echo "Warning: $src not found for $rev"
    fi
  done
  git worktree remove .worktree/$short || true
done

echo "Copied plots to doku/plots/<shortsha>/"
```

Wenn alles geklappt hat, findest Du die Bilder unter `doku/plots/c48b2ba/`, `doku/plots/b4ca7ae/` und `doku/plots/d84b738/`.

## Vergleichsübersicht (Plots nebeneinander)

Tabelle: Spalten sind die Commits (15W / 30W / 50W). Sollte die Datei `doku/plots/<shortsha>/...` vorhanden sein, werden die Bilder angezeigt.

<table>
  <tr>
    <th>Plot</th>
    <th>c48b2ba (15W)</th>
    <th>b4ca7ae (30W)</th>
    <th>d84b738 (50W)</th>
  </tr>
  <tr>
    <td>Accuracy Comparison</td>
    <td><img src="doku/plots/c48b2ba/accuracy_comparison_INT8_ORT_INT8.png" width="320" alt="accuracy c48b2ba"/></td>
    <td><img src="doku/plots/b4ca7ae/accuracy_comparison_INT8_ORT_INT8.png" width="320" alt="accuracy b4ca7ae"/></td>
    <td><img src="doku/plots/d84b738/accuracy_comparison_INT8_ORT_INT8.png" width="320" alt="accuracy d84b738"/></td>
  </tr>
  <tr>
    <td>Throughput Comparison</td>
    <td><img src="doku/plots/c48b2ba/latency_comparison_INT8_ORT_INT8.png" width="320" alt="latency c48b2ba"/></td>
    <td><img src="doku/plots/b4ca7ae/throughput_comparison_INT8_ORT_INT8.png" width="320" alt="throughput b4ca7ae"/></td>
    <td><img src="doku/plots/d84b738/throughput_comparison_INT8_ORT_INT8.png" width="320" alt="throughput d84b738"/></td>
  </tr>
  <tr>
    <td>Power Bar (ORT_INT8)</td>
    <td><img src="doku/plots/c48b2ba/power_bar_plot_ORT.png" width="320" alt="power ort c48b2ba"/></td>
    <td><img src="doku/plots/b4ca7ae/power_bar_plot_ORT.png" width="320" alt="power ort b4ca7ae"/></td>
    <td><img src="doku/plots/d84b738/power_bar_plot_ORT.png" width="320" alt="power ort d84b738"/></td>
  </tr>
  <tr>
    <td>Power Bar (INT8)</td>
    <td><img src="doku/plots/c48b2ba/power_bar_plot_TRT.png" width="320" alt="power trt c48b2ba"/></td>
    <td><img src="doku/plots/b4ca7ae/power_bar_plot_TRT.png" width="320" alt="power trt b4ca7ae"/></td>
    <td><img src="doku/plots/d84b738/power_bar_plot_TRT.png" width="320" alt="power trt d84b738"/></td>
  </tr>
</table>

---

Wenn Du möchtest, kann ich die Kopier-Schritte jetzt ausführen und die Bilder in `doku/plots/` ablegen. Soll ich das tun? 
