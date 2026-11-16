# One-time Jetson usage measurement

This does not change the GitLab pipeline or DVC tracking.

In a separate SSH session on the Jetson, run:

```bash
mkdir -p ~/measurements/vision-base-2
tegrastats --interval 100 \
  > ~/measurements/vision-base-2/tegrastats.log
```

Run the GitLab pipeline, then press `Ctrl+C` to stop `tegrastats`. The log is
stored at `~/measurements/vision-base-2/tegrastats.log`.

The log contains:

- `RAM`: shared CPU/GPU memory usage
- `CPU`: per-core CPU utilization
- `GR3D_FREQ`: GPU utilization

The measurement job does not upload raw `tegrastats` logs as GitLab artifacts,
so copy this file manually if it needs to be retained.
