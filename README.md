# scam-miner

Rust miner for Scummy Bank (dn42). Rewrite of `miner.py`.

## Build

Vulkan/OpenCL/HIP is enabled by default:

```sh
cargo build --release
```

CPU-only:

```sh
cargo build --release --no-default-features
```

## Usage

```sh
# list detected backends and their devices
scam-miner list-backends

# benchmark each backend for 5s (or a specific device)
scam-miner benchmark
scam-miner benchmark --backend opencl --device 0

# start mining (optionally pick a specific device)
scam-miner mine --token <API_TOKEN> [--difficulty 6] [--batch-size 100] [--account DN420042...] [--device 0]
```

See `scam-miner mine --help` for all options.

## Backends

- `cpu` — multi-threaded SHA-256 (SHA-NI / ARMv8 / software)
- `vulkan` — GPU compute via `wgpu`
- `opencl` — GPU compute via `opencl3` (ROCm OpenCL / Mesa RustiCL / other ICDs)
- `hip` — GPU compute via ROCm HIP (hand-rolled FFI, hiprtc runtime compilation)

> **Mesa RustiCL** requires the `RUSTICL_ENABLE=radeonsi` environment variable to expose AMD GPUs:
>
> ```sh
> RUSTICL_ENABLE=radeonsi scam-miner benchmark --backend opencl
> ```
>
> **HIP** loads `libamdhip64.so` + `libhiprtc.so` at runtime from the ROCm
> install. If ROCm is under `/opt/rocm/lib` (not the default loader path), make
> sure it is resolvable, e.g. via `/etc/ld.so.conf.d/rocm.conf` or
> `LD_LIBRARY_PATH=/opt/rocm/lib`.

## LICENSE

AGPL-v3
