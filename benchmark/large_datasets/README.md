# Large datasets

Ristretto alone on two full-size scanner datasets, at 1, 4 and 8 host threads and on one GPU. The catalog
cases of the cross-toolkit comparison (RistrettoBenchmarking) are small enough that every toolkit finishes them; these are not,
and they show how Ristretto scales to a full 3D volume and to a full multislice non-Cartesian scan.

| case | data | size |
|---|---|---|
| `large_3d_knee_cartesian` | MRIDATA knee `52c2fd53-d233-4444-8bfd-7c454240d314`, the whole volume | 320×320×256, 8 coils, ky-kz variable density, R = 6 |
| `large_multislice_breast_radial` | fastMRI breast `fastMRI_breast_006_2`, every partition of the stack of stars | 83 slices of 320², 16 coils, 144 of 288 golden-angle spokes × 320 samples |

The methods and their effort are the cross-toolkit comparison suite's (`benchmark/utils/ristretto_methods.jl`):
CG-SENSE (10 CGNR iterations), L1-wavelet (20 FISTA iterations), TV (20 ADMM iterations of 10 CG
each) and anisotropic TV by PDHG (200 iterations), λ from the synthetic analogue
(`shepp_logan_3d_8ch_cartesian`, `shepp_logan_2d_8ch_radial`). Each row is compiled on the
analogue case and then timed once, from host data to host image. See `run.jl`'s header for how
each case is prepared.

```sh
benchmark/slurm/submit.sh large_datasets.sh --threads=16 --prepare      # once: build the case cache
benchmark/slurm/submit.sh --production large_datasets.sh --threads=1,4,8
benchmark/slurm/submit.sh --production large_datasets_gpu.sh
```

`large_datasets.sh` books one node and runs the thread counts at the same time, each on the
physical cores of its own NUMA domain. Results land in `results/<backend>.json`, one row per case
and method.

## Results (2026-10-03, commit 24b8ca845 + this directory)

Seconds per reconstruction, host OpenBLAS on an AMD EPYC node of the `cpu` partition, device an
A100-SXM4-40GB. NRMSE is the same on every backend to three digits.

| case | method | 1 thread | 4 threads | 8 threads | A100 | NRMSE |
|---|---|---:|---:|---:|---:|---:|
| knee 3D | CG-SENSE | 44.3 | 66.2 | 37.4 | 0.96 | 0.606 |
| knee 3D | L1-wavelet | 234 | 273 | 153 | 3.15 | 0.537 |
| knee 3D | TV (ADMM) | 743 | 303 | 279 | 12.3 | 0.353 |
| knee 3D | anisotropic TV (PDHG) | 1254 | 704 | 616 | 19.0 | 0.338 |
| breast multislice radial | CG-SENSE | 71.7 | 31.2 | 23.3 | 7.80 | 0.308 |
| breast multislice radial | L1-wavelet | 292 | 101 | 65.9 | 8.85 | 0.313 |
| breast multislice radial | TV (ADMM) | 1055 | 327 | 201 | 18.0 | 0.174 |
| breast multislice radial | anisotropic TV (PDHG) | 1321 | 441 | 309 | 21.8 | 0.174 |

Peak host memory was 21 GB on the host runs and 8.5 GB of host memory on the GPU run. The knee's
host rows barely scale with threads (CG-SENSE is slower at 4 than at 1 thread, and repeated calls
in one process time the same, so FFTW planning is not the cause); the breast scales 3–5× to 8
threads.
