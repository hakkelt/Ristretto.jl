# Examples — one real dataset per catalog type

Every script here reconstructs a real scanner dataset with Ristretto. The set
covers one representative file from **every (source, type) group of the
[MRITestData](https://github.com/hakkelt/MRITestData.jl) catalog** — 35 groups over its seven
sources — so between them the scripts exercise every shape of raw data the package is expected
to read: Cartesian 2D and 3D, multi-slice, cine, mapping, EPI, spiral and radial, single-coil
and 58-coil, fully sampled and prospectively undersampled.

They are also the regression net for the raw-data path: the header defects documented in the
comments (a missing dwell time, an unrecorded echo position, a calibration block with a
different readout length, calibration profiles overwriting the centre of the imaging k-space, a
degenerate single-partition 3D slab) were all found by running these datasets, and each script
says what the symptom looks like so the next person recognises it.

## Running them

The `examples` project is a workspace member of the repository, so it shares the root
`Manifest.toml`:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/m4raw/brain_t2.jl
```

One script, one dataset, no arguments. `run_all.jl` runs the whole set, or a subset:

```sh
julia --project=examples examples/run_all.jl              # everything except the large files
julia --project=examples examples/run_all.jl m4raw ocmr   # by source
julia --project=examples examples/run_all.jl fastmri/brain
```

Magnitude images are written to `examples/output/` as 8-bit PGM (`P5`) — readable by every
image viewer and by `ImageMagick`, and free of any plotting dependency. `ExampleUtils.jl`
holds everything that is not specific to one dataset: download and load, the dwell-time
repair, slab selection, noise covariance, the printing helpers and the PGM writer.

### Credentials and disk

Downloads are cached (by default in the `MRITestData` Scratch space; override with
`RISTRETTO_EXAMPLES_DOWNLOAD_PATH`). Two sources are gated:

| Source | What it needs |
| --- | --- |
| `CMRXRECON2024`, `CMRXRECON300` | a Synapse token — `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN` |
| `FASTMRI` | the signed URLs from the fastMRI access e-mail, registered once with `MRITestData.set_fastmri_urls!`; they expire after 90 days |

Sizes run from 0.1 MB to 5.4 GB. The largest files are loaded one slice at a time
(`load_raw(entry; slice = 1)`) so that memory stays bounded, and `run_all.jl` skips them
unless they are named.

## Last verified

All 35 scripts ran to completion on 2026-10-09 at commit `0e3aabe1` (Julia 1.13.1, 8 threads, login
node; the large files included). `m4raw/brain_t1.jl` needed a second attempt after a Zenodo download
timed out, which is the server, not the script.

| source | scripts | result |
| --- | --- | --- |
| mridata.org | 9 | ok |
| OCMR | 2 | ok |
| CMRxRecon2024 | 7 | ok |
| CMRxRecon-300 | 3 | ok |
| USC Speech | 1 | ok |
| M4Raw | 4 | ok |
| fastMRI | 9 | ok |

## What is where

### mridata.org — vendor-exported ISMRMRD, the least uniform source

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `knee_2d_fully_sampled.jl` | `e3573a0f…` | the plain case: 2D Cartesian, 15 coils, fully sampled |
| `knee_3d_undersampled.jl` | `24777e9a…` | `center_sample = 0`: no recorded echo position, symmetric readout assumed with a warning |
| `knee_3d_fully_sampled.jl` | `52c2fd53…` | a single-partition 3D slab, where the calibration window does not fit and the maps come back all zero |
| `brain_3d_with_calibration_block.jl` | `25952770…` | two readout lengths in one file: imaging profiles plus a flagged 32×32 calibration scan |
| `other_3d_fully_sampled.jl` | `0e0b0437…` | catalogued 3D, actually a 15-slice 2-echo 2D FSE |
| `other_3d_undersampled.jl` | `4a39e0bb…` | 1.9 GB MPRAGE, 58 channels: the scaling test |
| `spectroscopy_semilaser.jl` | `a133799b…` | not an image — semi-LASER spectroscopy, and how to tell |
| `spectroscopy_svs_unlabelled.jl` | `a9c5a204…` | single-voxel spectroscopy, 0.1 MB, one FID |
| `breast_2d_spectroscopy.jl` | `e3a5bed8…` | the same protocol under a `breast` label |

### OCMR — cardiac cine

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `heart_cine_undersampled.jl` | `us_0167_pt_1_5T` | `:time` batch dimension, noise profiles for prewhitening, x-f sparsity |
| `heart_cine_fully_sampled.jl` | `fs_0152_0_55T` | the 0.55 T fully sampled reference to undersample against |

### CMRxRecon2024 — `.mat` challenge data, 10 virtual coils

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `cine_lvot.jl` | `Cine/TrainingSet/P045/cine_lvot` | cardiac phases arrive in `:contrast`, and what to do about it |
| `aorta_sag.jl` | `Aorta/TrainingSet/P087/aorta_sag` | two batch dimensions, flow-dominated contrast |
| `flow2d.jl` | `Flow2d/TestSet/P039/flow2d` | phase-sensitive data: one set of maps for the whole pair |
| `t1map.jl` | `Mapping/TrainingSet/P157/T1map` | 9 inversion times, near-null images among them |
| `t2map.jl` | `Mapping/TrainingSet/P196/T2map` | 3 preparation times; the smallest CMRxRecon file |
| `tagging.jl` | `Tagging/TestSet/P032/tagging` | a tag grid that regularization likes to erase |
| `blackblood.jl` | `BlackBlood/TestSet/P001/blackblood` | plain multi-slice 2D, no frame dimension |

### CMRxRecon-300 — prospectively undersampled, scanner coils

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `cine_lax.jl` | `TrainingSet/P003/cine_lax` | 30 channels, 48 of 132 ky lines, coil compression |
| `t2map.jl` | `TestSet/P111/t2map` | the cheap smoke test for Synapse credentials |
| `t1map_calibration_lines.jl` | `TestSet/P111/t1map` | calibration profiles mixed into the imaging k-space: zero maps until they are split off |

### USC Speech — the spiral source

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `vocal_tract_spiral_realtime.jl` | `sub029/2drt/04_bvt_r2` | NUFFT gridding, vendor density compensation in the trajectory, a recovered dwell time, frames rebuilt from profile order |

### M4Raw — 0.3 T brain, 4 channels

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `brain_flair.jl` | `multicoil_train/2022062402_FLAIR01` | per-slice sensitivity maps over an 18-slice stack |
| `brain_gre.jl` | `gre/2022062704_GRE02` | the noisiest contrast: regularization strength |
| `brain_t1.jl` | `multicoil_train/2022062402_T103` | RSS against SENSE; repeated acquisitions |
| `brain_t2.jl` | `multicoil_train/2022062402_T203` | fully sampled, ready to undersample retrospectively |

### fastMRI — knee, brain, prostate, breast

| Script | Dataset | Why it is here |
| --- | --- | --- |
| `brain_multicoil_undersampled.jl` | `multicoil_test/file_brain_AXFLAIR_203_6000903` | 23 of 213 ky lines: what `subsampling` is for |
| `brain_multicoil_fully_sampled.jl` | `multicoil_train/file_brain_AXT1POST_201_6002738` | the reference half of fastMRI |
| `knee_multicoil_undersampled.jl` | `multicoil_test/file1000747` | slice-at-a-time loading of an 820 MB file |
| `knee_multicoil_fully_sampled.jl` | `multicoil_train/file1001465` | coil compression at 15, 8 virtual coils |
| `knee_singlecoil_undersampled.jl` | `singlecoil_test/file1001483` | emulated single channel: only the prior can help |
| `knee_singlecoil_fully_sampled.jl` | `singlecoil_train/file1002078` | the simplest real dataset in the catalog |
| `prostate_t2.jl` | `file_prostate_AXT2_023` | large FOV, small structure of interest |
| `prostate_diffusion.jl` | `file_prostate_AXDIFF_056` | the only EPI data, and what it still needs |
| `breast_stack_of_stars.jl` | `fastMRI_breast_006_2` | golden-angle stack-of-stars radial, 3D non-Cartesian |

## Dataset terms

The scripts call `MRITestData.dismiss_terms_notice!()` so that the output stays readable. That
suppresses a notice, not the terms: mridata.org, OCMR, CMRxRecon, USC Speech, M4Raw and
fastMRI each license their data separately, several require citation, and fastMRI and
CMRxRecon require registration. See
[MRITestData's licensing page](https://hakkelt.github.io/MRITestData.jl/stable/legal/).
