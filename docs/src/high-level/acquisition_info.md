# AcquisitionInfo

`AcquisitionInfo` holds everything about an acquisition that a reconstruction needs: k-space data,
sensitivity maps, image size, subsampling pattern or trajectory, FFT-shift conventions and a
metadata header. It checks them against each other at construction, so a mismatch fails there
rather than inside a solver. The [AcquisitionInfo tutorial](../tutorials/02_acquisition_info.md)
builds one step by step.

## Constructor

```@docs
AcquisitionInfo
```

## Accepted Input Types

### K-space Data

The first argument can be:

1. **Plain `AbstractArray`**: Standard Julia arrays containing k-space data
2. **`NamedDimsArray`**: Arrays with named dimensions for automatic inference
3. **`nothing`**: When setting up acquisition parameters without actual data (e.g., for simulation)

#### Constraints on Dimensions

- The first two dimensions correspond to transformed spatial axes (kx, ky)
- It must be followed by transformed z-axis if 3D encoding is used (kz)
- The next dimension must correspond to coils, if sensitivity maps are provided
- If 2D encoding is used and sensitivity maps are provided, then slice dimensions must be after the coil dimension
- Sensitivity maps must match spatial dimensions of the image:
    - 2D encoding: ``(N_x, N_y, N_c, [N_z])``
    - 3D encoding: ``(N_x, N_y, N_z, N_c)``

| Argument | Accepted forms |
|---|---|
| k-space data (first argument) | a plain array (pass `is3D`); a `NamedDimsArray` (`is3D` inferred from `:kz`); `nothing`, to set up a simulation |
| `sensitivity_maps` | `(nx, ny, ncoil)` in 2D, `(nx, ny, ncoil, nz)` for per-slice maps, `(nx, ny, nz, ncoil)` in 3D; same element type as the k-space |
| `subsampling` | a boolean mask over the encoded axes; `(:, mask_ky)` for a fully sampled readout; a tuple of ranges or index vectors; an array of any of these, one per batch element |
| `image_size` | inferred from fully sampled k-space; required with `nothing` data or a subsampling pattern |

### Dimension Names

Named arrays make the layout explicit and let `AcquisitionInfo` infer what plain arrays need
keywords for:

| Data | Names |
|---|---|
| k-space | `:kx`, `:ky`, then `:kz` for 3D encoding, then `:coil` |
| image, sensitivity maps | `:x`, `:y`, `:z`, `:coil` |
| non-Cartesian k-space, trajectory | the sample axes, `:coil`, batch dimensions, e.g. `(:sample, :spoke, :coil, :time)`; the trajectory `:coord` followed by the same sample axes, e.g. `(:coord, :sample, :spoke)` |
| batch dimensions | any other name: `:time`, `:contrast`, `:echo`, …; the slices of a 2D multi-slice acquisition are `:z` |

The presence of `:kz` makes the acquisition 3D; without it, it is 2D and `is3D` need not be given.
`reconstruct` returns an image carrying the image-side names of the same dimensions.

```@setup acqinfo
using Ristretto
using NamedDims
ksp = rand(ComplexF32, 64, 64, 8)
ksp_named = NamedDimsArray{(:kx, :ky, :coil)}(ksp)
smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(64, 64, 4))
measured_ksp = NamedDimsArray{(:sample, :spoke, :coil)}(rand(ComplexF32, 64, 32, 4))
```

A per-batch-element pattern is the usual way to describe a dynamic acquisition whose lines shift
from frame to frame, which makes the aliasing incoherent along time:

```@example acqinfo
base = falses(64); base[1:3:64] .= true
acq = AcquisitionInfo(nothing; is3D = false, image_size = (64, 64),
    subsampling = [(:, circshift(base, t - 1)) for t in 1:8])
```

When every element keeps the same number of samples, `kspace_data` stays one dense array.

#### Unequal sample counts per frame

A dense `kspace_data` array has one length along its sample axis, so it can only hold a per-frame
pattern when every frame selects the same number of samples. When the counts genuinely differ —
a variable-density dynamic acquisition, rather than a shifted mask of fixed size — the measurement
is held one frame at a time in a [`PartitionedKSpace`](@ref) instead. `simulate_acquisition`
produces one automatically:

```@example acqinfo
counts = [copy(base) for _ in 1:4]
counts[2][2] = true                 # one extra line in frame 2
acq_ragged = AcquisitionInfo(nothing; is3D = false, image_size = (64, 64),
    subsampling = [(:, m) for m in counts])
```

The frames' full k-space grids are identical, so the encoding operator is a `VCAT` of one
`GetIndex` per frame — no new operator type — and its codomain is an `ArrayPartition`, one block per
frame. Reconstruction is otherwise unchanged: task splitting hands each frame its own slice and its
own spec, a temporal regularizer keeps the whole partitioned operator, and **`reconstruct` still
returns an ordinary dense array**, because the *image* size is the same for every frame.

`PartitionedKSpace` is deliberately not an `AbstractArray`: dimension `ragged_dim` has no single
size, and `size(ksp)` says so rather than inventing a number. Ask for `size(ksp, d)` of a
non-ragged dimension, or reach the frames through `parts(ksp)`. Preprocessing and analysis that
needs a rectangle — prewhitening, coil compression, sensitivity estimation, `pseudo_replica`,
partial-Fourier band detection, GRAPPA/SPIRiT, the `KSpaceToImage` signal model — throws a clear
error for a partitioned acquisition instead of quietly reading the wrong samples. `add_noise`
works, since noise is well defined per frame.

```@docs
Ristretto.CartesianAcquisitionInfo
Ristretto.NonCartesianAcquisitionInfo
PartitionedKSpace
Ristretto.is_partitioned
```

### FFT Shift Conventions

K-space is assumed to have DC at the centre (`N ÷ 2 + 1`) and the image origin at index 1.
`shifted_kspace_dims` names the axes whose k-space is stored pre-shifted (DC at index 1), and
`shifted_image_dims` the axes whose image is centred in the field of view, as on a scanner; both
take indices or, for named arrays, dimension names:

```@example acqinfo
AcquisitionInfo(ksp; is3D = false, shifted_kspace_dims = (1, 2))
AcquisitionInfo(ksp_named; shifted_image_dims = (:x, :y))
```

### Building from raw ISMRMRD data (`MRIBase.RawAcquisitionData`)

An `MRIBase.RawAcquisitionData`, as `MRIFiles` reads it from an ISMRMRD file (`RawAcquisitionData(ISMRMRDFile(path))`, or `MRITestData.load_raw`),
converts directly once `MRIFiles` is loaded (MRITestData loads it). The constructor derives the encoding
matrix, k-space layout, coil dimension, subsampling pattern and Cartesian/non-Cartesian dispatch
directly from the ISMRMRD header, instead of hand-assembling arrays from `raw.profiles`:

```julia
using MRIFiles  # loads the MRIFiles extension
using Ristretto

info = AcquisitionInfo(raw)  # raw::MRIBase.RawAcquisitionData
```

- **`kspace_data`** is a `NamedDimsArray` with `:kx`, `:ky` (and `:kz` for a true 3D acquisition —
  see below), `:coil`, and one batch dimension per ISMRMRD counter that actually varies across
  profiles: `:z` (slices), `:contrast`, `:time` (cardiac/dynamic phase), `:repetition`, `:set`,
  `:average`, in that order. A counter that never varies contributes no dimension.
- **`is3D`** is true only when `kspace_encode_step_2` actually varies (`encodedSize[3] > 1` alone
  is not enough — a 2D multi-slice acquisition also has `encodedSize[3] > 1`, but its slices come
  back as the `:z` batch dimension instead).
- **`subsampling`** reflects exactly the samples present in `raw.profiles` — `kspace_data` is *not*
  zero-padded to `encodedSize` — reduced to `Colon()` (fully sampled), a range (one contiguous
  block, e.g. an asymmetric-echo/partial-Fourier readout), or a boolean mask (e.g. an accelerated
  ky pattern), same as any other `AcquisitionInfo`.
- **`sensitivity_maps`** is not derivable from raw k-space and must be passed as a keyword if
  needed: `AcquisitionInfo(raw; sensitivity_maps)`.

#### FFT-shift derivation

Every profile's `kspace_encode_step_1`/`kspace_encode_step_2` and readout sample index are counted
from 0, with the true k=0 line/sample at `encoding_limits.center` / `head.center_sample` — which
need not be the geometric middle of the encoded axis (partial-Fourier and asymmetric-echo
acquisitions in particular). Consistent with "FFT Shift Conventions" above (DC at
`N ÷ 2 + 1`), this constructor places every sample at `raw_index - center + N ÷ 2` along its axis
before construction, so the result already satisfies that convention and no `shifted_kspace_dims`
is needed. Naively placing sample/line `i` at raw position `i + 1` (ignoring `center`) is exactly
the bug this avoids: it silently shifts the reconstructed image by `center - N ÷ 2` samples along
the affected axis.

The *image* domain needs the matching half of the convention, and the constructor sets that too:
`shifted_image_dims` is `(:x, :y)`, or `(:x, :y, :z)` when `is3D`. Ristretto's own default is the
plain-DFT one — image origin at index 1 — which round-trips consistently for k-space that Ristretto
itself simulated, but is not how a scanner stores data: an ISMRMRD acquisition images an object
**centred in the FOV**. Without `shifted_image_dims` every reconstruction from raw data comes out
rolled by half the FOV along each spatial axis. With it, no manual `fftshift` is needed anywhere,
and `estimate_sensitivities(acq)` returns maps on the same (centred) image grid — see
[Coil Sensitivity Estimation](@ref). Maps estimated by hand from a bare k-space array are in Ristretto's
*default* convention and must be `fftshift`ed before being attached to such an acquisition.

#### Non-Cartesian raw data

Raw data whose `raw.params["trajectory"]` is not `"cartesian"` builds a
`NonCartesianAcquisitionInfo` from the profiles themselves (not from `MRIBase.trajectory` /
`MRIBase.rawdata`, which disagree about how profiles are laid out and, for files that number
every profile with its own `repetition`, return a trajectory of mostly zeros against a one-profile
k-space):

- **`kspace_data`** is `(:sample, :readout, :coil, batch...)`, **`trajectory`** is
  `(:coord, :sample, :readout)`, and **`dcf`** — when the profiles carry a trajectory row beyond
  the encoding dimensions, the vendor's density compensation weighting — is `(:sample, :readout)`.
- **`:readout`** indexes the profiles of one slab in acquisition order: spiral interleaves, radial
  spokes, EPI shots.
- **Batch dimensions** come from the same ISMRMRD counters as the Cartesian path, except that a
  counter taking a *different value in every profile* is treated as a running profile counter, not
  as a batch dimension. Dynamic series that separate their frames that way (USC Speech's spiral
  files, for instance) therefore arrive as one long `:readout` axis; split `raw.profiles` into
  frames yourself, one `AcquisitionInfo` per frame, if you want an image per frame.

## Metadata header

An acquisition carries a [`Header`](@ref) with its metadata: geometry, sequence parameters, and
anything else you pass. It is optional and may be incomplete; with no `header` keyword the
acquisition gets an empty one. The images `reconstruct` returns carry a copy of it (see
[`ReconImage`](@ref)), and the [export functions](export.md) write it.

```@example acqinfo
acq = AcquisitionInfo(ksp; is3D = false, header = (; fov = (240, 180), TE = 4.2, protocol = "t1_se"))
settag!(acq, :subject, "s01")
Ristretto.header(acq).TE    # 4.2
```

The keys Ristretto knows (`fov`, `spacing`, `voxel_size`, `slice_thickness`, `orientation`,
`offset`, `TE`, `TR`, `TI`, `flip_angle`, `field_strength`) are checked and converted when set, and read as
properties that are `nothing` when unset (`header(acq).TR`). Any other key is kept as given. A
`fov` without a `spacing` gives the spacing, `fov ./ image_size`. When `fov`, `spacing` and the
image size disagree you get a warning, not an error, since a scanner's field of view may describe
the oversampled grid.

A `Header` you pass is stored as given, not copied (only a missing `spacing` is filled in), and
copies made with `AcquisitionInfo(acq; ...)` or by preprocessing share it: they describe the same
scan.

**Coordinates.** Lengths are in mm, times in ms, angles in degrees and the field in T. Positions
are in the patient coordinate system **LPS** (x towards the patient's left, y posterior, z
superior), as MRD and DICOM store them; the NIfTI export converts to RAS. `orientation` is a 3×3
matrix whose columns are the directions of the image axes x, y and z, `offset` the centre of the
first voxel, and voxel `i` lies at `offset + orientation * ((i .- 1) .* spacing)`. The geometry
always has three axes: `spacing` is positive along x, y and z, and for a multi-slice 2D
acquisition its third entry is the distance between neighbouring slices, so slice `k` lies
`(k - 1) * spacing[3]` along `orientation[:, 3]`; `fov` covers only the encoded axes. A voxel is
as big as the spacing unless `voxel_size` says otherwise, as it does after a strided selection
(`img[z = 1:2:end]`), and a slice of a 2D acquisition is `slice_thickness` thick. The slice
axis of such an acquisition is the one named `:z`. [`AcquisitionInfo(raw::RawAcquisitionData)`](#Building-from-raw-ISMRMRD-data-(MRIBase.RawAcquisitionData))
fills all of these from the MRD header and profiles.

```@docs
Header
Ristretto.header
settag!
gettag
tags
```

## Validation Rules

The constructor throws an error when:

| Rule | Example of a violation |
|---|---|
| k-space dimensions carry k-space names | `NamedDimsArray{(:x, :y, :coil)}` as k-space |
| encoding dimensions come first and in order | `(:ky, :kx, :coil)` |
| sensitivity maps match the image size | 128 × 128 maps for 64 × 64 k-space |
| sensitivity maps and k-space have one element type | `ComplexF64` maps with `ComplexF32` data |
| named k-space with sensitivity maps has a `:coil` axis | `(:kx, :ky)` data with `(:x, :y, :coil)` maps |
| the image size is known | a subsampling pattern with neither k-space data nor `image_size` |
| a 2D acquisition has no `:kz` axis | `(:kx, :ky, :kz, :coil)` with `is3D = false` |

## Updating Existing Configurations

`AcquisitionInfo(acq; field = value, ...)` copies an acquisition and replaces the given fields,
re-running the validation: `AcquisitionInfo(acq; sensitivity_maps = smaps)` attaches maps to an
acquisition built without them.

## Non-Cartesian Acquisitions

Passing a `trajectory` dispatches `AcquisitionInfo` to the non-Cartesian variant. No k-space
data is required to set up the acquisition — this is the normal way to prepare a trajectory for
[`simulate_acquisition`](@ref), with no placeholder array to invent:

```@example acqinfo
traj = NamedDimsArray{(:coord, :sample, :spoke)}(rand(Float32, 2, 64, 32) .- 0.5f0)
acq_nc = AcquisitionInfo(; trajectory = traj, image_size = (64, 64))              # for simulation
acq_nc = AcquisitionInfo(; trajectory = traj, image_size = (64, 64), sensitivity_maps = smaps)
acq_nc = AcquisitionInfo(measured_ksp; trajectory = traj, image_size = (64, 64))  # measured data
```

`AcquisitionInfo` constructs a `NonCartesianAcquisitionInfo` under the hood. That concrete type
is `public` (documented, stable, dispatchable) but **not exported** — `AcquisitionInfo(...)` is
the advertised, non-expert-facing constructor. Code that needs the concrete type explicitly
(e.g. for a type annotation or `isa` check) imports it or qualifies it:

```@example acqinfo
using Ristretto: NonCartesianAcquisitionInfo
# or: Ristretto.NonCartesianAcquisitionInfo
```

[Simulation Tools](simulation.md#Non-Cartesian-Trajectories) lists the ready-made trajectory
generators.

### Per-frame trajectories

A dynamic acquisition whose trajectory changes from frame to frame — golden-angle radial spokes
that keep rotating, spiral arms that interleave across frames — carries one trajectory per frame
as trailing trajectory axes that match the trailing k-space axes. A trajectory
`(:coord, :sample, :spoke, :time)` goes with k-space `(:sample, :spoke, :coil, :time)`: frame `t`
is encoded with `trajectory[:, :, :, t]` alone. A trajectory without the frame axis is shared by
every frame, as before.

```@example acqinfo
traj_t = NamedDimsArray{(:coord, :sample, :spoke, :time)}(rand(Float32, 2, 64, 32, 4) .- 0.5f0)
acq_t = AcquisitionInfo(; trajectory = traj_t, image_size = (64, 64))
# simulated k-space: (:sample, :spoke, :time)
```

The encoding operator builds one NFFT per frame (see [`get_fourier_operator`](@ref)), so a temporal
regularizer couples frames that were each sampled differently — the incoherence that dynamic
compressed sensing relies on. [`density_compensation`](@ref) computes one set of weights per frame,
[`estimate_sensitivities`](@ref) averages the frames after gridding each with its own trajectory,
and the gradient-delay estimators pool the spokes of every frame.

A trajectory axis is a frame axis when it matches the end of the k-space, by name for named
arrays and by size otherwise. With plain arrays, a trajectory whose trailing size also matches the
axis after the samples (a coil count equal to the frame count, or data without a coil axis) keeps
its older meaning — one shared trajectory with more sample axes — so name the axes there.

## Density Compensation (Non-Cartesian)

Non-Cartesian acquisitions (such as radial, spiral, or arbitrary k-space trajectories) need density compensation factors (DCF) to turn the adjoint NFFT into a usable direct (gridding) reconstruction — the adjoint on its own is *not* an inverse for non-uniformly sampled data. `NonCartesianAcquisitionInfo` (`public`, not exported — see above) holds the trajectory and an optional `dcf` array.

**`acq.dcf` defaults to `nothing`, and `nothing` means no density compensation is applied**: the
encoding operator's adjoint stays the mathematically true adjoint of the forward NFFT. This
matters for anything that assumes `𝒜'` is the true adjoint of `𝒜` — operator-norm estimation,
CG/CGNR, and any algorithm built on that relationship. Reconstructing directly from a
`NonCartesianAcquisitionInfo` you have not run `density_compensation` on therefore does *not*
silently pull in a density-weighted adjoint; call `density_compensation` explicitly when you want
one (e.g. for a quick direct reconstruction), and be aware that once you do, the operator's
adjoint is a density-compensated approximate inverse, not the true adjoint. This is also why the
non-Cartesian trajectory generators and [`simulate_acquisition`](@ref) never apply density
compensation on their own.

You can compute the DCF directly using `density_compensation`:

```@example acqinfo
acq = acq_nc # hide
# Compute iterative Pipe-Menon DCF (default)
acq_dcf = density_compensation(acq; method = PipeMenonDCF(maxit = 20))

# Or compute geometric Voronoi DCF for 2D trajectories
acq_vor = density_compensation(acq; method = VoronoiDCF())

# Both repair the two ends of each readout by default; this is how to see what they produced raw
acq_raw = density_compensation(acq; method = VoronoiDCF(; edge_correction = false))
```

```@docs
density_compensation
DensityCompensation
PipeMenonDCF
VoronoiDCF
correct_dcf_edges
```

`PipeMenonDCF` iterates $w \leftarrow w / (C * w)$, with $C$ the gridding kernel convolution,
until the weighted sampling density is flat (Pipe & Menon 1999); `VoronoiDCF` takes each sample's
weight as the area of its Voronoi cell (Rasche et al. 1999).

- Pipe, J. G., & Menon, P. (1999). *Sampling density compensation in MRI: Rationale and an iterative numerical solution.* Magnetic Resonance in Medicine, 41(1), 179-186. <https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V>
- Rasche, V., Proksa, R., Sinkus, R., Börnert, P., & Eggers, H. (1999). *Resampling of data between arbitrary grids using convolution interpolation.* IEEE Transactions on Medical Imaging, 18(5), 385-392.
