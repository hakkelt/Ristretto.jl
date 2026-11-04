"""
    add_noise(data; snr_db=nothing, noise_std=nothing, snr=nothing, signal_box=nothing, rng=Random.default_rng())
    add_noise(acq_info::AcquisitionInfo; snr_db=nothing, noise_std=nothing, rng=Random.default_rng())

Add complex white Gaussian noise to simulated data — k-space samples, or an image.

Exactly one of `snr_db`, `noise_std` or `snr` must be given (passing more than one, or none, is
an `ArgumentError`):

- `snr_db`: target signal-to-noise ratio in dB, defined relative to the RMS of the data,
  `σ = rms(data) * 10^(-snr_db / 20)`. This is the convention already used by the
  `add_noise` helper of the comparison suite (RistrettoBenchmarking) and matches how SNR is
  reported for the pseudo-replica method (Robson et al., MRM 60:895-907, 2008; see
  [`pseudo_replica`](@ref)).
- `noise_std`: the absolute standard deviation `σ` of the complex noise, split evenly between
  the real and imaginary parts (each `~ 𝒩(0, (σ/√2)²)`, so that `std(noise) == σ`). This
  matches the `noise_std` keyword already used by [`pseudo_replica`](@ref) and BART's
  `noise -n <stdev>` command (a fixed absolute standard deviation rather than a relative SNR).
- `snr`: target signal-to-noise ratio in the **clinical**, image-domain sense, as a bare ratio
  rather than in decibels — the mean signal in a box at the centre of the image, of size
  `signal_box` in voxels, divided by the noise level (see [`estimate_snr`](@ref), which measures
  the same quantity back from that box and the corners). It is meant for an *image*:
  `add_noise(phantom; snr = 20)` gives an image whose `estimate_snr` is ≈ 20. Passing it for
  k-space data is meaningless, since the "background" of a k-space array is not noise, and it is
  rejected for an `AcquisitionInfo`.

`data` may be a plain `AbstractArray` or a `NamedDimsArray` (dimension names are
preserved). Passing an `AcquisitionInfo` (`CartesianAcquisitionInfo` or
`NonCartesianAcquisitionInfo`) adds noise to `acq_info.kspace_data` and returns a *copy* of
`acq_info` with the noisy k-space (via the existing copy-constructor pattern;
`acq_info` itself is left untouched). `acq_info.kspace_data` must not be `nothing`.

`rng` fixes the noise realization for reproducibility, following the `rng` keyword already used
throughout the package (e.g. [`pseudo_replica`](@ref), `LocallyLowRank`).

# Examples
```julia
noisy_kspace = add_noise(kspace_data; snr_db = 20)
noisy_acq = add_noise(acq; noise_std = 0.01)
noisy_image = add_noise(phantom; snr = 20)
```
"""
function add_noise(
        kspace_data::AbstractArray;
        snr_db::Union{Real, Nothing} = nothing,
        noise_std::Union{Real, Nothing} = nothing,
        snr::Union{Real, Nothing} = nothing,
        signal_box = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    σ = _resolve_noise_std(kspace_data, snr_db, noise_std, snr, signal_box)
    raw = kspace_data isa NamedDimsArray ? NamedDims.unname(kspace_data) : kspace_data
    T = eltype(raw)
    noise = (randn(rng, real(T), size(raw)) .+ im .* randn(rng, real(T), size(raw))) .* (real(T)(σ / sqrt(2)))
    noisy = raw .+ noise
    return kspace_data isa NamedDimsArray ? NamedDimsArray{dimnames(kspace_data)}(noisy) : noisy
end

# Partitioned k-space needs no special case beyond the bookkeeping: σ is resolved once over the
# whole acquisition, so the SNR means the same thing it does for a dense array, and the noise is
# then drawn per frame.
function add_noise(
        kspace_data::PartitionedKSpace;
        snr_db::Union{Real, Nothing} = nothing,
        noise_std::Union{Real, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    σ = _resolve_noise_std(kspace_data, snr_db, noise_std)
    noisy = map(part -> add_noise(part; noise_std = σ, rng), parts(kspace_data))
    return PartitionedKSpace(noisy, kspace_data.ragged_dim, kspace_data.dimnames)
end

function add_noise(
        acq_info::AcquisitionInfo;
        snr_db::Union{Real, Nothing} = nothing,
        noise_std::Union{Real, Nothing} = nothing,
        snr::Union{Real, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    # `snr` segments an image into foreground and background; a k-space array has neither.
    @argcheck isnothing(snr) "`snr` is an image-domain measure and cannot be targeted on k-space; use `snr_db` or `noise_std` here, or add the noise to the image before simulating"
    @argcheck !isnothing(acq_info.kspace_data) "acq_info.kspace_data must not be `nothing`"
    noisy_kspace = add_noise(acq_info.kspace_data; snr_db, noise_std, rng)
    return AcquisitionInfo(acq_info; kspace_data = noisy_kspace)
end

_resolve_noise_std(kspace_data::PartitionedKSpace, snr_db, noise_std, snr = nothing, signal_box = nothing) =
    _resolve_noise_std(to_array_partition(kspace_data), snr_db, noise_std, snr, signal_box)

function _resolve_noise_std(kspace_data, snr_db, noise_std, snr = nothing, signal_box = nothing)
    given = count(!isnothing, (snr_db, noise_std, snr))
    @argcheck given == 1 "exactly one of `snr_db`, `noise_std` or `snr` must be provided"
    if !isnothing(noise_std)
        return noise_std
    end
    if !isnothing(snr)
        @argcheck snr > 0 "`snr` must be positive"
        # `estimate_snr` reports `mean(signal box) / σ_component`, so the noise level per component
        # is `mean(signal box) / snr`; `σ` here is the standard deviation of the complex noise,
        # which is `√2` times the per-component one.
        return sqrt(2) * signal_box_mean(kspace_data, signal_box) / snr
    end
    raw = kspace_data isa NamedDimsArray ? NamedDims.unname(kspace_data) : kspace_data
    rms = norm(raw) / sqrt(length(raw))
    return rms * 10.0^(-snr_db / 20)
end
