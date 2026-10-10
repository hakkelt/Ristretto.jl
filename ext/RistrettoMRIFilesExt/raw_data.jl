# ISMRMRD flag bit for `ACQ_IS_NOISE_MEASUREMENT` (1-based bit index 19, per the ISMRMRD spec /
# MRIBase.jl's `Flags.jl`). Noise-calibration profiles carry no image k-space and must be
# excluded before the profile list is turned into an array.
const _NOISE_MEASUREMENT_BIT = UInt64(1) << (19 - 1)
_is_noise_profile(p) = (p.head.flags & _NOISE_MEASUREMENT_BIT) != 0

function _image_profiles(raw::RawAcquisitionData)
    profiles = [p for p in raw.profiles if !_is_noise_profile(p)]
    @argcheck !isempty(profiles) "RawAcquisitionData contains no image (non-noise) profiles"
    return profiles
end

"""
    AcquisitionInfo(raw::MRIBase.RawAcquisitionData; sensitivity_maps=nothing)

Build an `AcquisitionInfo` directly from an ISMRMRD `RawAcquisitionData` (as returned by
`MRIFiles.RawAcquisitionData`/`MRITestData.load_raw`), without hand-assembling k-space arrays
from `raw.profiles`.

Derived automatically:
- **Encoding matrix / image size** from `raw.params["encodedSize"]`.
- **Header** ([`Header`](@ref Ristretto.Header)): `fov` and `spacing` from `encodedFOV`, `TE`,
  `TR`, `TI` and `flip_angle` from the sequence parameters, `field_strength` from
  `H1resonanceFrequency_Hz`, and `orientation`, `offset` and `slice_spacing` from the profiles'
  `position`, `read_dir`, `phase_dir` and `slice_dir` (LPS, as MRD stores them).
- **Cartesian vs. non-Cartesian dispatch**, from `raw.params["trajectory"]`: `"cartesian"``
  (case-insensitive) builds a [`CartesianAcquisitionInfo`](@ref); anything else builds a
  [`NonCartesianAcquisitionInfo`](@ref) from `MRIBase.trajectory`/`MRIBase.rawdata`.
- **`is3D`**, from whether `kspace_encode_step_2` actually varies (`enc_lim_kspace_encoding_step_2`)
  — a multi-slice 2D acquisition has `encodedSize[3] > 1` but a *singleton* `kspace_encode_step_2`,
  so it stays 2D with slices as a `:z` batch dimension (see below), not a 3D encoding.
- **Coil dimension** (`:coil`), from `head.active_channels`.
- **The subsampling pattern actually present in the profiles** (not merely "fully sampled"): the
  readout, phase-encode and (if 3D) partition-encode axes are each reduced to the tightest
  `AcquisitionInfo` subsampling spec that reproduces them — `Colon()` when everything in
  `encodedSize` is present, an `OrdinalRange` when the present samples form one contiguous block
  (e.g. an asymmetric-echo/partial-Fourier readout), or a `BitArray` mask otherwise (e.g. an
  accelerated ky pattern). `kspace_data` itself stores only the samples that were actually
  acquired — it is not zero-padded to `encodedSize`.
- **Slices/contrasts/repetitions/etc. as batch dimensions**: any of `slice`, `contrast`,
  `phase`, `repetition`, `set`, `average` that takes more than one distinct value across the
  profiles becomes a batch dimension, named respectively `:z`, `:contrast`, `:time`,
  `:repetition`, `:set`, `:average` (`:z` first, matching the convention in
  `docs/src/high-level/nameddims.md`: "Multi-slice data should include `:z` as the first batch
  dimension"). A counter that never varies contributes no dimension. `is3D` acquisitions with
  more than one slice (multi-slab 3D) are not supported (`:z` cannot coexist with 3D k-space) —
  filter `raw.profiles` yourself and construct a `CartesianAcquisitionInfo` slab-by-slab instead.

Not derivable, so it must be passed explicitly if needed:
- `sensitivity_maps`: `RawAcquisitionData` carries no coil sensitivity information.

## FFT-shift convention

`AcquisitionInfo` expects k-space with the DC component at the geometric centre of each Fourier
axis — index `N ÷ 2 + 1` for an axis of length `N` — which is what `fftshift` produces and what
[`get_fourier_operator`](@ref)'s internal `ifftshift`/`fftshift` pair assumes (see "FFT Shift
Conventions" in `docs/src/high-level/acquisition_info.md`). A profile's `kspace_encode_step_1`
(and, if 3D, `kspace_encode_step_2`) is an index into the *encoded* matrix starting at 0, with the
true k=0 line at `encoding_limits.center` — not necessarily `encodedSize ÷ 2` (partial-Fourier /
asymmetric acquisitions in particular need not centre their limits). Likewise a profile's readout
samples are indexed from 0 with the true k=0 sample at `head.center_sample`, again not necessarily
the middle of the acquired samples (asymmetric echo). This constructor places every sample at
`raw_index - center + N ÷ 2` (1-based: `+1`) along its axis before handing anything to
`CartesianAcquisitionInfo`, so the result is already centred: no `fftshift` call is needed in user
code afterwards. Ignoring these offsets — placing sample/line `i` at raw position `i + 1` — is
exactly the bug this constructor exists to avoid: it silently shifts the reconstructed image by
`center - N ÷ 2` samples along the affected axis.

The *image* domain needs the matching convention. Ristretto's default is the plain-DFT one — image
origin at index 1, so a reconstruction comes out in "FFT order" — which round-trips consistently
for k-space that Ristretto itself simulated, but is not how a scanner stores data: an ISMRMRD
acquisition images an object centred in the FOV. This constructor therefore sets
`shifted_image_dims` to every spatial axis (`(:x, :y)`, or `(:x, :y, :z)` when `is3D`), so
`reconstruct` returns the object centred in the frame and no image-domain `fftshift` is needed
either. Without it every reconstruction from real data comes out shifted by half the FOV along
both in-plane axes.
"""
function Ristretto.AcquisitionInfo(raw::RawAcquisitionData; sensitivity_maps = nothing)
    trajectory_name = lowercase(String(get(raw.params, "trajectory", "cartesian")))
    if trajectory_name == "cartesian"
        return _cartesian_acquisition_info(raw; sensitivity_maps)
    end
    return _noncartesian_acquisition_info(raw; sensitivity_maps)
end

# The metadata header of an MRD acquisition: the encoded field of view, the sequence parameters,
# and the geometry from the profiles' `position` (the centre of the slice or slab) and direction
# cosines, in the patient coordinate system LPS. `img_size` is the reconstruction grid, whose
# centre voxel (`n ÷ 2 + 1`, the FFT-shifted origin) is where `position` lies.
function _raw_header(raw::RawAcquisitionData, profiles, img_size)
    params = raw.params
    nd = length(img_size)
    h = Header()
    fov = get(params, "encodedFOV", nothing)
    if !isnothing(fov) && all(>(0), fov)
        h.fov = fov[1:nd]
        h.spacing = h.fov ./ img_size
        nd == 2 && length(fov) >= 3 && (h.slice_thickness = fov[3])
    end
    for (key, name) in (("TE", :TE), ("TR", :TR), ("TI", :TI), ("flipAngle_deg", :flip_angle))
        v = get(params, key, nothing)
        isnothing(v) && continue
        v isa AbstractVector && length(v) == 1 && (v = only(v))
        (v isa Real || (v isa AbstractVector && !isempty(v))) && (h[name] = v)
    end
    f = get(params, "H1resonanceFrequency_Hz", nothing)
    isnothing(f) || f <= 0 || (h.field_strength = f / _GAMMA_HZ_PER_T)

    head = first(profiles).head
    R = Float64[collect(head.read_dir) collect(head.phase_dir) collect(head.slice_dir)]
    if all(c -> norm(c) > 0.5, eachcol(R))
        h.orientation = R
        # The first slice of a multi-slice acquisition (lowest slice index, the first along `:z`)
        # and the signed distance between neighbouring slices along the slice direction.
        slices = sort!(unique(Int(p.head.idx.slice) for p in profiles))
        centres = [collect(Float64.(first(p for p in profiles if Int(p.head.idx.slice) == s).head.position)) for s in slices]
        if nd == 2 && length(centres) > 1
            steps = [dot(centres[i + 1] - centres[i], R[:, 3]) for i in 1:(length(centres) - 1)]
            h.slice_spacing = steps[1]
            if !all(st -> isapprox(st, steps[1]; rtol = 1.0e-3, atol = 1.0e-3), steps) ||
                    !all(i -> norm(centres[i + 1] - centres[i] - steps[i] * R[:, 3]) < 1.0e-3, eachindex(steps))
                @warn "the slices of this acquisition are not equidistant and parallel; the header's slice_spacing describes the first two"
            end
        end
        if !isnothing(h.spacing)
            sp = nd == 3 ? collect(h.spacing) : [h.spacing..., 0.0]
            n = nd == 3 ? collect(img_size) : [img_size..., 1]
            h.offset = centres[1] .- R * ((n .÷ 2) .* sp)
        end
    end
    return h
end

# Sorted-unique raw ids -> 1-based compact index (preserves ascending order, so a monotonic
# offset such as the FFT-centring shift below cannot reorder them).
function _compact_index_map(ids::AbstractVector{<:Integer})
    uids = sort(unique(ids))
    return Dict(id => i for (i, id) in enumerate(uids)), uids
end

# The most specific `AcquisitionInfo` subsampling spec that reproduces a sorted set of 1-based
# positions along a length-`n` axis: `Colon()` when everything is present, a range when the
# present positions form one contiguous block, else a boolean mask.
function _positions_to_subsampling(n::Integer, positions::AbstractVector{<:Integer})
    if length(positions) == n
        return Colon()
    elseif length(positions) == last(positions) - first(positions) + 1
        return first(positions):last(positions)
    end
    mask = falses(n)
    mask[positions] .= true
    return mask
end

function _cartesian_acquisition_info(raw::RawAcquisitionData; sensitivity_maps = nothing)
    profiles = _image_profiles(raw)

    enc = Int.(raw.params["encodedSize"])
    nkx, nky, nkz_full = enc[1], enc[2], enc[3]
    lim1 = raw.params["enc_lim_kspace_encoding_step_1"]::Limit
    lim2 = get(raw.params, "enc_lim_kspace_encoding_step_2", Limit(0, 0, 0))::Limit
    is3D = lim2.maximum > lim2.minimum
    nkz = is3D ? nkz_full : 1

    p1 = first(profiles)
    nsamples = size(p1.data, 1)
    ncoil = Int(p1.head.active_channels)
    pre = Int(p1.head.discard_pre)
    post = Int(p1.head.discard_post)
    center_sample = Int(p1.head.center_sample)
    T = eltype(p1.data)
    for p in profiles
        @argcheck size(p.data, 1) == nsamples "profiles have differing readout lengths; assemble manually"
        @argcheck Int(p.head.discard_pre) == pre && Int(p.head.discard_post) == post "profiles have differing discard_pre/discard_post; assemble manually"
        @argcheck Int(p.head.center_sample) == center_sample "profiles have differing center_sample; assemble manually"
        @argcheck Int(p.head.active_channels) == ncoil "profiles have differing active_channels; assemble manually"
    end

    rows = (pre + 1):(nsamples - post)
    row_offset = nkx ÷ 2 - center_sample
    kx_lo, kx_hi = first(rows) + row_offset, last(rows) + row_offset
    if center_sample == 0 && !(1 <= kx_lo && kx_hi <= nkx)
        # `center_sample = 0` is how several exporters (mridata.org's GE files among them) spell
        # "not recorded", and it is indistinguishable from an echo genuinely at the first sample.
        # Only the reading that cannot be true is discarded: if placing the echo at sample 0 puts
        # the readout outside the encoded matrix, assume a symmetric readout instead.
        center_sample = length(rows) ÷ 2
        row_offset = nkx ÷ 2 - center_sample
        kx_lo, kx_hi = first(rows) + row_offset, last(rows) + row_offset
        @warn "This ISMRMRD file records no echo position (`center_sample = 0` in every " *
            "profile) and placing the echo at the first sample would put the readout outside " *
            "the encoded matrix. Assuming a symmetric readout, echo at sample " *
            "$(center_sample). Pass k-space assembled by hand if the readout is asymmetric." maxlog = 1
    end
    @argcheck 1 <= kx_lo && kx_hi <= nkx "readout centering places samples outside the encoded matrix ($(kx_lo):$(kx_hi) vs 1:$(nkx)); assemble manually"
    kx_sub = _positions_to_subsampling(nkx, collect(kx_lo:kx_hi))

    step1_ids = [Int(p.head.idx.kspace_encode_step_1) for p in profiles]
    ky_map, ky_uids = _compact_index_map(step1_ids)
    ky_positions = sort(ky_uids .- lim1.center .+ (nky ÷ 2) .+ 1)
    @argcheck first(ky_positions) >= 1 && last(ky_positions) <= nky "phase-encode centering places samples outside the encoded matrix; check enc_lim_kspace_encoding_step_1"
    ky_sub = _positions_to_subsampling(nky, ky_positions)

    kz_map = nothing
    kz_uids = nothing
    kz_sub = nothing
    if is3D
        step2_ids = [Int(p.head.idx.kspace_encode_step_2) for p in profiles]
        kz_map, kz_uids = _compact_index_map(step2_ids)
        kz_positions = sort(kz_uids .- lim2.center .+ (nkz ÷ 2) .+ 1)
        @argcheck first(kz_positions) >= 1 && last(kz_positions) <= nkz "kz centering places samples outside the encoded matrix; check enc_lim_kspace_encoding_step_2"
        kz_sub = _positions_to_subsampling(nkz, kz_positions)
    end

    # Batch dimensions, named per docs/src/high-level/nameddims.md ("Multi-slice data should
    # include :z as the first batch dimension"). Only counters that actually vary contribute a
    # dimension.
    batch_candidates = (
        (:z, p -> Int(p.head.idx.slice)),
        (:contrast, p -> Int(p.head.idx.contrast)),
        (:time, p -> Int(p.head.idx.phase)),
        (:repetition, p -> Int(p.head.idx.repetition)),
        (:set, p -> Int(p.head.idx.set)),
        (:average, p -> Int(p.head.idx.average)),
    )
    batch_names = Symbol[]
    batch_maps = Dict{Int, Int}[]
    batch_getters = Function[]
    batch_sizes = Int[]
    for (name, getter) in batch_candidates
        ids = [getter(p) for p in profiles]
        if length(unique(ids)) > 1
            @argcheck !is3D || name !== :z "3D k-space with more than one slice (multi-slab 3D) is not supported; filter raw.profiles and construct CartesianAcquisitionInfo slab-by-slab"
            map, uids = _compact_index_map(ids)
            push!(batch_names, name)
            push!(batch_maps, map)
            push!(batch_getters, getter)
            push!(batch_sizes, length(uids))
        end
    end

    kx_count = length(rows)
    ky_count = length(ky_uids)
    fourier_size = is3D ? (kx_count, ky_count, length(kz_uids)) : (kx_count, ky_count)
    ksp = zeros(T, fourier_size..., ncoil, batch_sizes...)

    for p in profiles
        ky_idx = ky_map[Int(p.head.idx.kspace_encode_step_1)]
        batch_idx = ntuple(j -> batch_maps[j][batch_getters[j](p)], length(batch_names))
        data = @view p.data[rows, :]
        if is3D
            kz_idx = kz_map[Int(p.head.idx.kspace_encode_step_2)]
            @views ksp[:, ky_idx, kz_idx, :, batch_idx...] .= data
        else
            @views ksp[:, ky_idx, :, batch_idx...] .= data
        end
    end

    fourier_names = is3D ? (:kx, :ky, :kz) : (:kx, :ky)
    dimnames_full = (fourier_names..., :coil, batch_names...)
    kspace_data = NamedDimsArray{dimnames_full}(ksp)

    subsampling = is3D ? (kx_sub, ky_sub, kz_sub) : (kx_sub, ky_sub)
    if all(s -> s isa Colon, subsampling)
        subsampling = nothing
    end

    image_size = is3D ? (nkx, nky, nkz) : (nkx, nky)
    return CartesianAcquisitionInfo(
        kspace_data;
        is3D,
        image_size,
        header = _raw_header(raw, profiles, image_size),
        sensitivity_maps,
        subsampling,
        # Scanner data images an object centred in the FOV, unlike Ristretto's plain-DFT default of the
        # image origin at index 1 — see "FFT-shift convention" in the docstring above.
        shifted_image_dims = is3D ? (:x, :y, :z) : (:x, :y),
    )
end

# Non-Cartesian data is assembled from the profiles directly rather than through
# `MRIBase.trajectory`/`MRIBase.rawdata`. Those two disagree for real files: `trajectory` lays
# profiles out by `(kspace_encode_step_1, kspace_encode_step_2, slice, repetition)` and then keeps
# only the first slice and repetition, while `rawdata` selects profiles by `repetition = 1`. An
# exporter that numbers every profile with its own repetition index — USC Speech's spiral files do
# — therefore gets a trajectory that is all but one interleaf of zeros, and a k-space of a single
# profile, which do not even have matching sizes.
#
# The layout produced here is `(:sample, :readout, :coil, batch...)` for k-space, with the
# trajectory `(:coord, :sample, :readout)` and, when the file carries one, a density compensation
# array `(:sample, :readout)`. `:readout` indexes the profiles of one slab in acquisition order —
# the spiral interleaves, radial spokes, EPI shots. A dynamic series whose frames are *not*
# separated by one of the ISMRMRD counters (again USC Speech, which increments `repetition` per
# profile rather than per frame) therefore arrives as one long readout axis: split it into frames
# yourself if you want one image per frame.
function _noncartesian_acquisition_info(raw::RawAcquisitionData; sensitivity_maps = nothing)
    profiles = _image_profiles(raw)

    enc = Int.(raw.params["encodedSize"])
    lim2 = get(raw.params, "enc_lim_kspace_encoding_step_2", Limit(0, 0, 0))::Limit
    is3D = lim2.maximum > lim2.minimum
    D = is3D ? 3 : 2
    image_size = is3D ? (enc[1], enc[2], enc[3]) : (enc[1], enc[2])

    p1 = first(profiles)
    pre = Int(p1.head.discard_pre)
    post = Int(p1.head.discard_post)
    nsamples = size(p1.data, 1)
    ncoil = Int(p1.head.active_channels)
    traj_dims = size(p1.traj, 1)
    @argcheck traj_dims >= D "profiles carry $(traj_dims) trajectory rows, fewer than the $(D) encoding dimensions"
    for p in profiles
        @argcheck size(p.data, 1) == nsamples "profiles have differing readout lengths; assemble manually"
        @argcheck Int(p.head.discard_pre) == pre && Int(p.head.discard_post) == post "profiles have differing discard_pre/discard_post; assemble manually"
        @argcheck Int(p.head.active_channels) == ncoil "profiles have differing active_channels; assemble manually"
        @argcheck size(p.traj, 1) == traj_dims "profiles have differing trajectory dimensions; assemble manually"
    end
    rows = (pre + 1):(nsamples - post)

    # A trajectory row beyond the encoding dimensions is the vendor's density compensation
    # weighting (this is what USC Speech's third row is: 0 at the centre of k-space, 1 at the
    # edge), not a kz coordinate — a 2D acquisition has no kz to store.
    dcf_row = traj_dims > D ? D + 1 : nothing

    # Only counters that vary *and* are not simply a per-profile running index become batch
    # dimensions: an exporter that numbers every profile with its own repetition index is
    # counting profiles, not repetitions, and turning that into a batch dimension would give one
    # readout per "frame".
    batch_candidates = (
        (:z, p -> Int(p.head.idx.slice)),
        (:contrast, p -> Int(p.head.idx.contrast)),
        (:time, p -> Int(p.head.idx.phase)),
        (:repetition, p -> Int(p.head.idx.repetition)),
        (:set, p -> Int(p.head.idx.set)),
        (:average, p -> Int(p.head.idx.average)),
    )
    batch_names = Symbol[]
    batch_maps = Dict{Int, Int}[]
    batch_getters = Function[]
    batch_sizes = Int[]
    for (name, getter) in batch_candidates
        ids = [getter(p) for p in profiles]
        nuniq = length(unique(ids))
        (nuniq == 1 || nuniq == length(profiles)) && continue
        map, uids = _compact_index_map(ids)
        push!(batch_names, name)
        push!(batch_maps, map)
        push!(batch_getters, getter)
        push!(batch_sizes, length(uids))
    end

    nbatch = isempty(batch_sizes) ? 1 : prod(batch_sizes)
    nreadout, r = divrem(length(profiles), nbatch)
    @argcheck r == 0 "the $(length(profiles)) profiles do not split evenly over the batch dimensions $(batch_names); assemble manually"

    T = eltype(p1.data)
    R = real(T)
    nsamp = length(rows)
    ksp = zeros(T, nsamp, nreadout, ncoil, batch_sizes...)
    traj = zeros(R, D, nsamp, nreadout)
    dcf = dcf_row === nothing ? nothing : zeros(R, nsamp, nreadout)

    counters = zeros(Int, batch_sizes...)
    for p in profiles
        batch_idx = ntuple(j -> batch_maps[j][batch_getters[j](p)], length(batch_names))
        counters[batch_idx...] += 1
        i = counters[batch_idx...]
        @views ksp[:, i, :, batch_idx...] .= p.data[rows, :]
        # The trajectory of a given readout is the same in every batch, so the first batch to
        # reach readout `i` writes it and the rest agree.
        @views traj[:, :, i] .= p.traj[1:D, rows]
        dcf === nothing || (@views dcf[:, i] .= p.traj[dcf_row, rows])
    end

    ksp_names = (:sample, :readout, :coil, batch_names...)
    kspace_data = NamedDimsArray{ksp_names}(ksp)
    trajectory = NamedDimsArray{(:coord, :sample, :readout)}(traj)
    dcf_named = dcf === nothing ? nothing : NamedDimsArray{(:sample, :readout)}(dcf)

    return NonCartesianAcquisitionInfo(
        kspace_data;
        trajectory,
        dcf = dcf_named,
        sensitivity_maps,
        image_size,
        header = _raw_header(raw, profiles, image_size),
    )
end
