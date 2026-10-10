@testmodule RawAcqHelpers begin
    # Loading MRIFiles loads the extension that builds an AcquisitionInfo from raw data.
    using MRIFiles: MRIFiles
    using MRIBase: RawAcquisitionData, Profile, AcquisitionHeader, EncodingCounters, Limit

    export make_profile, make_traj_profile, make_raw

    function make_profile(
            data::Matrix{ComplexF32}; step1 = 0, step2 = 0, slice = 0, contrast = 0, phase = 0,
            repetition = 0, set = 0, average = 0, discard_pre = 0, discard_post = 0, center_sample = 0,
            position = (0, 0, 0), read_dir = (0, 0, 0), phase_dir = (0, 0, 0), slice_dir = (0, 0, 0),
        )
        ncoil = size(data, 2)
        head = AcquisitionHeader(;
            position = Float32.(position), read_dir = Float32.(read_dir),
            phase_dir = Float32.(phase_dir), slice_dir = Float32.(slice_dir),
            number_of_samples = UInt16(size(data, 1)),
            available_channels = UInt16(ncoil),
            active_channels = UInt16(ncoil),
            discard_pre = UInt16(discard_pre),
            discard_post = UInt16(discard_post),
            center_sample = UInt16(center_sample),
            idx = EncodingCounters(;
                kspace_encode_step_1 = UInt16(step1),
                kspace_encode_step_2 = UInt16(step2),
                slice = UInt16(slice),
                contrast = UInt16(contrast),
                phase = UInt16(phase),
                repetition = UInt16(repetition),
                set = UInt16(set),
                average = UInt16(average),
            ),
        )
        return Profile(head, zeros(Float32, 1, 1), data)
    end

    # A minimal non-Cartesian ("custom" trajectory) profile: MRIBase's own `trajectory(f)` needs
    # a nonzero `sample_time_us` (it divides by it) and `trajectory_dimensions` set from `traj`.
    function make_traj_profile(
            data::Matrix{ComplexF32}, traj::Matrix{Float32}; slice = 0, contrast = 0, phase = 0,
            repetition = 0,
        )
        ncoil = size(data, 2)
        head = AcquisitionHeader(;
            number_of_samples = UInt16(size(data, 1)),
            available_channels = UInt16(ncoil),
            active_channels = UInt16(ncoil),
            trajectory_dimensions = UInt16(size(traj, 1)),
            sample_time_us = 5.0f0,
            idx = EncodingCounters(;
                slice = UInt16(slice), contrast = UInt16(contrast), phase = UInt16(phase),
                repetition = UInt16(repetition),
            ),
        )
        return Profile(head, traj, data)
    end

    function make_raw(profiles; encoded_size, lim1, lim2 = Limit(0, 0, 0), trajectory = "cartesian", params...)
        p = Dict{String, Any}(
            "encodedSize" => collect(encoded_size),
            "trajectory" => trajectory,
            "enc_lim_kspace_encoding_step_1" => lim1,
            "enc_lim_kspace_encoding_step_2" => lim2,
        )
        for (k, v) in params
            p[string(k)] = v
        end
        return RawAcquisitionData(p, profiles)
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — Cartesian" tags = [:extension, :acquisition] setup = [RawAcqHelpers] begin
    using Ristretto
    using MRIBase: Profile, Limit
    using Ristretto: CartesianAcquisitionInfo
    using NamedDims: dimnames, unname

    @testset "asymmetric readout + partial-Fourier ky + multi-slice (:z)" begin
        # encodedSize = (10, 8); readout: 8 acquired samples, center_sample=3 (0-based) is off
        # from the geometric center (5), so placing them must shift by `10÷2 - 3 = 2` — this is
        # exactly the FFT-shift bug the constructor exists to avoid (see its docstring).
        # ky: only steps 0:5 of 0:7 are present (partial Fourier), and encoding_limits.center=2
        # is likewise off from the geometric center (4).
        ncoil = 2
        profiles = Profile[]
        for slice in (0, 1), step1 in 0:5
            data = ComplexF32[1000 * slice + 10 * step1 + i + 100im * c for i in 1:8, c in 1:ncoil]
            push!(profiles, make_profile(data; step1, slice, center_sample = 3))
        end
        raw = make_raw(profiles; encoded_size = (10, 8, 1), lim1 = Limit(0, 7, 2))

        info = AcquisitionInfo(raw)
        @test info isa CartesianAcquisitionInfo
        @test info.is3D == false
        @test info.image_size == (10, 8)
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :z)
        @test size(info.kspace_data) == (8, 6, ncoil, 2)
        @test info.subsampling == (3:10, 3:8)

        ksp = unname(info.kspace_data)
        @test ksp[:, 1, :, 1] == ComplexF32[i + 100im * c for i in 1:8, c in 1:ncoil] # slice=0, step1=0
        @test ksp[:, 6, :, 2] == ComplexF32[1000 + 50 + i + 100im * c for i in 1:8, c in 1:ncoil] # slice=1, step1=5
    end

    @testset "fully sampled -> Colon()/nothing subsampling, single slice" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, center_sample = 2)
                for step1 in 0:3
        ]
        raw = make_raw(profiles; encoded_size = (4, 4, 1), lim1 = Limit(0, 3, 2))

        info = AcquisitionInfo(raw)
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil)
        @test size(info.kspace_data) == (4, 4, 1)
        @test isnothing(info.subsampling)
        @test info.image_size == (4, 4)
        @test info.shifted_image_dims == (:x, :y)
    end

    @testset "an unrecorded center_sample falls back to a symmetric readout" begin
        # `center_sample = 0` is what several exporters write when they record no echo position
        # (mridata.org's GE files among them). Taken literally it places the readout outside the
        # encoded matrix; the constructor then assumes the echo sits mid-readout instead.
        profiles = Profile[
            make_profile(ComplexF32[i + 0im for i in 1:4, c in 1:1]; step1, center_sample = 0)
                for step1 in 0:3
        ]
        raw = make_raw(profiles; encoded_size = (4, 4, 1), lim1 = Limit(0, 3, 2))

        info = @test_logs (:warn, r"records no echo position") AcquisitionInfo(raw)
        @test size(info.kspace_data) == (4, 4, 1)
        @test isnothing(info.subsampling)

        # A `center_sample = 0` that *is* consistent with the encoded matrix is left alone: here
        # the four acquired samples sit at the start of an 8-sample readout axis.
        edge = Profile[
            make_profile(ComplexF32[i + 0im for i in 1:4, c in 1:1]; step1, center_sample = 0)
                for step1 in 0:3
        ]
        raw_edge = make_raw(edge; encoded_size = (8, 4, 1), lim1 = Limit(0, 3, 2))
        info_edge = AcquisitionInfo(raw_edge)
        @test info_edge.subsampling[1] == 5:8
    end

    @testset "3D sets shifted_image_dims on all three spatial axes" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, step2, center_sample = 2)
                for step1 in 0:3, step2 in 0:2
        ]
        raw = make_raw(
            vec(profiles); encoded_size = (4, 4, 3), lim1 = Limit(0, 3, 2), lim2 = Limit(0, 2, 1)
        )
        @test AcquisitionInfo(raw).shifted_image_dims == (:x, :y, :z)
    end

    @testset "irregular undersampled ky -> boolean mask" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, center_sample = 2)
                for step1 in (0, 2, 4, 6)
        ]
        raw = make_raw(profiles; encoded_size = (4, 8, 1), lim1 = Limit(0, 7, 4))

        info = AcquisitionInfo(raw)
        @test size(info.kspace_data) == (4, 4, 1)
        @test info.subsampling[1] == Colon()
        @test info.subsampling[2] == Bool[1, 0, 1, 0, 1, 0, 1, 0]
    end

    @testset "cardiac phase becomes the :time batch dimension" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, phase, center_sample = 2)
                for step1 in 0:3, phase in 0:2
        ]
        raw = make_raw(vec(profiles); encoded_size = (4, 4, 1), lim1 = Limit(0, 3, 2))

        info = AcquisitionInfo(raw)
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :time)
        @test size(info.kspace_data) == (4, 4, 1, 3)
    end

    @testset "3D acquisition (kspace_encode_step_2 varies)" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, step2, center_sample = 2)
                for step1 in 0:3, step2 in 0:2
        ]
        raw = make_raw(
            vec(profiles); encoded_size = (4, 4, 3), lim1 = Limit(0, 3, 2), lim2 = Limit(0, 2, 1)
        )

        info = AcquisitionInfo(raw)
        @test info.is3D == true
        @test dimnames(info.kspace_data) == (:kx, :ky, :kz, :coil)
        @test size(info.kspace_data) == (4, 4, 3, 1)
        @test info.image_size == (4, 4, 3)
    end

    @testset "multi-slab 3D (is3D and slice both vary) is rejected, not silently dropped" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, step2, slice, center_sample = 2)
                for step1 in 0:3, step2 in 0:2, slice in 0:1
        ]
        raw = make_raw(
            vec(profiles); encoded_size = (4, 4, 3), lim1 = Limit(0, 3, 2), lim2 = Limit(0, 2, 1)
        )
        @test_throws ArgumentError AcquisitionInfo(raw)
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — header" tags = [:extension, :acquisition] setup = [RawAcqHelpers] begin
    using Test
    using Ristretto
    using Ristretto: header
    using MRIBase: Profile, Limit

    # Two slices of a coronal-ish acquisition: read along -x, phase along z, slices along y (LPS),
    # 6 mm apart; `position` is the centre of each slice.
    read_dir, phase_dir, slice_dir = (-1, 0, 0), (0, 0, 1), (0, 1, 0)
    profiles = Profile[
        make_profile(
            ones(ComplexF32, 8, 1); step1, slice, center_sample = 4,
            position = (10, 20 + 6 * slice, 30), read_dir, phase_dir, slice_dir,
        )
            for slice in (0, 1) for step1 in 0:7
    ]
    raw = make_raw(
        profiles; encoded_size = (8, 8, 1), lim1 = Limit(0, 7, 4),
        encodedFOV = [160.0, 80.0, 3.0], TE = [4.5], TR = 300.0, flipAngle_deg = 15.0,
        H1resonanceFrequency_Hz = 127_731_000,
    )
    h = header(AcquisitionInfo(raw))
    @test h.fov == (160.0, 80.0)
    @test h.spacing == (20.0, 10.0)
    @test h.slice_thickness == 3.0
    @test h.TE == 4.5 && h.TR == 300.0 && h.flip_angle == 15.0
    @test h.field_strength ≈ 3.0 atol = 1.0e-3
    @test h.orientation == [-1.0 0 0; 0 0 1; 0 1 0]
    @test h.slice_spacing == 6.0
    # The centre voxel (index n ÷ 2 + 1 = 5) of the first slice is at its `position`.
    @test collect(h.offset) + h.orientation * ([4, 4, 0] .* [20.0, 10.0, 0]) ≈ [10, 20, 30]

    # Without direction cosines the geometry is left unset.
    raw0 = make_raw(
        [make_profile(ones(ComplexF32, 4, 1); step1, center_sample = 2) for step1 in 0:3];
        encoded_size = (4, 4, 1), lim1 = Limit(0, 3, 2),
    )
    h0 = header(AcquisitionInfo(raw0))
    @test isnothing(h0.orientation) && isnothing(h0.offset) && isnothing(h0.fov)
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — object stays centred in the FOV" tags = [:extension, :acquisition, :reconstruction] setup = [RawAcqHelpers] begin
    using Ristretto
    using MRIBase: Profile, Limit
    using NamedDims: unname
    using FFTW: fft, fftshift, ifftshift

    # A scanner images an object centred in the FOV and stores k-space with DC at the centre.
    # Ristretto's plain-DFT default puts the image origin at index 1, so without `shifted_image_dims`
    # the reconstruction of such data comes out rolled by half the FOV along every spatial axis
    # (the whole object lands in the four corners). The constructor must set it for us.
    n = 16
    img = zeros(ComplexF32, n, n)
    img[6:9, 7:10] .= 1              # an off-centre blob, so a half-FOV roll is unambiguous
    img[7, 8] = 3
    # The scanner's DFT runs over CENTRED coordinates on both sides: k and x both range over
    # -n÷2 : n÷2-1. That is `fftshift ∘ fft ∘ ifftshift`, not a bare `fft` — a bare `fft` would
    # treat array index 1 as the spatial origin, which is Ristretto's own (unshifted) default.
    ksp_true = fftshift(fft(ifftshift(img)))    # DC at index n ÷ 2 + 1 = 9, as ISMRMRD stores it

    profiles = Profile[
        make_profile(ComplexF32.(reshape(ksp_true[:, j], n, 1)); step1 = j - 1, center_sample = n ÷ 2)
            for j in 1:n
    ]
    raw = make_raw(profiles; encoded_size = (n, n, 1), lim1 = Limit(0, n - 1, n ÷ 2))

    info = AcquisitionInfo(raw)
    @test info.shifted_image_dims == (:x, :y)

    rec = abs.(unname(reconstruct(info; verbosity = Silent()))[:, :, 1])
    @test Tuple(argmax(rec)) == (7, 8)                       # not (15, 16), the half-FOV-rolled peak
    @test rec ≈ (rec[7, 8] / 3) .* abs.(img)                 # whole image, not just the peak
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — non-Cartesian dispatch" tags = [:extension, :acquisition, :nfft] setup = [RawAcqHelpers] begin
    using Ristretto
    using MRIBase: Profile, Limit
    using Ristretto: NonCartesianAcquisitionInfo
    using NamedDims: dimnames, unname

    nsamp, ncoil = 5, 1
    profiles = Profile[]
    for k in 0:3
        traj = Float32.(hcat(range(-0.4f0, 0.4f0; length = nsamp), fill(Float32(k) / 10 - 0.2f0, nsamp))')
        data = ComplexF32.(reshape(1:nsamp, nsamp, 1) .+ 0im)
        push!(profiles, make_traj_profile(data, traj))
    end
    raw = make_raw(profiles; encoded_size = (8, 8, 1), lim1 = Limit(0, 0, 0), trajectory = "custom")

    info = AcquisitionInfo(raw)
    @test info isa NonCartesianAcquisitionInfo
    @test info.is3D == false
    @test info.image_size == (8, 8)
    @test dimnames(info.trajectory) == (:coord, :sample, :readout)
    @test size(info.trajectory) == (2, nsamp, length(profiles))
    @test dimnames(info.kspace_data) == (:sample, :readout, :coil)
    @test size(info.kspace_data) == (nsamp, length(profiles), ncoil)
    @test isnothing(info.dcf)
    # Every profile's samples land under its own readout index, in acquisition order.
    for (i, p) in enumerate(profiles)
        @test unname(info.kspace_data)[:, i, :] == p.data
        @test unname(info.trajectory)[:, :, i] == p.traj
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — non-Cartesian density compensation and batches" tags = [:extension, :acquisition, :nfft] setup = [RawAcqHelpers] begin
    using Ristretto
    using MRIBase: Profile, Limit
    using Ristretto: NonCartesianAcquisitionInfo
    using NamedDims: dimnames, unname

    nsamp, ncoil, ninterleaf, nframe = 5, 2, 3, 4

    # A profile whose trajectory carries a third row on a 2D acquisition: that row is the vendor's
    # density compensation weighting, not a kz coordinate.
    function spiral_profiles(; frame_counter)
        profiles = Profile[]
        for frame in 0:(nframe - 1), k in 0:(ninterleaf - 1)
            angle = 2π * k / ninterleaf
            r = range(0.0f0, 0.45f0; length = nsamp)
            traj = Float32.(vcat((r .* cos(angle))', (r .* sin(angle))', collect(r)'))
            data = ComplexF32.(fill(frame * ninterleaf + k + 1, nsamp, ncoil))
            # `repetition` counts profiles, as several real exporters do; `phase` counts frames.
            counters = frame_counter == :phase ?
                (; phase = frame, repetition = frame * ninterleaf + k) :
                (; repetition = frame * ninterleaf + k)
            push!(profiles, make_traj_profile(data, traj; counters...))
        end
        return profiles
    end

    @testset "the extra trajectory row becomes the dcf" begin
        raw = make_raw(
            spiral_profiles(; frame_counter = :none); encoded_size = (8, 8, 1),
            lim1 = Limit(0, ninterleaf - 1, 0), trajectory = "spiral",
        )
        info = AcquisitionInfo(raw)
        @test info isa NonCartesianAcquisitionInfo
        @test size(info.trajectory) == (2, nsamp, ninterleaf * nframe)
        @test dimnames(info.dcf) == (:sample, :readout)
        @test size(info.dcf) == (nsamp, ninterleaf * nframe)
        @test unname(info.dcf)[:, 1] == Float32.(range(0.0f0, 0.45f0; length = nsamp))
        # A counter that takes a different value in every profile is a profile counter, not a
        # batch dimension: all the profiles stay on one `:readout` axis.
        @test dimnames(info.kspace_data) == (:sample, :readout, :coil)
    end

    @testset "a counter that separates frames becomes a batch dimension" begin
        raw = make_raw(
            spiral_profiles(; frame_counter = :phase); encoded_size = (8, 8, 1),
            lim1 = Limit(0, ninterleaf - 1, 0), trajectory = "spiral",
        )
        info = AcquisitionInfo(raw)
        @test dimnames(info.kspace_data) == (:sample, :readout, :coil, :time)
        @test size(info.kspace_data) == (nsamp, ninterleaf, ncoil, nframe)
        @test size(info.trajectory) == (2, nsamp, ninterleaf)
        @test unname(info.kspace_data)[1, 1, 1, 3] == ComplexF32(2 * ninterleaf + 1)
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — real M4Raw data" tags = [:extension, :acquisition, :integration] begin
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo
    using NamedDims: dimnames, unname
    using MRITestData
    using MRIBase

    if MRITestData.get_download_path() === nothing
        MRITestData.set_download_path!(:cache)
    end
    entry = MRITestData.dataset(MRITestData.M4RAW, "multicoil_train/2022062402_T203"; offline = true)
    if !MRITestData.is_cached(entry)
        @test_skip "M4Raw sample not cached locally; not downloading it for this test"
    else
        raw = MRITestData.load_raw(entry)
        info = AcquisitionInfo(raw)
        @test info isa CartesianAcquisitionInfo
        @test info.is3D == false
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :z)
        @test size(info.kspace_data) == (256, 256, 4, 18)
        @test isnothing(info.subsampling) # M4Raw stores every encoded ky line (some are all-zero)
        @test info.shifted_image_dims == (:x, :y)

        # Must reproduce the notebook's hand-rolled single-slice assembly exactly (no extra
        # `fftshift` needed — see the constructor's docstring on the FFT-shift convention).
        function assemble_slice(raw, slice)
            profiles = [
                p for p in raw.profiles if
                    Int(p.head.idx.slice) == slice && Int(p.head.idx.contrast) == 0 &&
                    Int(p.head.idx.repetition) == 0 && Int(p.head.idx.average) == 0
            ]
            nsamples, ncoils = size(profiles[1].data)
            pre, post = Int(profiles[1].head.discard_pre), Int(profiles[1].head.discard_post)
            nkx = nsamples - pre - post
            ksp = zeros(ComplexF32, nkx, 256, ncoils)
            for p in profiles
                ksp[:, Int(p.head.idx.kspace_encode_step_1) + 1, :] .= ComplexF32.(p.data[(pre + 1):(pre + nkx), :])
            end
            return ksp
        end
        manual = assemble_slice(raw, 5)
        @test unname(info.kspace_data)[:, :, :, 6] == manual # slice id 5 -> compact index 6
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — real OCMR data (asymmetric-echo readout)" tags = [:extension, :acquisition, :integration] begin
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo
    using NamedDims: dimnames
    using MRITestData
    using MRIBase

    if MRITestData.get_download_path() === nothing
        MRITestData.set_download_path!(:cache)
    end
    entry = MRITestData.dataset(MRITestData.OCMR_SOURCE, "fs_0001_1_5T"; offline = true)
    if !MRITestData.is_cached(entry)
        @test_skip "OCMR sample not cached locally; not downloading it for this test"
    else
        raw = MRITestData.load_raw(entry)
        info = AcquisitionInfo(raw)
        @test info isa CartesianAcquisitionInfo
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :time)
        @test info.image_size == (512, 208)
        @test size(info.kspace_data) == (404, 208, 15, 19)
        # Asymmetric-echo readout: only a contiguous block of the 512-sample encoded readout was
        # acquired, recentered via `head.center_sample` (148) rather than the naive `row + 1`.
        @test info.subsampling[1] == 109:512
        @test info.subsampling[2] == Colon()
    end
end
