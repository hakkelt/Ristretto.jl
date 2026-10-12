@testitem "add_noise on plain arrays and NamedDimsArrays" tags = [:simulation] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra
    using Statistics: std
    using Random

    Random.seed!(0)

    @testset "snr_db on a plain array" begin
        k = 5.0f0 .* ones(ComplexF32, 32, 32)
        noisy = add_noise(k; snr_db = 20)
        @test size(noisy) == size(k)
        @test eltype(noisy) == ComplexF32
        @test noisy != k
        # Empirical SNR should be roughly in the right ballpark.
        measured_snr_db = 20 * log10(norm(k) / norm(noisy - k))
        @test isapprox(measured_snr_db, 20; atol = 3)
    end

    @testset "noise_std on a plain array" begin
        k = zeros(ComplexF32, 64, 64)
        noisy = add_noise(k; noise_std = 0.1)
        measured_std = std(vec(noisy); corrected = false)
        @test isapprox(measured_std, 0.1; atol = 0.02)
    end

    @testset "NamedDimsArray preserves dimension names" begin
        k = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 16, 16, 4))
        noisy = add_noise(k; snr_db = 15)
        @test noisy isa NamedDimsArray
        @test dimnames(noisy) == (:kx, :ky, :coil)
        @test unname(noisy) != unname(k)
    end

    @testset "exactly one of snr_db / noise_std required" begin
        k = rand(ComplexF32, 8, 8)
        @test_throws ArgumentError add_noise(k)
        @test_throws ArgumentError add_noise(k; snr_db = 10, noise_std = 0.1)
    end

    @testset "reproducible with an explicit rng" begin
        k = rand(ComplexF32, 16, 16)
        a = add_noise(k; snr_db = 20, rng = MersenneTwister(42))
        b = add_noise(k; snr_db = 20, rng = MersenneTwister(42))
        @test a == b
    end
end

@testitem "estimate_snr and the image-domain `snr` keyword" tags = [:simulation, :analysis] begin
    using Ristretto
    using NamedDims
    using Random
    using Statistics: std

    Random.seed!(0)

    # A disc of signal on an empty background, wide enough to contain the centred signal box and
    # far enough from the corners to leave them signal-free.
    n = 128
    image = ComplexF32[
        (i - n ÷ 2)^2 + (j - n ÷ 2)^2 < (n ÷ 3)^2 ? 1 : 0 for i in 1:n, j in 1:n
    ]

    @testset "add_noise(; snr) round-trips through estimate_snr" begin
        # Fixed boxes measure the noise where there is no signal, so there is no threshold to clip
        # the background tail and the round trip holds at low SNR too.
        for target in (5, 10, 20, 40, 100)
            noisy = add_noise(image; snr = target, rng = MersenneTwister(1))
            @test isapprox(estimate_snr(noisy), target; rtol = 0.1)
        end
    end

    @testset "the box sizes are in voxels and the corners can be chosen" begin
        noisy = add_noise(image; snr = 20, rng = MersenneTwister(4))
        @test isapprox(estimate_snr(noisy; signal_box = 24, noise_box = 12), 20; rtol = 0.15)
        @test isapprox(estimate_snr(noisy; signal_box = (24, 16)), 20; rtol = 0.15)
        # One corner holds a quarter of the noise samples and still measures the same noise.
        @test isapprox(estimate_snr(noisy; corners = 1), estimate_snr(noisy); rtol = 0.15)
        @test isapprox(estimate_snr(noisy; corners = (1, 4)), estimate_snr(noisy); rtol = 0.15)
    end

    @testset "a noiseless image has no background noise" begin
        @test estimate_snr(image) == Inf
    end

    @testset "NamedDimsArray input" begin
        noisy = add_noise(NamedDimsArray{(:x, :y)}(image); snr = 50, rng = MersenneTwister(2))
        @test noisy isa NamedDimsArray
        @test isapprox(estimate_snr(noisy), 50; rtol = 0.1)
    end

    @testset "snr_masks is the pair of regions estimate_snr measures over" begin
        noisy = add_noise(image; snr = 50, rng = MersenneTwister(3))
        sig, noise = snr_masks(noisy)
        # A centred box of n ÷ 8 a side, and four corner boxes of the same size.
        @test count(sig) == (n ÷ 8)^2
        @test count(noise) == 4 * (n ÷ 8)^2
        @test !any(sig .& noise)
        # The signal box lands on the disc and the corner boxes land off it.
        @test all(!iszero, image[sig])
        @test all(iszero, image[noise])
        # Measuring by hand over those masks reproduces `estimate_snr`.
        mag = abs.(noisy)
        by_hand = sqrt(2 - π / 2) * (sum(mag[sig]) / count(sig)) / std(mag[noise])
        # Float32 magnitudes, summed in a different order.
        @test by_hand ≈ estimate_snr(noisy) rtol = 1.0e-6
    end

    @testset "argument checking" begin
        @test_throws ArgumentError add_noise(image; snr = 20, noise_std = 0.1)
        @test_throws ArgumentError add_noise(image; snr = -1)
        # A box that does not fit twice along a dimension would meet its opposite number.
        @test_throws ArgumentError estimate_snr(image; noise_box = 100)
        # A signal box that reaches into the corners is not a signal box.
        @test_throws ArgumentError estimate_snr(image; signal_box = 120, noise_box = 16)
        @test_throws ArgumentError estimate_snr(image; corners = 5)
        @test_throws ArgumentError estimate_snr(image; signal_box = (8, 8, 8))
        # `snr` is image-domain, so it is rejected on an acquisition's k-space.
        acq = AcquisitionInfo(; is3D = false, image_size = (16, 16))
        data = simulate_acquisition(image[1:16, 1:16], acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        @test_throws ArgumentError add_noise(data; snr = 20)
    end
end

@testitem "add_noise on AcquisitionInfo (Cartesian and non-Cartesian)" tags = [:simulation, :acquisition, :nfft] begin
    using Ristretto: CartesianAcquisitionInfo
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using Random

    Random.seed!(1)

    @testset "CartesianAcquisitionInfo: returns a copy, original untouched" begin
        ksp = rand(ComplexF32, 32, 32)
        acq = AcquisitionInfo(ksp; is3D = false)
        noisy_acq = add_noise(acq; snr_db = 25)
        @test noisy_acq isa CartesianAcquisitionInfo
        @test noisy_acq.kspace_data != acq.kspace_data
        @test acq.kspace_data == ksp # original untouched
    end

    @testset "NonCartesianAcquisitionInfo: returns a copy" begin
        traj = radial_trajectory(24, 8)
        ksp = rand(ComplexF32, 24, 8)
        acq = NonCartesianAcquisitionInfo(ksp; trajectory = traj, image_size = (16, 16))
        noisy_acq = add_noise(acq; noise_std = 0.01)
        @test noisy_acq isa NonCartesianAcquisitionInfo
        @test noisy_acq.kspace_data != acq.kspace_data
        @test acq.kspace_data == ksp
        @test noisy_acq.trajectory === acq.trajectory
    end

    @testset "errors without k-space data" begin
        traj = radial_trajectory(16, 4)
        acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (16, 16))
        @test_throws ArgumentError add_noise(acq; snr_db = 20)
    end
end

@testitem "radial_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra

    @testset "shape, dimension names and k-space extent" begin
        traj = radial_trajectory(64, 32)
        @test size(traj) == (2, 64, 32)
        @test dimnames(traj) == (:coord, :sample, :spoke)
        @test all(x -> -0.5 <= x < 0.5, unname(traj))
    end

    @testset "orderings differ and stay within bounds" begin
        for ordering in (LinearOrdering(), GoldenAngle(), TinyGoldenAngle(), TinyGoldenAngle(3))
            traj = radial_trajectory(32, 16; ordering)
            @test size(traj) == (2, 32, 16)
            @test all(x -> -0.5 <= x < 0.5, unname(traj))
        end
        @test unname(radial_trajectory(8, 4; ordering = LinearOrdering())) != unname(radial_trajectory(8, 4; ordering = GoldenAngle()))
        # Index 1 of the tiny family *is* the standard golden angle.
        @test unname(radial_trajectory(8, 4; ordering = TinyGoldenAngle(1))) ≈ unname(radial_trajectory(8, 4; ordering = GoldenAngle()))
        # Consecutive tiny-golden-angle spokes stay closer together than golden-angle ones.
        tiny = unname(radial_trajectory(2, 2; ordering = TinyGoldenAngle(4)))
        golden = unname(radial_trajectory(2, 2; ordering = GoldenAngle()))
        @test norm(tiny[:, :, 2] - tiny[:, :, 1]) < norm(golden[:, :, 2] - golden[:, :, 1])
    end

    @testset "an ordering is a type, not a symbol" begin
        @test_throws TypeError radial_trajectory(8, 4; ordering = :golden_angle)
        @test_throws ArgumentError TinyGoldenAngle(0)
    end

    @testset "center-out half spokes" begin
        traj = unname(radial_trajectory(16, 8; center_out = true))
        @test size(traj) == (2, 16, 8)
        @test all(iszero, traj[:, 1, :])
        radii = [norm(traj[:, i, s]) for i in 1:16, s in 1:8]
        @test all(r -> r ≈ 0.5 * 15 / 16, radii[end, :])
        @test all(diff(radii; dims = 1) .> 0)
        # Linear half spokes cover the full circle: spoke s points at 2π(s - 1)/8.
        lin = unname(radial_trajectory(4, 8; ordering = LinearOrdering(), center_out = true))
        @test atan(lin[2, end, 5], lin[1, end, 5]) ≈ π
        # The golden angle for rays is 2π/φ, double the one for lines.
        ga = unname(radial_trajectory(4, 2; center_out = true))
        @test mod(atan(ga[2, end, 2], ga[1, end, 2]), 2π) ≈ 2π * (sqrt(5) - 1) / 2 atol = 1.0e-5
        tiny = unname(radial_trajectory(4, 2; ordering = TinyGoldenAngle(3), center_out = true))
        @test mod(atan(tiny[2, end, 2], tiny[1, end, 2]), 2π) ≈ 2π / ((1 + sqrt(5)) / 2 + 2) atol = 1.0e-5
        sos = unname(stack_of_stars_trajectory(8, 4, 2; center_out = true))
        @test all(iszero, sos[1:2, 1, :, :])
    end

    @testset "simulate_acquisition + direct NFFT reconstruction is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny = 48, 48
        img = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = radial_trajectory(72, 150; ordering = GoldenAngle())
        acq = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny))
        data = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        @test size(data.kspace_data) == (72, 150)
        acq_dcf = density_compensation(data; method = PipeMenonDCF(maxit = 15))
        rec = reconstruct(acq_dcf; verbosity = Silent())

        a, r = abs.(rec), abs.(img)
        α = sum(a .* r) / sum(abs2, a)
        @test norm(α .* a .- r) / norm(r) < 0.35
    end
end

@testitem "stack_of_stars_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims

    @testset "shape, dimension names and partition grid" begin
        traj = stack_of_stars_trajectory(48, 24, 6)
        @test size(traj) == (3, 48, 24, 6)
        @test dimnames(traj) == (:coord, :sample, :spoke, :partition)
        raw = unname(traj)
        @test all(x -> -0.5 <= x < 0.5, raw)
        # in-plane pattern is identical across partitions
        @test raw[1:2, :, :, 1] == raw[1:2, :, :, end]
        # partition (kz) coordinate varies across partitions and is constant within one
        @test length(unique(raw[3, 1, 1, :])) == 6
        @test all(==(raw[3, 1, 1, 3]), raw[3, :, :, 3])
    end
end

@testitem "kooshball_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra

    @testset "shape, dimension names and k-space extent" begin
        traj = kooshball_trajectory(32, 500)
        @test size(traj) == (3, 32, 500)
        @test dimnames(traj) == (:coord, :sample, :spoke)
        raw = unname(traj)
        @test all(x -> -0.5 <= x < 0.5, raw)
        # spoke directions should cover the sphere roughly isotropically: the mean direction of
        # the outermost sample over many spokes should be close to zero.
        outer = raw[:, end, :]
        dirs = outer ./ mapslices(norm, outer; dims = 1)
        @test norm(sum(dirs; dims = 2)) / size(dirs, 2) < 0.15
    end

    @testset "half spokes" begin
        full = unname(kooshball_trajectory(8, 50))
        half = unname(kooshball_trajectory(8, 50; center_out = true))
        @test all(iszero, half[:, 1, :])
        # Same directions as the full spokes, from the center outwards.
        dir(t) = t[:, end, :] ./ mapslices(norm, t[:, end, :]; dims = 1)
        @test dir(half) ≈ dir(full)
        @test all(x -> -0.5 <= x < 0.5, half)
    end

    @testset "simulate_acquisition + direct NFFT reconstruction of a 3D phantom is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny, nz = 24, 24, 24
        img = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = kooshball_trajectory(32, 2000)
        acq = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny, nz))
        data = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        @test size(data.kspace_data) == (32, 2000)
        acq_dcf = density_compensation(data; method = PipeMenonDCF(maxit = 10))
        rec = reconstruct(acq_dcf; verbosity = Silent())

        a, r = abs.(rec), abs.(img)
        α = sum(a .* r) / sum(abs2, a)
        # A coarse gridded (non-iterative) reconstruction of a heavily undersampled 3D kooshball
        # is not expected to be very accurate; this only checks it is not garbage.
        @test norm(α .* a .- r) / norm(r) < 0.7
    end
end

@testitem "spiral_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra

    @testset "shape, dimension names and k-space extent" begin
        for variant in (Archimedean(), VariableDensity(), VariableDensity(0.5))
            traj = spiral_trajectory(128, 6; variant, nturns = 8)
            @test size(traj) == (2, 128, 6)
            @test dimnames(traj) == (:coord, :sample, :interleave)
            @test all(x -> -0.5 <= x < 0.5, unname(traj))
        end
    end

    @testset "variable density oversamples the center relative to archimedean" begin
        arch = unname(spiral_trajectory(64, 1; variant = Archimedean(), nturns = 8))
        vd = unname(spiral_trajectory(64, 1; variant = VariableDensity(2.0), nturns = 8))
        r_arch = sqrt.(arch[1, :, 1] .^ 2 .+ arch[2, :, 1] .^ 2)
        r_vd = sqrt.(vd[1, :, 1] .^ 2 .+ vd[2, :, 1] .^ 2)
        # A variable-density spiral with exponent > 1 grows its radius more slowly at the start
        # of the arm, packing more samples near the center than the Archimedean spiral.
        @test r_vd[8] < r_arch[8]
        # Exponent 1 is the Archimedean spiral.
        @test unname(spiral_trajectory(64, 1; variant = VariableDensity(1), nturns = 8)) ≈ arch
    end

    @testset "a variant is a type, not a symbol" begin
        @test_throws TypeError spiral_trajectory(16, 2; variant = :archimedean)
        @test_throws ArgumentError VariableDensity(0)
    end

    @testset "simulate_acquisition + direct NFFT reconstruction is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny = 48, 48
        img = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = spiral_trajectory(1024, 6; nturns = 12)
        acq = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny))
        data = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        acq_dcf = density_compensation(data; method = PipeMenonDCF(maxit = 15))
        rec = reconstruct(acq_dcf; verbosity = Silent())

        a, r = abs.(rec), abs.(img)
        α = sum(a .* r) / sum(abs2, a)
        @test norm(α .* a .- r) / norm(r) < 0.35
    end
end

@testitem "phyllotaxis_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra

    unit(t) = (o = unname(t)[:, end, :]; o ./ mapslices(norm, o; dims = 1))
    traj = phyllotaxis_trajectory(8, 610)
    @test size(traj) == (3, 8, 610)
    @test dimnames(traj) == (:coord, :sample, :spoke)
    @test all(x -> -0.5 <= x < 0.5, unname(traj))
    # Full-spoke directions fill the upper hemisphere; spoke 1 is the pole.
    phy = unit(traj)
    @test all(>=(-1.0e-6), phy[3, :])
    @test phy[:, 1] ≈ [0, 0, 1]
    # Consecutive spokes turn by the golden angle and descend monotonically towards the equator.
    @test all(diff(phy[3, :]) .< 0)
    # Interleaves reorder the same directions.
    phy21 = unit(phyllotaxis_trajectory(8, 610; interleaves = 21))
    @test sortslices(phy21; dims = 2) ≈ sortslices(phy; dims = 2)
    @test phy21[:, 2] ≈ phy[:, 22]
    @test_throws ArgumentError phyllotaxis_trajectory(8, 10; interleaves = 0)
    # Half spokes start on the center and spread over the whole sphere.
    half = phyllotaxis_trajectory(8, 1000; center_out = true)
    @test all(iszero, unname(half)[:, 1, :])
    d = unit(half)
    @test norm(sum(d; dims = 2)) / size(d, 2) < 0.01
    @test count(<(0), d[3, :]) == 500
    @test all(x -> -0.5 <= x < 0.5, unname(half))
end

@testitem "floret_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra

    traj = floret_trajectory(64, 20)
    @test size(traj) == (3, 64, 20, 3)
    @test dimnames(traj) == (:coord, :sample, :interleave, :hub)
    raw = unname(traj)
    @test all(x -> -0.5 <= x < 0.5, raw)
    @test all(iszero, raw[:, 1, :, :])
    # Radius grows as √t; every arm stays within 45° of the plane normal to its hub axis.
    @test norm(raw[:, 17, 1, 1]) ≈ 0.5 * sqrt(16 / 64) rtol = 1.0e-5
    for (h, axis) in enumerate((3, 1, 2))
        elevation = [asin(abs(raw[axis, end, j, h]) / norm(raw[:, end, j, h])) for j in 1:20]
        @test maximum(elevation) <= π / 4 + 1.0e-5
    end
    # The hubs are the same arms about orthogonal axes.
    @test raw[[2, 3, 1], :, :, 2] ≈ raw[:, :, :, 1]
    one_hub = unname(floret_trajectory(32, 50; nhubs = 1))
    @test size(one_hub, 4) == 1
    @test maximum(abs, one_hub[3, end, :]) > 0.45
    @test_throws ArgumentError floret_trajectory(32, 4; nhubs = 4)

    @testset "simulate_acquisition through a 4D trajectory layout" begin
        img = rand(ComplexF32, 12, 12, 12)
        acq = AcquisitionInfo(; trajectory = floret_trajectory(32, 30), image_size = (12, 12, 12))
        @test size(simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data) == (32, 30, 3)
    end
end

@testitem "sparkling_trajectory" tags = [:simulation, :nfft] begin
    using Ristretto
    using NamedDims
    using LinearAlgebra

    function constraints(k)
        steps = [norm(k[:, i + 1, s] - k[:, i, s]) for i in 1:(size(k, 2) - 1), s in axes(k, 3)]
        bends = [norm(k[:, i + 1, s] - 2k[:, i, s] + k[:, i - 1, s]) for i in 2:(size(k, 2) - 1), s in axes(k, 3)]
        return maximum(steps), maximum(bends)
    end
    # Radius-binned sample counts against those of the target density, both normalised.
    function radial_profile(k, edges)
        r = vec(sqrt.(sum(abs2, k; dims = 1)))
        return [count(x -> lo <= x < hi, r) for (lo, hi) in zip(edges[1:(end - 1)], edges[2:end])] ./ length(r)
    end

    @testset "2D: shape, constraints, start on the center" begin
        traj = sparkling_trajectory(128, 12; iterations = 40)
        @test size(traj) == (2, 128, 12)
        @test dimnames(traj) == (:coord, :sample, :shot)
        k = unname(traj)
        @test all(x -> -0.5 <= x < 0.5, k)
        @test all(iszero, k[:, 1, :])
        α = max(0.5 / 64, 2 * 0.5 / 128)
        step, bend = constraints(k)
        @test step <= α * (1 + 1.0e-5)
        @test bend <= α / 10 * (1 + 1.0e-4)
        @test sparkling_trajectory(128, 12; iterations = 40) == traj
        @test sparkling_trajectory(128, 12; iterations = 40, threaded = false) == traj
    end

    @testset "2D: iterations move the samples towards the target density" begin
        edges = range(0, 0.5; length = 6)
        before = radial_profile(unname(sparkling_trajectory(256, 16; iterations = 0)), edges)
        after = radial_profile(unname(sparkling_trajectory(256, 16; iterations = 60)), edges)
        # The target decays as |k|⁻², so the samples concentrate at the center.
        @test after[1] > before[1]
        @test after[end] < before[end]
    end

    @testset "3D and argument checks" begin
        traj = sparkling_trajectory(64, 8; ndims = 3, iterations = 5, grid_size = 16)
        @test size(traj) == (3, 64, 8)
        k = unname(traj)
        @test all(x -> -0.5 <= x < 0.5, k)
        @test all(iszero, k[:, 1, :])
        @test_throws ArgumentError sparkling_trajectory(64, 8; ndims = 4)
        @test_throws ArgumentError sparkling_trajectory(64, 8; max_step = 0.001)
    end
end

@testitem "AcquisitionInfo non-Cartesian construction needs no k-space placeholder" tags = [:acquisition, :nfft] begin
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo

    traj = radial_trajectory(16, 4)
    smaps = coil_sensitivities(16, 16, 3)

    # No positional k-space argument at all, and `kspace_data = nothing` explicitly: both work
    # without inventing a placeholder array.
    acq1 = AcquisitionInfo(; trajectory = traj, image_size = (16, 16), sensitivity_maps = smaps)
    @test acq1 isa NonCartesianAcquisitionInfo
    @test isnothing(acq1.kspace_data)

    acq2 = AcquisitionInfo(nothing; trajectory = traj, image_size = (16, 16))
    @test isnothing(acq2.kspace_data)
end

@testitem "subsampling: tuple of indexing expressions (default idiom)" tags = [:acquisition, :simulation] begin
    using Ristretto

    @testset "partial Fourier as a UnitRange" begin
        nx, ny = 32, 32
        pf_pattern = (:, 1:round(Int, 0.65 * ny))
        acq = AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), subsampling = pf_pattern)
        @test acq.subsampling === pf_pattern

        img = rand(ComplexF32, nx, ny)
        data = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        @test size(data.kspace_data) == (nx, round(Int, 0.65 * ny))
        rec = reconstruct(data, DirectReconstruction(); verbosity = Silent())
        @test size(rec) == (nx, ny)
    end

    @testset "GRAPPA-style uniform undersampling with an ACS block, as a StepRange ∪ UnitRange" begin
        nx, ny = 32, 32
        R, acs_half_width = 4, 3
        center = div(ny, 2)
        grappa_lines = sort(union(1:R:ny, (center - acs_half_width):(center + acs_half_width)))
        grappa_pattern = (:, grappa_lines)
        acq = AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), subsampling = grappa_pattern)

        img = rand(ComplexF32, nx, ny)
        data = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        @test size(data.kspace_data) == (nx, length(grappa_lines))
        rec = reconstruct(data, DirectReconstruction(); verbosity = Silent())
        @test size(rec) == (nx, ny)
    end
end
