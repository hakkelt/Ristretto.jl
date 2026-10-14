@testmodule DelaySim begin
    # Radial data of an analytic object sampled where a gradient delay really puts the spokes.
    # The object is a sum of off-centre Gaussian blobs (centres and widths in pixels), whose
    # spectrum is known in closed form, so the data contain no gridding or NFFT error.
    using LinearAlgebra
    using Random

    const BLOBS = ((1.0 + 0.3im, (3.0, -2.0), 2.0), (0.6 - 0.2im, (-4.0, 1.5), 1.5))

    blob_spectrum(k, coil) = sum(
        a * cis(0.4 * coil * j) * 2π * σ^2 * exp(-2π^2 * σ^2 * sum(abs2, k)) * cispi(-2 * dot(k, c))
            for (j, (a, c, σ)) in enumerate(BLOBS)
    )

    # Nominal spokes: `N` samples spaced `dr` apart, k = 0 at sample N ÷ 2 + 1 unless `offset`
    # (in samples) moves the grid.
    function spokes(angles, N; dr = 1 / N, offset = 0.0)
        u = ((1:N) .- (N ÷ 2 + 1) .+ offset) .* dr
        traj = zeros(2, N, length(angles))
        for (s, θ) in enumerate(angles)
            traj[1, :, s] = u .* cos(θ)
            traj[2, :, s] = u .* sin(θ)
        end
        return traj
    end

    # The actual trajectory: every spoke displaced by S n (S in trajectory units), or only by the
    # component of S n along the spoke.
    function delayed(traj, S; transverse = true)
        out = copy(traj)
        for s in axes(traj, 3)
            n = traj[:, end, s] - traj[:, 1, s]
            n ./= norm(n)
            d = transverse ? S * n : dot(S * n, n) * n
            out[:, :, s] .+= d
        end
        return out
    end

    function data(traj; ncoils = 1, noise = 0.0, seed = 1)
        ksp = [blob_spectrum(traj[:, m, s], c) for m in axes(traj, 2), s in axes(traj, 3), c in 1:ncoils]
        if noise > 0
            rng = Random.Xoshiro(seed)
            ksp .+= noise * maximum(abs, ksp) .* randn(rng, ComplexF64, size(ksp))
        end
        return ncoils == 1 ? dropdims(ksp; dims = 3) : ksp
    end

    golden(n) = [mod2pi(k * π / Base.MathConstants.golden) for k in 0:(n - 1)]
    tensor(d) = [d.dx d.dxy; d.dxy d.dy]
end

@testitem "Gradient delays: both estimators recover an anisotropic delay tensor" tags = [:preprocessing, :acquisition] setup = [DelaySim] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using LinearAlgebra
    using .DelaySim: spokes, delayed, data, golden, tensor

    # 2× oversampled readout of a 64² image: 128 samples, k = 0 at sample 65 (even length).
    N, n = 128, 64
    dr = 1 / N
    nominal = spokes(golden(201), N; dr)
    S = [1.5 0.4; 0.4 0.8] .* dr                   # samples → trajectory units
    actual = delayed(nominal, S)
    acq = NonCartesianAcquisitionInfo(data(actual); trajectory = nominal, image_size = (n, n))

    est = estimate_gradient_delays(acq; method = RING())
    @test keys(est) == (:dx, :dy, :dxy)
    @test maximum(abs, tensor(est) - S) < 0.01 * dr

    # OpposingSpokes neglects the sideways part of the displacement: exact without it, biased by
    # a fraction of a sample with it.
    acq_along = NonCartesianAcquisitionInfo(data(delayed(nominal, S; transverse = false)); trajectory = nominal, image_size = (n, n))
    @test maximum(abs, tensor(estimate_gradient_delays(acq_along; method = OpposingSpokes())) - S) < 0.01 * dr
    @test maximum(abs, tensor(estimate_gradient_delays(acq; method = OpposingSpokes())) - S) < 0.2 * dr

    # Correction moves the spokes onto the actual trajectory, including the part across the spoke.
    full = correct_gradient_delays(acq; method = RING())
    @test maximum(abs, unname(full.trajectory) - actual) < 0.01 * dr

    # No delay: both estimate zero.
    acq0 = NonCartesianAcquisitionInfo(data(nominal); trajectory = nominal, image_size = (n, n))
    for method in (OpposingSpokes(), RING())
        @test maximum(abs, tensor(estimate_gradient_delays(acq0; method))) < 0.01 * dr
    end
end

@testitem "Gradient delays: independent of where the readout puts k = 0" tags = [:preprocessing, :acquisition] setup = [DelaySim] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using .DelaySim: spokes, delayed, data, golden, tensor

    # A symmetric grid `range(-0.45, 0.45; length = 64)` has k = 0 between two samples; a shorter
    # than ±0.5 readout changes the sample spacing. The estimate is relative to the nominal
    # trajectory, whatever its grid.
    N = 64
    r = range(-0.45, 0.45; length = N)
    dr = step(r)
    nominal = spokes(golden(150), N; dr, offset = 0.5)
    @test nominal[:, :, 1] ≈ [r' .* cos(0); r' .* sin(0)]
    S = [1.0 -0.3; -0.3 2.0] .* dr
    for method in (OpposingSpokes(), RING())
        acq = NonCartesianAcquisitionInfo(
            data(delayed(nominal, S; transverse = method isa RING)); trajectory = nominal, image_size = (64, 64)
        )
        @test maximum(abs, tensor(estimate_gradient_delays(acq; method)) - S) < 0.01 * dr
    end
end

@testitem "Gradient delays: 180° orderings need RING" tags = [:preprocessing, :acquisition] setup = [DelaySim] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using .DelaySim: spokes, delayed, data, tensor

    N = 128
    dr = 1 / N
    nominal = spokes(range(0, π; length = 97)[1:96], N; dr)
    S = [-0.7 0.5; 0.5 1.2] .* dr
    acq = NonCartesianAcquisitionInfo(data(delayed(nominal, S)); trajectory = nominal, image_size = (64, 64))
    @test_throws ArgumentError estimate_gradient_delays(acq; method = OpposingSpokes())
    @test maximum(abs, tensor(estimate_gradient_delays(acq; method = RING())) - S) < 0.01 * dr
    # RING works from a handful of spokes, as the paper states.
    few = NonCartesianAcquisitionInfo(
        data(delayed(nominal[:, :, 1:12:end], S)); trajectory = nominal[:, :, 1:12:end], image_size = (64, 64)
    )
    @test maximum(abs, tensor(estimate_gradient_delays(few; method = RING(; pairs = 3))) - S) < 0.05 * dr
end

@testitem "Gradient delays: noisy multi-coil data" tags = [:preprocessing, :acquisition, :gpu] setup = [DelaySim, GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using NamedDims
    using .DelaySim: spokes, delayed, data, golden, tensor

    N = 128
    dr = 1 / N
    nominal = spokes(golden(233), N; dr)
    S = [1.2 -0.4; -0.4 0.6] .* dr
    # RING on the physical displacement S n, OpposingSpokes on its component along the spoke.
    for method in (OpposingSpokes(), RING())
        ksp = data(delayed(nominal, S; transverse = method isa RING); ncoils = 4, noise = 0.02)
        acq = NonCartesianAcquisitionInfo(
            NamedDimsArray{(:sample, :spoke, :coil)}(ksp);
            trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(nominal), image_size = (64, 64)
        )
        @test maximum(abs, tensor(estimate_gradient_delays(acq; method)) - S) < 0.1 * dr
    end

    # On a device only the trajectory changes: the k-space stays where it is.
    acq = NonCartesianAcquisitionInfo(
        NamedDimsArray{(:sample, :spoke, :coil)}(data(delayed(nominal, S); ncoils = 4, noise = 0.02));
        trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(nominal), image_size = (64, 64)
    )
    d = estimate_gradient_delays(acq)
    corrected = correct_gradient_delays(acq)
    @test dimnames(corrected.trajectory) == (:coord, :sample, :spoke)
    for backend in all_backends()
        @testset "$(backend.name)" begin
            acq_d = to_device(backend, acq)
            @test all(values(estimate_gradient_delays(acq_d)) .≈ values(d))
            corrected_d = correct_gradient_delays(acq_d)
            @test _is_device(corrected_d)
            @test corrected_d.kspace_data === acq_d.kspace_data
            @test unname(corrected_d.trajectory) ≈ unname(corrected.trajectory)
        end
    end
end

@testitem "Gradient delays: input checks" tags = [:preprocessing, :acquisition] setup = [DelaySim] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo, CartesianAcquisitionInfo
    using .DelaySim: spokes, data, golden

    @test_throws ArgumentError correct_gradient_delays(AcquisitionInfo(zeros(ComplexF32, 16, 16); is3D = false))

    # Non-uniform sample spacing (e.g. ramp sampling) is refused rather than misread.
    nominal = spokes(golden(40), 64)
    bent = copy(nominal)
    bent[:, 1:8, :] .*= 1.2
    acq = NonCartesianAcquisitionInfo(data(nominal); trajectory = bent, image_size = (32, 32))
    @test_throws ArgumentError estimate_gradient_delays(acq; method = RING())
end

@testitem "Gradient delays: RING starts and per-spoke phase" tags = [:preprocessing, :acquisition] setup = [DelaySim] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using .DelaySim: spokes, delayed, data, golden, tensor

    N = 128
    dr = 1 / N
    nominal = spokes(golden(201), N; dr)
    S = [1.5 0.4; 0.4 0.8] .* dr
    ksp = data(delayed(nominal, S))
    acq = NonCartesianAcquisitionInfo(ksp; trajectory = nominal, image_size = (64, 64))
    for init in (:auto, :intersections, :opposing, :grid)
        @test maximum(abs, tensor(estimate_gradient_delays(acq; method = RING(; init))) - S) < 0.01 * dr
    end
    @test_throws ArgumentError estimate_gradient_delays(acq; method = RING(; init = :nearest))

    # A per-spoke phase ψx cos θ + ψy sin θ (B0 eddy currents) breaks the plain RING condition;
    # fitting it alongside the tensor recovers S.
    shifted = copy(ksp)
    for s in axes(nominal, 3)
        θ = atan(nominal[2, end, s] - nominal[2, 1, s], nominal[1, end, s] - nominal[1, 1, s])
        shifted[:, s] .*= cis(0.4 * cos(θ) - 0.25 * sin(θ))
    end
    acq_phase = NonCartesianAcquisitionInfo(shifted; trajectory = nominal, image_size = (64, 64))
    @test maximum(abs, tensor(estimate_gradient_delays(acq_phase; method = RING())) - S) > 0.5 * dr
    for init in (:auto, :intersections, :opposing, :grid)
        @test maximum(abs, tensor(estimate_gradient_delays(acq_phase; method = RING(; init, phase = true))) - S) < 0.01 * dr
    end
end
