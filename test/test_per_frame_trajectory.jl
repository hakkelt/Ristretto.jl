@testmodule PerFrameTrajectory begin
    using NamedDims

    export radial_spokes, rotating_series, moving_square, gaussian_maps, align_nrmse

    # `nspokes` radial spokes whose angles continue from spoke `first` (0.3 rad apart), so consecutive
    # frames of a series sample different directions.
    function radial_spokes(nsamples, nspokes, first)
        r = (-(nsamples ÷ 2):(nsamples - nsamples ÷ 2 - 1)) ./ nsamples
        t = Array{Float32}(undef, 2, nsamples, nspokes)
        for j in 1:nspokes
            θ = (first + j - 1) * 0.3
            t[1, :, j] .= r .* cos(θ)
            t[2, :, j] .= r .* sin(θ)
        end
        return t
    end

    rotating_series(nsamples, nspokes, nframes) = stack(radial_spokes(nsamples, nspokes, nspokes * (t - 1)) for t in 1:nframes)

    function moving_square(n, nframes)
        img = zeros(ComplexF32, n, n, nframes)
        for t in 1:nframes
            img[(n ÷ 4 + 1):(3n ÷ 4), (n ÷ 4 + t):(n ÷ 2 + 4 + t), t] .= 1
        end
        return img
    end

    gaussian_maps(n, nc) = ComplexF32.(stack([exp(-((i - 8c)^2 + (j - n ÷ 2)^2) / 300) * cis(0.1c) for i in 1:n, j in 1:n] for c in 1:nc))

    # NRMSE of magnitudes after a least-squares scale.
    function align_nrmse(x, ref)
        a = vec(abs.(unname(x)))
        b = vec(abs.(ref))
        return sqrt(sum(abs2, a .* (sum(a .* b) / sum(abs2, a)) .- b) / sum(abs2, b))
    end
end

@testitem "Per-frame trajectory: Fourier operator" tags = [:encoding, :nfft] setup = [PerFrameTrajectory] begin
    using Ristretto
    using Ristretto: get_fourier_operator, get_image_dims, NonCartesianAcquisitionInfo, _get_sample_dims_count
    using NamedDims, LinearAlgebra, Random

    n, ns, nsp, nc, nt = 16, 32, 10, 3, 4
    traj = rotating_series(ns, nsp, nt)
    ksp = rand(MersenneTwister(1), ComplexF32, ns, nsp, nc, nt)
    x = rand(MersenneTwister(2), ComplexF32, n, n, nc, nt)

    # Frame t of every coil goes through frame t's own NFFT.
    F = get_fourier_operator(ksp, (n, n), traj)
    @test size(F, 2) == (n, n, nc, nt)
    @test size(F, 1) == (ns, nsp, nc, nt)
    single = [get_fourier_operator(ksp[:, :, 1, 1], (n, n), traj[:, :, :, t]) for t in 1:nt]
    ref = stack(stack(single[t] * x[:, :, c, t] for c in 1:nc) for t in 1:nt)
    @test F * x ≈ ref
    y = rand(MersenneTwister(3), ComplexF32, ns, nsp, nc, nt)
    @test dot(F * x, y) ≈ dot(x, F' * y) rtol = 1.0e-4

    # The named form resolves the frame axis by name and builds the same operator.
    kn = NamedDimsArray{(:sample, :spoke, :coil, :time)}(ksp)
    tn = NamedDimsArray{(:coord, :sample, :spoke, :time)}(traj)
    Fn = get_fourier_operator(kn, (n, n), tn)
    @test parent(Fn * NamedDimsArray{(:x, :y, :coil, :time)}(x)) ≈ ref

    acq = NonCartesianAcquisitionInfo(kn; trajectory = tn, image_size = (n, n))
    @test _get_sample_dims_count(acq) == 2
    @test get_image_dims(acq) == (:x, :y, :coil, :time)

    # A per-frame DCF is sliced with the trajectory.
    dcf = rand(MersenneTwister(4), Float32, ns, nsp, nt)
    Fd = get_fourier_operator(ksp, (n, n), traj; dcf)
    single_d = [get_fourier_operator(ksp[:, :, 1, 1], (n, n), traj[:, :, :, t]; dcf = dcf[:, :, t]) for t in 1:nt]
    @test Fd' * y ≈ stack(stack(single_d[t]' * y[:, :, c, t] for c in 1:nc) for t in 1:nt)

    # A shared trajectory is still read as shared, and a trajectory that fits neither way is rejected.
    @test _get_sample_dims_count(NonCartesianAcquisitionInfo(ksp; trajectory = traj[:, :, :, 1], image_size = (n, n))) == 2
    @test_throws ArgumentError NonCartesianAcquisitionInfo(ksp; trajectory = rotating_series(ns, nsp, nt + 1), image_size = (n, n))

    # By size alone, a coil count equal to the frame count reads as a shared trajectory with three
    # sample axes (what such a trajectory always meant); naming the axes resolves it.
    ksq = rand(ComplexF32, ns, nsp, nt, nt)
    @test _get_sample_dims_count(NonCartesianAcquisitionInfo(ksq; trajectory = traj, image_size = (n, n))) == 3
    named_sq = NonCartesianAcquisitionInfo(
        NamedDimsArray{(:sample, :spoke, :coil, :time)}(ksq); trajectory = tn, image_size = (n, n),
    )
    @test _get_sample_dims_count(named_sq) == 2
end

@testitem "Per-frame trajectory: simulation, DCF and reconstruction" tags = [:acquisition, :nfft, :reconstruction] setup = [PerFrameTrajectory] begin
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo, compute_dcf
    using NamedDims

    n, ns, nsp, nc, nt = 32, 64, 24, 4, 6
    traj = rotating_series(ns, nsp, nt)
    img = moving_square(n, nt)
    smaps = gaussian_maps(n, nc)
    template = NonCartesianAcquisitionInfo(;
        trajectory = NamedDimsArray{(:coord, :sample, :spoke, :time)}(traj),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps), image_size = (n, n),
    )
    acq = simulate_acquisition(NamedDimsArray{(:x, :y, :time)}(img), template; inverse_crime_check = false, keep_sensitivity_maps = true)
    @test dimnames(acq.kspace_data) == (:sample, :spoke, :coil, :time)
    @test size(acq.kspace_data) == (ns, nsp, nc, nt)

    # Each frame's samples are that frame's image seen through that frame's trajectory.
    for t in (1, nt)
        one = simulate_acquisition(
            NamedDimsArray{(:x, :y)}(img[:, :, t]),
            NonCartesianAcquisitionInfo(;
                trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(traj[:, :, :, t]),
                sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps), image_size = (n, n),
            ); inverse_crime_check = false, keep_sensitivity_maps = true
        )
        @test parent(acq.kspace_data)[:, :, :, t] ≈ parent(one.kspace_data)
    end

    # One set of density weights per frame, each from that frame's samples alone.
    acqd = density_compensation(acq)
    @test size(acqd.dcf) == (ns, nsp, nt)
    @test parent(acqd.dcf)[:, :, 2] ≈ compute_dcf(traj[:, :, :, 2], (n, n), PipeMenonDCF())

    xg = reconstruct(acqd, DirectReconstruction(); verbosity = Silent())
    @test dimnames(xg) == (:x, :y, :time)
    @test align_nrmse(parent(xg) ./ sum(abs2, smaps; dims = 3), img) < 0.35

    # CG-SENSE with the frames' own trajectories recovers the series; the same k-space
    # reconstructed as if every frame shared frame 1's trajectory does not.
    xc = reconstruct(acq, IterativeReconstruction(; maxit = 30); verbosity = Silent())
    @test align_nrmse(xc, img) < 0.1
    shared = NonCartesianAcquisitionInfo(
        acq.kspace_data; trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(traj[:, :, :, 1]),
        sensitivity_maps = acq.sensitivity_maps, image_size = (n, n),
    )
    @test align_nrmse(reconstruct(shared, IterativeReconstruction(; maxit = 30); verbosity = Silent()), img) > 0.3

    # A temporal regularizer couples the frames through the per-frame operator.
    xl = reconstruct(acq, IterativeReconstruction(LowRank(0.01; time_dim = :time); maxit = 20); verbosity = Silent())
    @test size(xl) == (n, n, nt)
    @test align_nrmse(xl, img) < 0.2

    # Maps are estimated from the time-averaged gridded images.
    est = estimate_sensitivities(NonCartesianAcquisitionInfo(acq; sensitivity_maps = nothing))
    @test dimnames(est.sensitivity_maps) == (:x, :y, :coil)
    @test size(est.sensitivity_maps) == (n, n, nc)

    # Gradient delays pool the spokes of every frame and correct every frame alike.
    @test all(abs.(values(estimate_gradient_delays(acq))) .< 1.0e-2)
    @test size(correct_gradient_delays(acq).trajectory) == size(traj)
end

@testitem "Per-frame trajectory: simulation reads frames as the constructor does" tags = [:acquisition, :nfft, :simulation] setup = [PerFrameTrajectory] begin
    using Ristretto: NonCartesianAcquisitionInfo

    # A plain-array trajectory with as many spokes as the image has frames: sizes cannot tell its
    # spoke axis from a frame axis, so it is shared, every frame getting all of its spokes.
    n, ns, nsp, nc = 16, 32, 6, 3
    traj = radial_spokes(ns, nsp, 0)
    smaps = rand(ComplexF32, n, n, nc)
    shared = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (n, n), sensitivity_maps = smaps)
    @test size(simulate_acquisition(rand(ComplexF32, n, n, nsp), shared; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data) == (ns, nsp, nc, nsp)
    # Named axes make the same trajectory a per-frame one, a spoke per frame.
    named = NonCartesianAcquisitionInfo(
        nothing; trajectory = NamedDimsArray{(:coord, :sample, :time)}(traj), image_size = (n, n), sensitivity_maps = smaps,
    )
    @test size(simulate_acquisition(NamedDimsArray{(:x, :y, :time)}(rand(ComplexF32, n, n, nsp)), named; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data) == (ns, nc, nsp)
    # An acquisition that carries k-space has had its frames read by its constructor already.
    template = NonCartesianAcquisitionInfo(
        zeros(ComplexF32, ns, nc, nsp); trajectory = traj, image_size = (n, n), sensitivity_maps = smaps,
    )
    @test size(simulate_acquisition(rand(ComplexF32, n, n, nsp), template; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data) == (ns, nc, nsp)

    # 4-D maps of a 2-D encoding are `(x, y, coil, slice)`: the coil count is their third axis.
    nsl = 2
    multislice = NonCartesianAcquisitionInfo(
        nothing; trajectory = traj, image_size = (n, n), sensitivity_maps = rand(ComplexF32, n, n, nc, nsl),
    )
    @test size(simulate_acquisition(rand(ComplexF32, n, n, nsl), multislice; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data) == (ns, nsp, nc, nsl)
end

@testitem "Per-frame trajectory: single coil without a coil axis" tags = [:acquisition, :nfft, :reconstruction] setup = [PerFrameTrajectory] begin
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo, get_image_dims, _get_sample_dims_count
    using NamedDims

    n, ns, nsp, nt = 32, 64, 24, 6
    traj = NamedDimsArray{(:coord, :sample, :spoke, :time)}(rotating_series(ns, nsp, nt))
    img = moving_square(n, nt)
    one_coil = simulate_acquisition(
        NamedDimsArray{(:x, :y, :time)}(img),
        NonCartesianAcquisitionInfo(;
            trajectory = traj, sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(ones(ComplexF32, n, n, 1)),
            image_size = (n, n),
        ); inverse_crime_check = false
    )
    ksp = parent(one_coil.kspace_data)

    # k-space (:sample, :spoke, :time) with a per-frame trajectory: the trailing :time axis is a
    # frame axis, so every frame keeps its own spokes instead of merging into one image.
    acq = NonCartesianAcquisitionInfo(NamedDimsArray{(:sample, :spoke, :time)}(ksp[:, :, 1, :]); trajectory = traj, image_size = (n, n))
    @test _get_sample_dims_count(acq) == 2
    @test get_image_dims(acq) == (:x, :y, :time)
    xc = reconstruct(acq, IterativeReconstruction(; maxit = 30); verbosity = Silent())
    @test size(xc) == (n, n, nt)
    @test align_nrmse(xc, img) < 0.15   # one coil: 0.103; the multi-coil item above reaches < 0.1

    # With a singleton :coil axis and no maps, a temporal term leaves :coil to task splitting; each
    # task's k-space then lacks a coil axis and must still be read per frame.
    acq_c = NonCartesianAcquisitionInfo(NamedDimsArray{(:sample, :spoke, :coil, :time)}(ksp); trajectory = traj, image_size = (n, n))
    xt = reconstruct(acq_c, IterativeReconstruction(TemporalTotalVariation(1.0e-3; time_dim = :time); maxit = 20); verbosity = Silent())
    @test size(xt) == (n, n, 1, nt)
    xt_whole = reconstruct(
        acq_c, IterativeReconstruction(TemporalTotalVariation(1.0e-3; time_dim = :time); maxit = 20);
        verbosity = Silent(), disable_task_splitting = true
    )
    @test parent(xt) ≈ parent(xt_whole) rtol = 1.0e-3
    @test align_nrmse(xt[:, :, 1, :], img) < 0.15
end
