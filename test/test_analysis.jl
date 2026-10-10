@testitem "Pseudo-replica analysis and g-factor maps" tags = [:reconstruction, :acquisition] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims
    using Random

    Nx, Ny = 16, 16
    img = NamedDimsArray{(:x, :y)}(ones(ComplexF32, Nx, Ny))
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(zeros(ComplexF32, Nx, Ny));
        is3D = false,
    )
    acq_sim = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

    # 1. Validation tests
    @test_throws ArgumentError pseudo_replica(acq_sim; replicas = 1)
    @test_throws ArgumentError pseudo_replica(acq_sim; replicas = 10, scaling = BartScaling())

    # 2. Fully sampled single-coil Cartesian: g-factor must be approximately 1.0 everywhere
    rng = Random.MersenneTwister(42)
    res = pseudo_replica(acq_sim; replicas = 100, noise_std = 1.0, rng)
    @test haskey(res, :mean)
    @test haskey(res, :std)
    @test haskey(res, :g_factor)
    @test res.mean isa NamedDimsArray
    @test dimnames(res.mean) == (:x, :y)
    @test size(res.g_factor) == (Nx, Ny)

    # Theoretical noise std in image space for 16x16 with 1/sqrt(256)=1/16 scaling is 0.0625
    @test isapprox(unname(res.g_factor), ones(Float32, Nx, Ny); rtol = 0.15)
end

@testitem "Subspace reconstruction: T2 decay simulation with TemporalBasis" tags = [:reconstruction, :simulation] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nt = 16, 16, 12
    K = 3

    # Generate synthetic T2-decay dynamic image series: img(x, y, t) = rho(x, y) * exp(-t / T2(x, y))
    rho = zeros(Float32, Nx, Ny)
    rho[4:12, 4:12] .= 1.0f0
    T2_map = fill(5.0f0, Nx, Ny) # T2 = 5 frames

    img_series = NamedDimsArray{(:x, :y, :time)}(zeros(ComplexF32, Nx, Ny, Nt))
    for t in 1:Nt
        img_series[:, :, t] = rho .* exp.(-(t - 1) ./ T2_map)
    end

    # Build low-dimensional temporal subspace basis from exponential decay dictionary
    T2_candidates = range(1.0f0, 20.0f0, length = 50)
    dict = zeros(Float32, Nt, length(T2_candidates))
    for (i, t2) in enumerate(T2_candidates)
        dict[:, i] = exp.(-(0:(Nt - 1)) ./ t2)
    end
    F_svd = svd(dict)
    Φ = Matrix{ComplexF32}(F_svd.U[:, 1:K]) # (Nt, K)

    # Simulate acquisition
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :time)}(zeros(ComplexF32, Nx, Ny, Nt));
        is3D = false,
    )
    acq_sim = simulate_acquisition(img_series, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

    # Reconstruct with subspace model
    method_subspace = IterativeReconstruction(;
        algorithm = CGNR(maxit = 20, tol = 1.0e-5),
        signal_model = TemporalBasis(Φ; time_dim = :time),
    )
    rec = reconstruct(acq_sim, method_subspace; verbosity = Silent())

    @test parent(rec) isa NamedDimsArray
    @test dimnames(rec) == (:x, :y, :time)
    @test size(rec) == (Nx, Ny, Nt)

    # Subspace model should reconstruct the series accurately
    mask = abs.(unname(img_series)) .> 0.1
    @test isapprox(unname(rec)[mask], unname(img_series)[mask]; rtol = 0.05)
end

@testitem "Pseudo-replica: non-Cartesian acquisition" tags = [:analysis, :nfft] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo

    Nsamples, Nspokes, Nc = 48, 21, 4
    angles = range(0, π, length = Nspokes + 1)[1:Nspokes]
    r = range(-0.49, 0.49, length = Nsamples)
    traj = zeros(2, Nsamples, Nspokes)
    for s in 1:Nspokes
        traj[1, :, s] = r .* cos(angles[s])
        traj[2, :, s] = r .* sin(angles[s])
    end
    ksp = randn(ComplexF64, Nsamples, Nspokes, Nc) .* 0.01
    acq = NonCartesianAcquisitionInfo(ksp; trajectory = traj, image_size = (24, 24))

    # Must not throw on `acq.subsampling` (a field NonCartesianAcquisitionInfo lacks)
    out = pseudo_replica(acq; replicas = 4, noise_std = 1.0, scaling = NoScaling())
    @test size(out.g_factor)[1:2] == (24, 24)   # per-coil output (no coil combination without sens maps)
    @test all(isfinite, out.g_factor)
end
