@testitem "Partial Fourier reconstruction: Homodyne, StepRamp, POCS" tags = [:reconstruction, :acquisition, :gpu] setup = [GpuEnvSetup, GpuHelpers] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny = 32, 32
    mag = zeros(Float32, Nx, Ny)
    mag[8:24, 8:24] .= 1.0f0
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    phase_true = 0.5f0 .* (X .+ Y)
    img_true = NamedDimsArray{(:x, :y)}(ComplexF32.(mag .* cis.(phase_true)))

    # Full k-space
    ksp_full = fftshift(fft(unname(img_true)))

    # Partial Fourier subsampling: lines 10 to 32 acquired (23 lines out of 32)
    mask_y = falses(Ny)
    mask_y[10:32] .= true
    subs = (:, mask_y)

    ksp_partial = NamedDimsArray{(:kx, :ky)}(ksp_full[:, mask_y])
    acq_pf = CartesianAcquisitionInfo(
        ksp_partial;
        is3D = false,
        image_size = (Nx, Ny),
        subsampling = subs,
    )

    # 1. Test partial_fourier_band
    band = partial_fourier_band(acq_pf)
    @test band.dim == 2
    @test first(band.acquired_range) == 10
    @test last(band.acquired_range) == 32

    # 2. Test Homodyne with LinearRamp
    rec_homodyne_linear = reconstruct(acq_pf, Homodyne(filter = LinearRamp()); verbosity = Silent())
    @test parent(rec_homodyne_linear) isa NamedDimsArray
    @test dimnames(rec_homodyne_linear) == (:x, :y)
    mask_obj = mag .> 0.5
    @test isapprox(abs.(unname(rec_homodyne_linear))[mask_obj], mag[mask_obj]; rtol = 0.08)

    # 3. Test Homodyne with StepRamp
    rec_homodyne_step = reconstruct(acq_pf, Homodyne(filter = StepRamp()); verbosity = Silent())
    @test isapprox(abs.(unname(rec_homodyne_step))[mask_obj], mag[mask_obj]; rtol = 0.08)

    # 4. Test POCS
    rec_pocs = reconstruct(acq_pf, POCS(maxit = 15); verbosity = Silent())
    @test isapprox(abs.(unname(rec_pocs))[mask_obj], mag[mask_obj]; rtol = 0.08)

    for method in (Homodyne(filter = LinearRamp()), Homodyne(filter = StepRamp()), POCS(maxit = 15))
        test_on_devices(a -> reconstruct(a, method; verbosity = Silent()), acq_pf)
    end
end

@testitem "Parallel imaging: GRAPPA and SPIRiT" tags = [:reconstruction, :acquisition, :encoding, :gpu] setup = [SyntheticCoils, GpuEnvSetup, GpuHelpers] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nc = 32, 32, 4
    img = zeros(Float32, Nx, Ny)
    img[8:24, 8:24] .= 1.0f0

    sens_true = synthetic_sensitivities(ComplexF32, Nx, Ny, Nc)

    # Full k-space
    ksp_full = zeros(ComplexF32, Nx, Ny, Nc)
    for c in 1:Nc
        ksp_full[:, :, c] = fftshift(fft(img .* sens_true[:, :, c]))
    end

    # R = 2 undersampling with central 12-line ACS region
    R_acc = 2
    cal_range = 11:22
    mask_y = falses(Ny)
    mask_y[1:R_acc:Ny] .= true
    mask_y[cal_range] .= true
    subs = (:, mask_y)

    ksp_under = NamedDimsArray{(:kx, :ky, :coil)}(ksp_full[:, mask_y, :])
    sens_named = NamedDimsArray{(:x, :y, :coil)}(sens_true)

    acq = CartesianAcquisitionInfo(
        ksp_under;
        is3D = false,
        image_size = (Nx, Ny),
        subsampling = subs,
        sensitivity_maps = sens_named,
    )

    mask_obj = img .> 0.5

    # 1. GRAPPA reconstruction
    rec_grappa = reconstruct(acq, GRAPPA(kernel_size = (3, 2), calib_size = (32, 12)); verbosity = Silent())
    @test parent(rec_grappa) isa NamedDimsArray
    @test dimnames(rec_grappa) == (:x, :y)
    @test isapprox(abs.(unname(rec_grappa))[mask_obj], img[mask_obj]; rtol = 0.05)

    # 2. SPIRiT reconstruction
    rec_spirit = reconstruct(acq, SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20); verbosity = Silent())
    @test parent(rec_spirit) isa NamedDimsArray
    @test dimnames(rec_spirit) == (:x, :y)
    # Tight tolerance: with its kernel applied in the calibration's own correlation convention
    # (see the `_spirit_gfft` test item) SPIRiT is accurate here, not merely in the right ballpark.
    spirit_err = norm(abs.(unname(rec_spirit))[mask_obj] .- img[mask_obj]) / norm(img[mask_obj])
    @test spirit_err < 0.03

    # Both run on a host copy of a device acquisition and hand the image back on the device, so an
    # executor that a device rejects (slices sharing one device) is fine for them.
    for method in (GRAPPA(kernel_size = (3, 2), calib_size = (32, 12)), SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20))
        test_on_devices(a -> reconstruct(a, method; verbosity = Silent()), acq)
        test_on_devices(a -> reconstruct(a, method; verbosity = Silent(), task_executor = MultiThreadingExecutor()), acq)
    end
end

@testitem "Partial Fourier: PhaseConstrained recovers a phased phantom" tags = [:reconstruction, :acquisition, :gpu] setup = [SyntheticCoils, GpuEnvSetup, GpuHelpers] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nc = 40, 40, 4
    img = zeros(ComplexF64, Nx, Ny)
    img[10:30, 12:28] .= 1.0
    img[18:24, 18:24] .= 0.4
    img .*= cis.(0.4 .* [x / Nx + y / Ny for x in 1:Nx, y in 1:Ny])

    sens = synthetic_sensitivities(ComplexF64, Nx, Ny, Nc)

    full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        full[:, :, c] = fftshift(fft(img .* sens[:, :, c]))
    end

    pf = 26
    mask_y = falses(Ny)
    mask_y[1:pf] .= true
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(full[:, 1:pf, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )

    rec = reconstruct(acq, PhaseConstrained(); verbosity = Silent())
    @test parent(rec) isa NamedDimsArray
    @test dimnames(rec) == (:x, :y)
    obj = abs.(img) .> 0.2
    @test isapprox(abs.(unname(rec))[obj], abs.(img)[obj]; rtol = 0.05)

    test_on_devices(a -> reconstruct(a, PhaseConstrained(); verbosity = Silent()), acq; rtol = 1.0e-3)
end

@testitem "GRAPPA: arbitrary undersampling factor and default even kernel" tags = [:reconstruction, :acquisition, :encoding] setup = [SyntheticCoils] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nc = 40, 40, 6
    img = zeros(ComplexF64, Nx, Ny)
    img[10:30, 10:30] .= 1.0
    img[15:20, 22:26] .= 0.5

    sens = synthetic_sensitivities(ComplexF64, Nx, Ny, Nc; phase_scale = 0.6)

    full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        full[:, :, c] = fftshift(fft(img .* sens[:, :, c]))
    end

    for R in (2, 3)
        mask_y = falses(Ny)
        mask_y[1:R:Ny] .= true
        c0 = Ny ÷ 2 + 1
        mask_y[(c0 - 6):(c0 + 5)] .= true
        acq = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(full[:, mask_y, :]);
            is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
            sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
        )
        # default kernel_size = (4, 3) is even along kx - must not throw
        rec = reconstruct(acq, GRAPPA(calib_size = (40, 12)); verbosity = Silent())
        @test dimnames(rec) == (:x, :y)
        rel = norm(abs.(unname(rec)) .- abs.(img)) / norm(abs.(img))
        @test rel < (R == 2 ? 0.05 : 0.2)
    end
end

@testitem "Direct methods: trailing time batch dimension is preserved" tags = [:reconstruction, :acquisition, :encoding] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using Ristretto: Homodyne, POCS, PhaseConstrained
    using NamedDims
    using FFTW

    Nx, Ny, Nc, Nt = 32, 32, 4, 3
    sens = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        sens[:, :, c] .= cis(2π * c / Nc) / sqrt(Nc)
    end
    imgs = zeros(ComplexF64, Nx, Ny, Nt)
    for t in 1:Nt
        imgs[8:24, 8:24, t] .= 1.0 + 0.1t
    end
    full = zeros(ComplexF64, Nx, Ny, Nc, Nt)
    for c in 1:Nc, t in 1:Nt
        full[:, :, c, t] = fftshift(fft(imgs[:, :, t] .* sens[:, :, c]))
    end
    mask_y = falses(Ny)
    mask_y[1:22] .= true                       # partial Fourier band
    acq_pf = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil, :time)}(full[:, 1:22, :, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )
    for M in (Homodyne(), POCS(maxit = 4), PhaseConstrained())
        rec = reconstruct(acq_pf, M; verbosity = Silent())
        @test size(rec) == (Nx, Ny, Nt)
        @test dimnames(rec) == (:x, :y, :time)
    end

    mask_r = falses(Ny)
    mask_r[1:2:Ny] .= true
    c0 = Ny ÷ 2 + 1
    mask_r[(c0 - 6):(c0 + 5)] .= true
    acq_r = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil, :time)}(full[:, mask_r, :, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_r),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )
    for M in (GRAPPA(calib_size = (32, 12)), SPIRiT(calib_size = (32, 12), maxit = 6))
        rec = reconstruct(acq_r, M; verbosity = Silent())
        @test size(rec) == (Nx, Ny, Nt)
        @test dimnames(rec) == (:x, :y, :time)
    end
end

@testitem "SPIRiTConsistency: operator adjoint test and KSpaceToImage reconstruction" tags = [:reconstruction, :regularization] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using NamedDims
    using LinearAlgebra
    using FFTW
    using Ristretto.StructuredOptimization

    Nx, Ny, Nc = 16, 16, 4
    Kx, Ky = 3, 3
    kernel = rand(ComplexF64, Kx, Ky, Nc, Nc)
    cx, cy = 2, 2
    for c in 1:Nc
        kernel[cx, cy, c, c] = 0.0
    end

    reg = SPIRiTConsistency(kernel; λ = 1.0)
    @test Ristretto.scale_regularization(reg, 2.0).λ == 2.0
    @test Ristretto.bind_dimensions(reg, (:x, :y)) === reg

    # 1. Adjoint dot-test on (I - G)
    x = randn(ComplexF64, Nx, Ny, Nc)
    y = randn(ComplexF64, Nx, Ny, Nc)
    op = Ristretto.get_operator(reg, x; threaded = false)
    Ax = op * x
    Aty = op' * y
    @test isapprox(dot(y, Ax), dot(Aty, x); rtol = 1.0e-10)

    # 2. Fully-sampled KSpaceToImage solve matches direct IFFT
    img = zeros(ComplexF64, Nx, Ny)
    img[4:12, 4:12] .= 1.0
    sens = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        sens[:, :, c] .= cis(2π * c / Nc) / sqrt(Nc)
    end
    ksp_full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        ksp_full[:, :, c] = fftshift(fft(img .* sens[:, :, c]))
    end
    acq_full = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp_full);
        is3D = false, image_size = (Nx, Ny),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )
    rec_kspace = reconstruct(acq_full, IterativeReconstruction(; signal_model = KSpaceToImage(AdjointSensitivity()), algorithm = CGNR(maxit = 5), fidelity = L2Loss()); verbosity = Silent())
    @test isapprox(abs.(unname(rec_kspace)), abs.(img); atol = 1.0e-5)
end

@testitem "Iterative SPIRiT reconstruction (lowering)" tags = [:reconstruction, :acquisition] setup = [SyntheticCoils] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using NamedDims
    using LinearAlgebra
    using FFTW

    Nx, Ny, Nc = 32, 32, 4
    img = zeros(Float32, Nx, Ny)
    img[8:24, 8:24] .= 1.0f0

    sens_true = synthetic_sensitivities(ComplexF32, Nx, Ny, Nc)

    ksp_full = zeros(ComplexF32, Nx, Ny, Nc)
    for c in 1:Nc
        ksp_full[:, :, c] = fftshift(fft(img .* sens_true[:, :, c]))
    end

    R_acc = 2
    cal_range = 11:22
    mask_y = falses(Ny)
    mask_y[1:R_acc:Ny] .= true
    mask_y[cal_range] .= true
    subs = (:, mask_y)

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp_full[:, mask_y, :]);
        is3D = false,
        image_size = (Nx, Ny),
        subsampling = subs,
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens_true),
    )

    rec_iter = reconstruct(acq, SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20, iterative = true); verbosity = Silent())
    @test parent(rec_iter) isa NamedDimsArray
    @test dimnames(rec_iter) == (:x, :y)
    mask_obj = img .> 0.5
    @test norm(abs.(unname(rec_iter))[mask_obj] .- img[mask_obj]) / norm(img[mask_obj]) < 0.2
end

@testitem "Direct FFT methods respect shifted_kspace_dims" tags = [:reconstruction, :acquisition] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using NamedDims
    using FFTW
    using LinearAlgebra

    Nx, Ny = 32, 32
    mag = zeros(Float32, Nx, Ny)
    mag[8:24, 8:24] .= 1.0f0
    img_true = NamedDimsArray{(:x, :y)}(ComplexF32.(mag))

    # Standard DC-centered k-space
    ksp_centered = fftshift(fft(unname(img_true))) ./ sqrt(Nx * Ny)
    # Pre-shifted (corner-centered) k-space
    ksp_unshifted = fft(unname(img_true)) ./ sqrt(Nx * Ny)

    acq_default = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(ksp_centered);
        is3D = false, image_size = (Nx, Ny),
    )
    acq_shifted = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(ksp_unshifted);
        is3D = false, image_size = (Nx, Ny),
        shifted_kspace_dims = (1, 2),
    )

    rec_default = reconstruct(acq_default, DirectReconstruction(); verbosity = Silent())
    rec_shifted = reconstruct(acq_shifted, DirectReconstruction(); verbosity = Silent())

    @test isapprox(abs.(unname(rec_default)), abs.(unname(rec_shifted)); atol = 1.0e-5)
end

@testitem "DirectReconstruction: coil combination without sensitivity maps" tags = [:reconstruction, :acquisition] setup = [SyntheticCoils] begin
    using Test
    using Ristretto
    using NamedDims

    Nx, Ny, Nc = 32, 32, 4
    img_true = ComplexF32.(reshape(range(0.0f0, 1.0f0; length = Nx * Ny), Nx, Ny))
    sens = synthetic_sensitivities(ComplexF32, Nx, Ny, Nc)
    data = simulate_acquisition(
        img_true, AcquisitionInfo(is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens); inverse_crime_check = false, keep_sensitivity_maps = true
    )
    # The same k-space, described without maps: its coil axis is a batch dimension unless the
    # method itself combines the channels.
    no_maps = AcquisitionInfo(data.kspace_data; is3D = false, image_size = (Nx, Ny))

    # The default resolves to NoCoilCombination; AdjointSensitivity is defined by the maps and is
    # an error without them.
    @test_throws ArgumentError reconstruct(no_maps, DirectReconstruction(AdjointSensitivity()); verbosity = Silent())

    # RootSumSquares needs no maps -- it is the maps-free reference -- so naming it explicitly
    # combines the channels of a maps-less acquisition instead of failing.
    rec_rss_no_maps = reconstruct(no_maps, DirectReconstruction(RootSumSquares()); verbosity = Silent())
    @test size(rec_rss_no_maps) == (Nx, Ny)
    per_coil = reconstruct(no_maps, DirectReconstruction(NoCoilCombination()); verbosity = Silent())
    @test unname(rec_rss_no_maps) ≈ sqrt.(sum(abs2, unname(per_coil); dims = 3))[:, :, 1]
    @test Ristretto.lower(DirectReconstruction(), no_maps).coil_combination === NoCoilCombination()
    rec_default = reconstruct(no_maps, DirectReconstruction(); verbosity = Silent())
    rec = reconstruct(no_maps, DirectReconstruction(NoCoilCombination()); verbosity = Silent())
    @test size(rec) == (Nx, Ny, Nc)
    @test unname(rec_default) == unname(rec)

    # With a second batch dimension the reconstruction is split over it, and the coil axis must
    # stay out of the split: a per-coil slab would reduce one channel at a time and combine
    # nothing. Named so the merged result can be checked to have lost exactly `:coil`.
    Nz = 3
    ksp_slices = NamedDimsArray{(:kx, :ky, :coil, :z)}(
        cat((unname(data.kspace_data) for _ in 1:Nz)...; dims = 4)
    )
    no_maps_3d = AcquisitionInfo(ksp_slices; is3D = false, image_size = (Nx, Ny))
    rec_rss_3d = reconstruct(no_maps_3d, DirectReconstruction(RootSumSquares()); verbosity = Silent())
    @test :coil ∉ dimnames(rec_rss_3d)
    @test last(dimnames(rec_rss_3d)) == :z
    @test size(rec_rss_3d) == (Nx, Ny, Nz)
    for z in 1:Nz
        @test unname(rec_rss_3d)[:, :, z] ≈ unname(rec_rss_no_maps)
    end

    # Single-channel data has no coil axis at all: the default still works.
    single = AcquisitionInfo(
        simulate_acquisition(img_true, AcquisitionInfo(is3D = false, image_size = (Nx, Ny)); inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data;
        is3D = false, image_size = (Nx, Ny)
    )
    @test size(reconstruct(single, DirectReconstruction(); verbosity = Silent())) == (Nx, Ny)
end

@testitem "DirectReconstruction: coil combination on non-Cartesian data" tags = [:reconstruction, :acquisition, :nfft] setup = [SyntheticCoils] begin
    using Test
    using Ristretto
    using NamedDims

    N, Nc = 32, 4
    img_true = ComplexF32.(reshape(range(0.0f0, 1.0f0; length = N * N), N, N))
    sens = synthetic_sensitivities(ComplexF32, N, N, Nc)
    traj = Float32.(radial_trajectory(N, 24))
    data = simulate_acquisition(
        img_true, AcquisitionInfo(; trajectory = traj, image_size = (N, N), sensitivity_maps = sens); inverse_crime_check = false, keep_sensitivity_maps = true
    )

    rec_adj = reconstruct(data, DirectReconstruction(); verbosity = Silent())
    rec_rss = reconstruct(data, DirectReconstruction(RootSumSquares()); verbosity = Silent())
    rec_none = reconstruct(data, DirectReconstruction(NoCoilCombination()); verbosity = Silent())
    @test size(rec_adj) == (N, N)
    @test size(rec_rss) == (N, N)
    @test size(rec_none) == (N, N, Nc)
    # Root sum of squares of the per-coil images is what `NoCoilCombination` leaves reconstructible.
    @test unname(rec_rss) ≈ sqrt.(sum(abs2, unname(rec_none); dims = 3))[:, :, 1]
    @test !isapprox(abs.(unname(rec_adj)), unname(rec_rss))

    # Dimension names survive the combination.
    traj_named = NamedDimsArray{(:coord, :sample, :spoke)}(traj)
    sens_named = NamedDimsArray{(:x, :y, :coil)}(unname(sens))
    data_named = simulate_acquisition(
        NamedDimsArray{(:x, :y)}(img_true),
        AcquisitionInfo(; trajectory = traj_named, image_size = (N, N), sensitivity_maps = sens_named); inverse_crime_check = false, keep_sensitivity_maps = true
    )
    @test dimnames(reconstruct(data_named, DirectReconstruction(RootSumSquares()); verbosity = Silent())) == (:x, :y)
    @test dimnames(reconstruct(data_named, DirectReconstruction(NoCoilCombination()); verbosity = Silent())) == (:x, :y, :coil)
end

@testitem "DirectReconstruction: coil_combination actually changes the result" tags = [:reconstruction, :acquisition] setup = [SyntheticCoils] begin
    using Test
    using Ristretto
    using NamedDims
    using LinearAlgebra

    Nx, Ny, Nc = 32, 32, 4
    X = [(i - Nx / 2) / Nx for i in 1:Nx, j in 1:Ny]
    Y = [(j - Ny / 2) / Ny for i in 1:Nx, j in 1:Ny]
    mag = zeros(Float32, Nx, Ny)
    mag[8:24, 8:24] .= 1.0f0
    # A genuinely complex ground truth (nonzero phase) so AdjointSensitivity and RootSumSquares
    # are mathematically distinguishable: both reduce to the same thing for a real-valued image.
    img_true = ComplexF32.(mag) .* cis.(0.8f0 .* Float32.(X .+ Y))

    sens = synthetic_sensitivities(ComplexF32, Nx, Ny, Nc)
    acq = AcquisitionInfo(is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens)
    data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

    rec_adj = reconstruct(data, DirectReconstruction(coil_combination = AdjointSensitivity()); verbosity = Silent())
    rec_rss = reconstruct(data, DirectReconstruction(coil_combination = RootSumSquares()); verbosity = Silent())
    rec_none = reconstruct(data, DirectReconstruction(coil_combination = NoCoilCombination()); verbosity = Silent())

    @test size(rec_adj) == (Nx, Ny)
    @test size(rec_rss) == (Nx, Ny)
    @test size(rec_none) == (Nx, Ny, Nc)
    # Genuinely different results, not the old bug where all three collapsed to 𝒜' * kspace_data.
    @test !isapprox(unname(rec_adj), unname(rec_rss))
    @test isapprox(unname(rec_adj), img_true; rtol = 0.05)
    @test isapprox(unname(rec_rss), abs.(img_true); rtol = 0.05)

    # Also correct on Cartesian-subsampled data (measured k-space only, not zero-filled by hand).
    mask_y = falses(Ny)
    mask_y[1:2:Ny] .= true
    mask_y[13:20] .= true
    acq_sub = AcquisitionInfo(
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y), sensitivity_maps = sens,
    )
    data_sub = simulate_acquisition(img_true, acq_sub; inverse_crime_check = false, keep_sensitivity_maps = true)
    rec_rss_sub = reconstruct(data_sub, DirectReconstruction(coil_combination = RootSumSquares()); verbosity = Silent())
    @test size(rec_rss_sub) == (Nx, Ny)

    # `AdjointSensitivity` reuses the adjoint of the encoding operator instead of combining
    # per-coil images; both must agree, fully sampled and subsampled.
    combine(coil_imgs) = dropdims(sum(unname(coil_imgs) .* conj.(unname(sens)); dims = 3); dims = 3)
    @test unname(rec_adj) ≈ combine(rec_none)
    rec_adj_sub = reconstruct(data_sub, DirectReconstruction(coil_combination = AdjointSensitivity()); verbosity = Silent())
    rec_none_sub = reconstruct(data_sub, DirectReconstruction(coil_combination = NoCoilCombination()); verbosity = Silent())
    @test unname(rec_adj_sub) ≈ combine(rec_none_sub)

    # Same three combinations with *named* dimensions. `get_image_dims` describes the combined
    # image, so with sensitivity maps present it has no `:coil` axis; `NoCoilCombination` returns
    # an array that does, and used to be labelled with the two-name tuple (a `NamedDimsArray`
    # constructor error).
    sens_named = NamedDimsArray{(:x, :y, :coil)}(unname(sens))
    acq_named = AcquisitionInfo(is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens_named)
    data_named = simulate_acquisition(NamedDimsArray{(:x, :y)}(img_true), acq_named; inverse_crime_check = false, keep_sensitivity_maps = true)
    @test data_named.kspace_data isa NamedDimsArray

    rec_named_adj = reconstruct(data_named, DirectReconstruction(coil_combination = AdjointSensitivity()); verbosity = Silent())
    rec_named_rss = reconstruct(data_named, DirectReconstruction(coil_combination = RootSumSquares()); verbosity = Silent())
    rec_named_none = reconstruct(data_named, DirectReconstruction(coil_combination = NoCoilCombination()); verbosity = Silent())
    @test dimnames(rec_named_adj) == (:x, :y)
    @test dimnames(rec_named_rss) == (:x, :y)
    @test dimnames(rec_named_none) == (:x, :y, :coil)
    @test size(rec_named_none) == (Nx, Ny, Nc)
end


@testitem "Verbosity modes and method-owned iteration parameters" tags = [:reconstruction, :integration] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: Verbosity
    using LinearAlgebra
    using Random
    using FFTW
    using Ristretto: lower

    Random.seed!(42)
    Nx, Ny, Nc = 32, 32, 2
    img = zeros(ComplexF32, Nx, Ny)
    img[9:24, 9:24] .= 1
    smaps = coil_sensitivities(Nx, Ny, Nc)

    acq = simulate_acquisition(
        img, AcquisitionInfo(nothing; is3D = false, image_size = (Nx, Ny), sensitivity_maps = smaps); inverse_crime_check = false, keep_sensitivity_maps = true
    )
    # Partial-Fourier acquisition (single coil), for the methods that need one.
    mask_y = falses(Ny)
    mask_y[10:Ny] .= true
    ksp_full = fftshift(fft(img))
    acq_pf = CartesianAcquisitionInfo(
        ksp_full[:, mask_y];
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
    )

    @testset "run keywords reject method parameters" begin
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); maxit = 5)
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); tol = 1.0e-5)
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); reltol = 1.0e-5)
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); algorithm = FISTA())
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); verbose = false)
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); printfunc = println)
        @test_throws ArgumentError reconstruct(acq, DirectReconstruction(); freq = 1)
        # `ReconstructionConfig` itself has no such field at all.
        @test_throws MethodError ReconstructionConfig(; maxit = 5)
    end

    @testset "as_verbosity shorthands" begin
        @test Ristretto.as_verbosity(:silent) === Silent()
        @test Ristretto.as_verbosity(:progress) isa ProgressBar
        @test Ristretto.as_verbosity(:verbose) isa Verbose
        @test ReconstructionConfig(; verbosity = :silent).verbosity === Silent()
        @test_throws ArgumentError ReconstructionConfig(; verbosity = :loud)
        # Three modes, so a Bool cannot name one.
        @test_throws ArgumentError Ristretto.as_verbosity(true)
        @test_throws ArgumentError Ristretto.as_verbosity(false)
        @test_throws ArgumentError ReconstructionConfig(; verbosity = false)
    end

    @testset "Silent produces no output" begin
        method = IterativeReconstruction(L2Image(0.01f0); maxit = 3)
        silent_out = mktemp() do path, io
            redirect_stdout(io) do
                reconstruct(acq, method; verbosity = Silent())
            end
            flush(io)
            read(path, String)
        end
        @test isempty(silent_out)

        lines = String[]
        reconstruct(acq, method; verbosity = Verbose(; printfunc = (s...) -> push!(lines, string(s...))))
        @test !isempty(lines)
    end

    # `IterativeReconstruction`'s own `maxit`/`reltol` win over the algorithm object's, and `nothing`
    # hands them back to it -- this pair is the regression test for the silent clobber that made
    # `algorithm = FISTA(maxit = ...)` unreachable.
    function iteration_numbers(method)
        seen = Int[]
        sink = (s...) -> begin
            m = match(r"^\s*(\d+)\s", string(s...))
            isnothing(m) || push!(seen, parse(Int, m[1]))
        end
        reconstruct(acq, method; verbosity = Verbose(; printfunc = sink, freq = 1, timing = false))
        return seen
    end

    @testset "maxit/reltol are method-owned" begin
        reg = L1Wavelet2D(1.0f-3)
        @test iteration_numbers(
            IterativeReconstruction(reg; algorithm = FISTA(maxit = 7), maxit = nothing, reltol = nothing)
        ) == collect(1:7)
        @test iteration_numbers(
            IterativeReconstruction(reg; algorithm = FISTA(maxit = 7), maxit = 4, reltol = 0)
        ) == collect(1:4)
        # keyword-only: there is no positional form
        @test_throws MethodError IterativeReconstruction(reg, 5)
    end

    @testset "direct methods honour maxit/reltol" begin
        # `POCS.reltol` used to be a dead field: a loose tolerance must now stop the loop early.
        loose = reconstruct(acq_pf, POCS(; maxit = 200, reltol = 1.0e-1); verbosity = Silent())
        tight = reconstruct(acq_pf, POCS(; maxit = 200, reltol = 0.0); verbosity = Silent())
        @test size(loose) == size(tight)
        @test norm(loose - tight) > 0

        # `PhaseConstrained`'s CG length was hardcoded at 25; it is now a field.
        pc_short = reconstruct(acq_pf, PhaseConstrained(; maxit = 1); verbosity = Silent())
        pc_long = reconstruct(acq_pf, PhaseConstrained(; maxit = 25); verbosity = Silent())
        @test norm(pc_short - pc_long) > 0
    end

    @testset "SPIRiT forwards maxit through lowering" begin
        mask = falses(Nx, Ny)
        mask[:, 1:2:end] .= true
        mask[:, 11:22] .= true
        acq_us = simulate_acquisition(
            img,
            AcquisitionInfo(
                nothing; is3D = false, image_size = (Nx, Ny), sensitivity_maps = smaps,
                subsampling = mask,
            ); inverse_crime_check = false, keep_sensitivity_maps = true
        )
        lowered = lower(SPIRiT(; calib_size = (16, 12), maxit = 3, iterative = true), acq_us)
        @test lowered isa IterativeReconstruction
        @test lowered.maxit == 3
    end

    @testset "one progress bar per reconstruct" begin
        cases = (
            (acq, DirectReconstruction(), false),                          # indeterminate indicator
            (acq_pf, POCS(; maxit = 5, reltol = 0.0), true),                # determinate
            (acq_pf, PhaseConstrained(; maxit = 5), true),                 # determinate
            (acq, IterativeReconstruction(L2Image(0.01f0); maxit = 5, reltol = 0), true),
        )
        for (src, method, determinate) in cases
            io = IOBuffer()
            reconstruct(src, method; verbosity = ProgressBar(output = io, dt = 0.0))
            s = String(take!(io))
            @test !isempty(s)
            # One meter draws one trailing newline when it is finished.
            @test count(==('\n'), s) == 1
            determinate && @test occursin("100%", s)
        end
    end

    @testset "task-split run gets a slice-level bar" begin
        nslices = 3
        ksp_ms = repeat(unname(acq.kspace_data), 1, 1, 1, nslices)
        smaps_ms = repeat(smaps, 1, 1, 1, nslices)
        acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)
        io = IOBuffer()
        x = reconstruct(
            acq_ms, IterativeReconstruction(L2Image(0.01f0); maxit = 3);
            verbosity = ProgressBar(output = io, dt = 0.0),
        )
        s = String(take!(io))
        @test size(x) == (Nx, Ny, nslices)
        # The slice bar outranks the per-method bar, so there is still exactly one meter.
        @test count(==('\n'), s) == 1
        @test occursin("100%", s)
    end
end

@testitem "SPIRiT: the frequency-domain kernel matches the calibration convention" tags = [:reconstruction, :regularization] begin
    using Test
    using Ristretto: _spirit_gfft, SPIRiTConsistencyOp
    using LinearAlgebra
    using Random

    # `_calibrate_spirit_kernel` fits a *correlation*: the target sample at `n` in coil `tgt` is
    #     Σ_{m, src} kernel[m, src, tgt] · k[n + m, src]
    # over the patch offsets `m` (`m = 0` at the kernel centre). `_spirit_gfft` has to reproduce
    # exactly that when its `Ĝ` is applied in the frequency domain, which needs (a) the kernel taps
    # reversed along both axes, because a frequency-domain product is a convolution `Σ h[m] k[n−m]`,
    # and (b) the `(src, tgt)` coil pair laid out so `Ĝ[:, :, src, tgt]` is the transform of
    # `kernel[:, :, src, tgt]`. Getting either wrong still yields a plausible-looking operator (it
    # stays a valid linear map, and the adjoint test below still passes) but applies the fitted
    # neighbourhood in the wrong orientation or mixes the coils the wrong way round, which is why
    # this is checked against an explicit reference sum rather than through a quality metric.
    Random.seed!(11)
    Kx, Ky, Nc, Nx, Ny = 5, 3, 3, 16, 12
    kernel = randn(ComplexF64, Kx, Ky, Nc, Nc)
    k = randn(ComplexF64, Nx, Ny, Nc)

    op = SPIRiTConsistencyOp(_spirit_gfft(kernel, Nx, Ny), Nx, Ny)
    Gk = k .- (op * k)                     # the operator is (I − G)

    px, py = Kx ÷ 2, Ky ÷ 2
    reference = zeros(ComplexF64, Nx, Ny, Nc)
    for tgt in 1:Nc, src in 1:Nc, jy in 1:Ky, ix in 1:Kx
        ox, oy = ix - 1 - px, jy - 1 - py
        for n2 in 1:Ny, n1 in 1:Nx
            reference[n1, n2, tgt] += kernel[ix, jy, src, tgt] * k[mod1(n1 + ox, Nx), mod1(n2 + oy, Ny), src]
        end
    end
    @test isapprox(Gk, reference; rtol = 1.0e-10)

    # Reversing the taps or transposing the coil pair on the way in must break the match — this is
    # what guards the two orientation conventions above against a silent regression.
    op_rev = SPIRiTConsistencyOp(_spirit_gfft(reverse(reverse(kernel, dims = 1), dims = 2), Nx, Ny), Nx, Ny)
    op_tr = SPIRiTConsistencyOp(_spirit_gfft(permutedims(kernel, (1, 2, 4, 3)), Nx, Ny), Nx, Ny)
    @test norm((k .- (op_rev * k)) - reference) / norm(reference) > 0.5
    @test norm((k .- (op_tr * k)) - reference) / norm(reference) > 0.5

    y = randn(ComplexF64, Nx, Ny, Nc)
    @test dot(op * k, y) ≈ dot(k, op' * y)
end

@testitem "SPIRiT: calibration regularization stabilizes a noisy kernel fit" tags = [:reconstruction, :acquisition] setup = [SyntheticCoils] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims
    using FFTW
    using Random

    Random.seed!(3)
    Nx, Ny, Nc = 32, 32, 4
    img = zeros(ComplexF64, Nx, Ny)
    img[8:24, 8:24] .= 1.0
    img[12:16, 18:22] .= 0.4
    sens = synthetic_sensitivities(ComplexF64, Nx, Ny, Nc; phase_scale = 0.6)
    full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        full[:, :, c] = fftshift(fft(img .* sens[:, :, c]))
    end
    # A noisy ACS is what makes the unregularized least-squares fit misbehave.
    full .+= (0.02 * norm(full) / sqrt(length(full))) .* randn(ComplexF64, size(full))

    mask_y = falses(Ny)
    mask_y[1:2:Ny] .= true
    mask_y[11:22] .= true
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(full[:, mask_y, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )

    obj = abs.(img) .> 0.2
    err(rec) = norm(abs.(unname(rec))[obj] .- abs.(img)[obj]) / norm(abs.(img)[obj])

    reg = reconstruct(acq, SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20); verbosity = Silent())
    plain = reconstruct(
        acq, SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20, calib_λ = 0.0);
        verbosity = Silent()
    )
    @test err(reg) < 0.1
    @test err(reg) < err(plain)
    @test_throws ArgumentError SPIRiT(calib_λ = -1.0)
end

@testitem "GRAPPA: check_applicable rejects unsupported sampling patterns" tags = [:reconstruction, :acquisition] setup = [SyntheticCoils] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using Ristretto: check_applicable
    using NamedDims
    using FFTW
    using Random

    Nx, Ny, Nc = 32, 32, 4
    img = zeros(ComplexF64, Nx, Ny)
    img[8:24, 8:24] .= 1.0
    sens = synthetic_sensitivities(ComplexF64, Nx, Ny, Nc)
    full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        full[:, :, c] = fftshift(fft(img .* sens[:, :, c]))
    end
    sens_named = NamedDimsArray{(:x, :y, :coil)}(sens)

    make_acq(mask_y) = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(full[:, mask_y, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y), sensitivity_maps = sens_named,
    )

    # Regular R = 2 plus a contiguous ACS block: applicable.
    regular = falses(Ny)
    regular[1:2:Ny] .= true
    regular[11:22] .= true
    @test check_applicable(GRAPPA(calib_size = (32, 12)), make_acq(regular)) === nothing

    # A random (variable-density) pattern has no fixed stride: GRAPPA's kernel assumes a regular
    # lattice, so this must be rejected rather than silently producing garbage.
    Random.seed!(7)
    random_mask = falses(Ny)
    random_mask[randperm(Ny)[1:16]] .= true
    random_mask[15:18] .= true
    err = try
        check_applicable(GRAPPA(calib_size = (32, 4)), make_acq(random_mask))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("regularly undersampled", sprint(showerror, err))

    # Regular stride but no fully sampled block at all: no ACS to calibrate on.
    no_acs = falses(Ny)
    no_acs[1:2:Ny] .= true
    @test_throws ArgumentError check_applicable(GRAPPA(), make_acq(no_acs))

    # ACS present but too short for the requested kernel.
    short_acs = falses(Ny)
    short_acs[1:3:Ny] .= true
    short_acs[16:17] .= true
    @test_throws ArgumentError check_applicable(GRAPPA(kernel_size = (4, 3)), make_acq(short_acs))

    # Fully sampled data has nothing to synthesize, but `subsampling === nothing` is still an error.
    acq_full = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(full); is3D = false, sensitivity_maps = sens_named,
    )
    @test_throws ArgumentError check_applicable(GRAPPA(), acq_full)
end

@testitem "device reconstructions hand their FFT plans back" tags = [:reconstruction, :gpu] setup = [GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo
    using Random

    # A reconstruction drops the encoding operator it built, and with it the operator's cuFFT
    # plans; their handles must be back in CUDA's cache when it returns, not left for the
    # garbage collector, or every later reconstruction plans its FFTs from scratch. A radial
    # iterative reconstruction also has the Toeplitz normal operator's plans, which the solver
    # asks the encoding operator for.
    Random.seed!(2)
    n, nc = 32, 4
    mask = falses(n)
    mask[1:2:end] .= true
    acq = CartesianAcquisitionInfo(
        randn(ComplexF32, n, count(mask), nc); is3D = false, image_size = (n, n),
        subsampling = (:, mask), sensitivity_maps = randn(ComplexF32, n, n, nc),
    )
    radial = NonCartesianAcquisitionInfo(
        randn(ComplexF32, 2n, 12, nc); trajectory = Float32.(parent(radial_trajectory(2n, 12))), image_size = (n, n),
        sensitivity_maps = randn(ComplexF32, n, n, nc),
    )
    methods = (
        DirectReconstruction(),
        IterativeReconstruction(; regularization = TotalVariation2D(1.0f-3), maxit = 3),
        IterativeReconstruction(; maxit = 3),
    )
    for backend in fft_backends()
        nameof(backend.array_type) === :CuArray || continue
        handles = Base.get_extension(Ristretto, :RistrettoCUDAExt).CUFFT.idle_handles
        for a in (acq, radial), method in methods
            dacq = to_device(backend, a)
            reconstruct(dacq, method; verbosity = Silent())
            active = length(handles.active_handles)
            reconstruct(dacq, method; verbosity = Silent())
            @test length(handles.active_handles) == active
        end
    end
end
