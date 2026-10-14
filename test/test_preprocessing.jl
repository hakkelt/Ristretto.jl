@testitem "Prewhitening and noise covariance estimation" tags = [:acquisition, :encoding, :gpu] setup = [GpuEnvSetup, GpuHelpers] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims

    Nc = 4
    Nx, Ny = 16, 16
    # Synthesize non-trivial positive definite noise covariance
    A = randn(ComplexF32, Nc, Nc)
    Ψ_true = A * A' + 0.2f0 * I

    L_true = cholesky(Hermitian(Ψ_true)).L
    Nsamples = 10000
    noise_raw = L_true * randn(ComplexF32, Nc, Nsamples)
    noise_named = NamedDimsArray{(:kx, :ky, :coil)}(
        reshape(permutedims(reshape(noise_raw, Nc, Nx, :), (2, 3, 1)), Nx, :, Nc)
    )

    # Estimate covariance
    Ψ_est = estimate_noise_covariance(noise_named)
    @test size(Ψ_est) == (Nc, Nc)
    @test isapprox(Ψ_est, Ψ_true; rtol = 0.1)

    # Prewhiten acquisition
    img = NamedDimsArray{(:x, :y)}(randn(ComplexF32, Nx, Ny))
    sens = NamedDimsArray{(:x, :y, :coil)}(randn(ComplexF32, Nx, Ny, Nc))
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false,
        sensitivity_maps = sens,
    )
    acq_sim = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
    # Add correlated noise to k-space
    ksp_flat = reshape(permutedims(unname(acq_sim.kspace_data), (3, 1, 2)), Nc, :)
    ksp_noisy_flat = ksp_flat + L_true * randn(ComplexF32, Nc, size(ksp_flat, 2))
    ksp_noisy = NamedDimsArray{(:kx, :ky, :coil)}(
        permutedims(reshape(ksp_noisy_flat, Nc, Nx, Ny), (2, 3, 1))
    )
    acq_noisy = CartesianAcquisitionInfo(acq_sim; kspace_data = ksp_noisy)

    acq_white = prewhiten(acq_noisy, Ψ_est)
    @test acq_white isa CartesianAcquisitionInfo
    @test acq_white.kspace_data isa NamedDimsArray
    @test dimnames(acq_white.kspace_data) == (:kx, :ky, :coil)
    @test acq_white.sensitivity_maps isa NamedDimsArray
    @test dimnames(acq_white.sensitivity_maps) == (:x, :y, :coil)

    test_on_devices(a -> prewhiten(a, Ψ_est), acq_noisy; backends = all_backends())
end

@testitem "Coil compression with SVDCompression" tags = [:acquisition, :encoding, :gpu] setup = [SyntheticCoils, GpuEnvSetup, GpuHelpers] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims

    Nc = 8
    Nv = 4
    Nx, Ny = 16, 16
    img = NamedDimsArray{(:x, :y)}(randn(ComplexF32, Nx, Ny))

    sens = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false,
        sensitivity_maps = sens,
    )
    acq_sim = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

    acq_comp, C = compress_coils(acq_sim, Nv; method = SVDCompression())
    @test size(C) == (Nv, Nc)
    @test size(acq_comp.kspace_data, :coil) == Nv
    @test size(acq_comp.sensitivity_maps, :coil) == Nv
    @test dimnames(acq_comp.kspace_data) == (:kx, :ky, :coil)
    @test dimnames(acq_comp.sensitivity_maps) == (:x, :y, :coil)

    # Direct reconstruction from compressed data
    rec_orig = reconstruct(acq_sim, DirectReconstruction(); verbosity = Silent())
    rec_comp = reconstruct(acq_comp, DirectReconstruction(); verbosity = Silent())
    @test isapprox(rec_comp, rec_orig; rtol = 0.05)

    # GeometricCompression
    acq_geom, C_geom = compress_coils(acq_sim, Nv; method = GeometricCompression())
    @test size(C_geom) == (Nv, Nc, Nx)
    @test size(acq_geom.kspace_data, :coil) == Nv
    @test size(acq_geom.sensitivity_maps, :coil) == Nv
    rec_geom = reconstruct(acq_geom, DirectReconstruction(); verbosity = Silent())
    @test isapprox(rec_geom, rec_orig; rtol = 0.05)

    for method in (SVDCompression(), GeometricCompression())
        test_on_devices(a -> compress_coils(a, Nv; method), acq_sim; backends = all_backends())
    end
    test_on_devices(k -> Ristretto.compress_coils_with_matrix(k, C), acq_sim.kspace_data; backends = all_backends())
end

@testitem "Sensitivity map estimation: SelfCalibrating, AdaptiveCombine, ESPIRiT" tags = [:acquisition, :encoding, :simulation, :gpu] setup = [SyntheticCoils, GpuEnvSetup, GpuHelpers] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nc = 32, 32, 4
    img = NamedDimsArray{(:x, :y)}(zeros(ComplexF32, Nx, Ny))
    img[8:24, 8:24] .= 1.0f0

    sens_true = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false,
        sensitivity_maps = sens_true,
    )
    acq_sim = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

    # 1. SelfCalibrating
    acq_selfcal = estimate_sensitivities(acq_sim; method = SelfCalibrating(calib_size = 16))
    @test !isnothing(acq_selfcal.sensitivity_maps)
    @test size(acq_selfcal.sensitivity_maps) == (Nx, Ny, Nc)
    @test dimnames(acq_selfcal.sensitivity_maps) == (:x, :y, :coil)

    # 2. AdaptiveCombine
    acq_adaptive = estimate_sensitivities(acq_sim; method = AdaptiveCombine(kernel_size = 5))
    @test !isnothing(acq_adaptive.sensitivity_maps)
    @test size(acq_adaptive.sensitivity_maps) == (Nx, Ny, Nc)
    @test dimnames(acq_adaptive.sensitivity_maps) == (:x, :y, :coil)

    # 3. ESPIRiT
    acq_espirit = estimate_sensitivities(acq_sim; method = ESPIRiT(calib_size = 16, kernel_size = 6))
    @test !isnothing(acq_espirit.sensitivity_maps)
    @test size(acq_espirit.sensitivity_maps) == (Nx, Ny, Nc)
    @test dimnames(acq_espirit.sensitivity_maps) == (:x, :y, :coil)

    # Verify phase-aligned sensitivity map accuracy in object support
    mask = abs.(unname(img)) .> 0.5
    for sens_est in (acq_selfcal.sensitivity_maps, acq_adaptive.sensitivity_maps, acq_espirit.sensitivity_maps)
        dot_prod = sum(unname(sens_est) .* conj(unname(sens_true)), dims = 3)
        phase_diff = cis.(-angle.(dot_prod))
        aligned = unname(sens_est) .* phase_diff
        rel_err = norm(aligned[mask, :] - unname(sens_true)[mask, :]) / norm(unname(sens_true)[mask, :])
        @test rel_err < 0.05
    end

    # Verify direct reconstruction magnitude with estimated maps
    rec_selfcal = reconstruct(acq_selfcal, DirectReconstruction(); verbosity = Silent())
    rec_adaptive = reconstruct(acq_adaptive, DirectReconstruction(); verbosity = Silent())
    rec_espirit = reconstruct(acq_espirit, DirectReconstruction(); verbosity = Silent())

    @test isapprox(abs.(unname(rec_selfcal))[mask], abs.(unname(img))[mask]; rtol = 0.15)
    @test isapprox(abs.(unname(rec_adaptive))[mask], abs.(unname(img))[mask]; rtol = 0.15)
    @test isapprox(abs.(unname(rec_espirit))[mask], abs.(unname(img))[mask]; rtol = 0.15)

    # Estimation runs on a host copy of a device acquisition; the maps come back on the device.
    for method in (SelfCalibrating(calib_size = 16), ESPIRiT(calib_size = 16, kernel_size = 6))
        test_on_devices(a -> estimate_sensitivities(a; method), acq_sim; backends = all_backends())
    end
end

@testitem "Sensitivity estimation: batch dimensions get their own maps" tags = [:preprocessing, :acquisition, :simulation] setup = [SyntheticCoils] begin
    using Test
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo
    using LinearAlgebra: norm
    using NamedDims

    # Every real multi-slice or cine acquisition arrives with a batch dimension after the coil
    # one; coil sensitivities differ from slice to slice, so each slab is estimated separately
    # and the maps come back in the same layout as the k-space.
    Nx, Ny, Nc, Nz = 32, 32, 4, 3
    sens_true = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    slabs = map(1:Nz) do z
        img = NamedDimsArray{(:x, :y)}(zeros(ComplexF32, Nx, Ny))
        img[(6 + z):(22 + z), 8:24] .= 1.0f0
        acq = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
            is3D = false, sensitivity_maps = sens_true,
        )
        return simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
    end

    ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(
        cat((unname(s.kspace_data) for s in slabs)...; dims = 4)
    )
    acq_ms = CartesianAcquisitionInfo(ksp; is3D = false)

    acq_est = estimate_sensitivities(acq_ms; method = SelfCalibrating(calib_size = 16))
    maps = acq_est.sensitivity_maps
    @test dimnames(maps) == (:x, :y, :coil, :z)
    @test size(maps) == (Nx, Ny, Nc, Nz)

    # Each slab's maps must equal what estimating that slab on its own gives.
    for z in 1:Nz
        alone = estimate_sensitivities(
            slabs[z].kspace_data; method = SelfCalibrating(calib_size = 16),
            image_size = (Nx, Ny),
        )
        @test unname(maps)[:, :, :, z] == unname(alone)
    end

    # And they reconstruct: one coil-combined image per slab, close to the object.
    rec = reconstruct(acq_est, DirectReconstruction(); verbosity = Silent())
    @test dimnames(rec) == (:x, :y, :z)
    for z in 1:Nz
        img = zeros(ComplexF32, Nx, Ny)
        img[(6 + z):(22 + z), 8:24] .= 1.0f0
        mask = abs.(img) .> 0.5
        @test isapprox(abs.(unname(rec)[:, :, z])[mask], abs.(img)[mask]; rtol = 0.15)
    end
end

@testitem "Sensitivity estimation: non-Cartesian data calibrates through gridding" tags = [:preprocessing, :acquisition, :nfft, :gpu] setup = [RadialCalibration, GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using NamedDims

    case = radial_case()
    acq = NonCartesianAcquisitionInfo(
        case.kspace; trajectory = case.traj, image_size = (case.N, case.N)
    )

    for method in (ESPIRiT(calib_size = 24, kernel_size = 6), SelfCalibrating(calib_size = 24))
        maps = estimate_sensitivities(acq; method).sensitivity_maps
        @test maps isa NamedDimsArray
        @test dimnames(maps) == (:x, :y, :coil)
        @test size(maps) == (case.N, case.N, case.ncoil)
        # The maps must sit on the same (centred) image grid the NFFT adjoint reconstructs onto,
        # which is what makes them agree with the simulated ones pixel by pixel.
        @test map_alignment(maps, case.smaps, case.mask) > 0.99
    end

    # ... and the maps are good enough to reconstruct with: a SENSE reconstruction of a flat
    # object is flat inside it and dark outside.
    acq_maps = estimate_sensitivities(acq; method = ESPIRiT(calib_size = 24, kernel_size = 6))
    rec = abs.(unname(reconstruct(density_compensation(acq_maps))))
    inside = rec[case.mask]
    @test maximum(abs.(inside ./ (sum(inside) / length(inside)) .- 1)) < 0.25
    @test (sum(rec[.!case.mask]) / count(.!case.mask)) < 0.1 * (sum(inside) / length(inside))

    # Unnamed k-space grids through the plain-array operator and returns plain maps.
    acq_unnamed = NonCartesianAcquisitionInfo(
        unname(case.kspace); trajectory = unname(case.traj), image_size = (case.N, case.N)
    )
    maps_unnamed = estimate_sensitivities(
        acq_unnamed; method = ESPIRiT(calib_size = 24, kernel_size = 6)
    ).sensitivity_maps
    @test !(maps_unnamed isa NamedDimsArray)
    @test size(maps_unnamed) == (case.N, case.N, case.ncoil)
    @test map_alignment(maps_unnamed, case.smaps, case.mask) > 0.99

    # Calibrated on a host copy, then a SENSE reconstruction on the device's NFFT with those maps.
    test_on_devices(a -> estimate_sensitivities(a; method = ESPIRiT(calib_size = 24, kernel_size = 6)), acq; backends = all_backends())
    test_on_devices(a -> reconstruct(density_compensation(a); verbosity = Silent()), acq_maps)
end

@testitem "Sensitivity estimation: non-Cartesian batch dimensions and average_dims" tags = [:preprocessing, :acquisition, :nfft] setup = [RadialCalibration] begin
    using Test
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using NamedDims

    case = radial_case()
    nframes = 3
    ksp_t = NamedDimsArray{(:sample, :spoke, :coil, :time)}(
        repeat(unname(case.kspace), 1, 1, 1, nframes)
    )
    acq = NonCartesianAcquisitionInfo(
        ksp_t; trajectory = case.traj, image_size = (case.N, case.N)
    )
    method = ESPIRiT(calib_size = 24, kernel_size = 6)

    # `:time` is averaged over by default, so one set of maps comes back for the whole series.
    averaged = estimate_sensitivities(acq; method).sensitivity_maps
    @test dimnames(averaged) == (:x, :y, :coil)
    @test map_alignment(averaged, case.smaps, case.mask) > 0.99

    # `average_dims = ()` estimates every frame on its own, in the k-space's own layout.
    per_frame = estimate_sensitivities(acq; method, average_dims = ()).sensitivity_maps
    @test dimnames(per_frame) == (:x, :y, :coil, :time)
    @test size(per_frame, 4) == nframes
    for t in 1:nframes
        @test map_alignment(unname(per_frame)[:, :, :, t], case.smaps, case.mask) > 0.99
    end

    # A spatial or coil axis is not a batch dimension, and density compensation is not optional.
    @test_throws ArgumentError estimate_sensitivities(acq; method, average_dims = (:x,))
    @test_throws ArgumentError estimate_sensitivities(acq; method, dcf = nothing)
end

@testitem "Sensitivity estimation: ESPIRiT maps are phase-referenced to the first coil" tags = [:preprocessing, :acquisition] begin
    using Test
    using Ristretto
    using NamedDims
    using FFTW: fft, fftshift, ifftshift

    # An eigenvector is defined only up to a phase, and LAPACK's choice varies from pixel to
    # pixel, so without a gauge the maps — and the phase of every image reconstructed with them —
    # would be arbitrary.
    N, ncoil = 32, 4
    img = zeros(ComplexF32, N, N)
    img[8:24, 10:22] .= 1
    smaps = ComplexF32.(coil_sensitivities(N, N, ncoil))
    coil_imgs = img .* smaps
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(
        ComplexF32.(fftshift(fft(ifftshift(coil_imgs, (1, 2)), (1, 2)), (1, 2)))
    )

    maps = estimate_sensitivities(
        AcquisitionInfo(ksp; is3D = false, shifted_image_dims = (:x, :y));
        method = ESPIRiT(calib_size = 16, kernel_size = 6),
    ).sensitivity_maps
    first_coil = unname(maps)[:, :, 1]
    nonzero = abs.(first_coil) .> 1.0f-3
    @test any(nonzero)
    @test maximum(abs.(angle.(first_coil[nonzero]))) < 1.0f-4
end

@testitem "Sensitivity estimation: all-zero maps are reported, not returned silently" tags = [:preprocessing, :acquisition] begin
    using Test
    using Ristretto
    using NamedDims

    # K-space whose signal sits away from the centre of the encoded matrix — a file whose
    # `center_sample`/encoding limits do not match where the echo actually is (CMRxRecon-300's
    # test set, for one) — leaves the calibration window empty. The maps then come out all zero,
    # which reconstructs to an all-zero image and, through an iterative solver, to NaNs.
    Nx, Ny, Nc = 64, 64, 2
    ksp = zeros(ComplexF32, Nx, Ny, Nc)
    ksp[5:12, 5:12, :] .= 1.0f0     # signal in a corner, nothing in the centre
    named = NamedDimsArray{(:kx, :ky, :coil)}(ksp)

    maps = @test_logs (:warn, r"all-zero maps") match_mode = :any estimate_sensitivities(
        named; method = SelfCalibrating(calib_size = 16)
    )
    @test iszero(unname(maps))

    # Signal in the centre: estimated normally, no warning.
    centred = zeros(ComplexF32, Nx, Ny, Nc)
    centred[29:36, 29:36, :] .= 1.0f0
    quiet = @test_logs estimate_sensitivities(
        NamedDimsArray{(:kx, :ky, :coil)}(centred); method = SelfCalibrating(calib_size = 16)
    )
    @test !iszero(unname(quiet))
end

@testitem "Sensitivity estimation: measured k-space smaller than image_size is zero-padded" tags = [:preprocessing, :acquisition] setup = [SyntheticCoils] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using NamedDims

    Nx, Ny, Nc, measured = 64, 64, 4, 41
    img = NamedDimsArray{(:x, :y)}(zeros(ComplexF32, Nx, Ny))
    img[16:48, 16:48] .= 1.0f0
    sens_true = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false, sensitivity_maps = sens_true,
    )
    acq_sim = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

    # Only the central `measured` phase-encode lines are kept, mirroring a real acquisition where
    # `kspace_data` stores just the measured extent rather than a zero-filled `image_size` grid.
    lo = (Ny - measured) ÷ 2 + 1
    ksp_measured = unname(acq_sim.kspace_data)[:, lo:(lo + measured - 1), :]
    acq_measured = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp_measured); is3D = false, image_size = (Nx, Ny),
    )

    # Previously threw an ArgCheck size-mismatch when re-attaching maps sized (Nx, measured, Nc).
    acq_out = estimate_sensitivities(acq_measured; method = ESPIRiT(calib_size = 24, kernel_size = 6))
    @test size(acq_out.sensitivity_maps) == (Nx, Ny, Nc)

    # The raw-array method accepts the same `image_size` keyword directly, and is a no-op without it.
    padded = estimate_sensitivities(
        ksp_measured; method = ESPIRiT(calib_size = 24, kernel_size = 6), image_size = (Nx, Ny)
    )
    @test size(padded) == (Nx, Ny, Nc)
    unpadded = estimate_sensitivities(ksp_measured; method = ESPIRiT(calib_size = 24, kernel_size = 6))
    @test size(unpadded) == (Nx, measured, Nc)
end

@testitem "Sensitivity estimation follows the acquisition's shifted_image_dims" tags = [:preprocessing, :acquisition, :reconstruction] setup = [SyntheticCoils] begin
    using Ristretto: CartesianAcquisitionInfo
    using Test
    using Ristretto
    using NamedDims: NamedDimsArray, unname
    using FFTW: fft, fftshift, ifftshift
    using LinearAlgebra: norm

    # Scanner data reconstructs in the CENTERED image convention (`shifted_image_dims`), while
    # every estimator inverts k-space into Ristretto's plain-DFT default. Maps handed back for such an
    # acquisition must therefore be shifted onto the same grid — otherwise they are rolled by half
    # the FOV relative to the images they multiply, and the reconstruction is not merely displaced
    # but wrong everywhere.
    Nx, Ny, Nc = 48, 48, 4
    truth = zeros(ComplexF32, Nx, Ny)
    truth[14:22, 20:32] .= 1                     # off-centre, so a half-FOV roll is unambiguous
    truth[18, 26] = 3
    sens_true = synthetic_sensitivities(ComplexF32, Nx, Ny, Nc)

    coil_images = sens_true .* reshape(truth, Nx, Ny, 1)
    # Centred image, centred k-space: the scanner's convention on both sides.
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(
        ComplexF32.(fftshift(fft(ifftshift(coil_images, (1, 2)), (1, 2)), (1, 2)))
    )

    acq_plain = CartesianAcquisitionInfo(ksp; is3D = false)
    acq_shifted = CartesianAcquisitionInfo(ksp; is3D = false, shifted_image_dims = (:x, :y))

    for method in (SelfCalibrating(calib_size = 24), ESPIRiT(calib_size = 24, kernel_size = 6))
        plain = unname(estimate_sensitivities(acq_plain; method).sensitivity_maps)
        shifted = unname(estimate_sensitivities(acq_shifted; method).sensitivity_maps)
        # Same k-space, same estimator: the only difference is which image grid the maps live on.
        @test shifted ≈ fftshift(plain, (1, 2))
        @test !(shifted ≈ plain)                 # the shift is not a no-op at this size

        # And the maps must actually explain the coil images they were estimated from.
        ρ = dropdims(sum(conj.(shifted) .* coil_images; dims = 3); dims = 3)
        residual = norm(shifted .* reshape(ρ, Nx, Ny, 1) - coil_images) / norm(coil_images)
        @test residual < 0.15
    end

    # End to end: the object must come back where it was put.
    acq = estimate_sensitivities(acq_shifted; method = SelfCalibrating(calib_size = 24))
    x = abs.(unname(reconstruct(acq; verbosity = Silent())))
    @test Tuple(argmax(x)) == (18, 26)
    α = sum(x .* abs.(truth)) / sum(abs2, x)
    @test norm(α .* x - abs.(truth)) / norm(abs.(truth)) < 0.15
end

@testitem "Sensitivity estimation: coil axis need not be trailing" tags = [:preprocessing] begin
    using Test
    using Ristretto
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nc = 32, 32, 4
    ksp = randn(ComplexF64, Nx, Ny, Nc)

    s_trailing = estimate_sensitivities(ksp; method = SelfCalibrating(), coil_dim = 3)
    s_leading = estimate_sensitivities(permutedims(ksp, (3, 1, 2)); method = SelfCalibrating(), coil_dim = 1)
    @test permutedims(s_leading, (2, 3, 1)) ≈ s_trailing

    # NamedDims with a non-trailing coil axis
    kn = NamedDimsArray{(:coil, :kx, :ky)}(permutedims(ksp, (3, 1, 2)))
    sn = estimate_sensitivities(kn; method = SelfCalibrating())
    @test dimnames(sn) == (:coil, :x, :y)
    @test permutedims(unname(sn), (2, 3, 1)) ≈ s_trailing
end

@testitem "Sensitivity normalization makes the coil sum of squares one" tags = [
    :preprocessing, :acquisition,
] begin
    using LinearAlgebra, Random

    Random.seed!(11)
    nx, ny, nc = 12, 10, 4
    maps = randn(ComplexF32, nx, ny, nc)
    maps[1:3, :, :] .= 0   # a region outside the object, where the sum of squares is zero

    normalized = normalize_sensitivity_maps(maps)
    total = dropdims(sum(abs2, normalized; dims = 3), dims = 3)
    @test eltype(normalized) === ComplexF32
    @test all(isapprox(1), total[4:end, :])
    @test all(iszero, normalized[1:3, :, :])
    # The scaling is a real, positive, per-voxel factor, so what the maps encode — the relative
    # magnitude and phase between coils — is unchanged.
    ratio = normalized[4, 1, :] ./ maps[4, 1, :]
    @test all(r -> isapprox(r, ratio[1]), ratio)
    @test real(ratio[1]) > 0
    # Real up to the round-off of forming the ratio itself; the factor applied was real.
    @test all(r -> abs(imag(r)) < 1.0e-6 * abs(real(r)), ratio)
    # Non-mutating: the input still has its own arbitrary scale.
    @test !all(isapprox(1), dropdims(sum(abs2, maps; dims = 3), dims = 3)[4:end, :])

    # A voxel far below the peak keeps its zeros rather than being divided by noise.
    faint = copy(maps)
    faint[5, 5, :] .*= 1.0f-4
    @test all(iszero, normalize_sensitivity_maps(faint; threshold = 1.0e-3)[5, 5, :])
    @test !all(iszero, normalize_sensitivity_maps(faint; threshold = 0)[5, 5, :])

    # The coil axis is found by name as well as by position.
    named = NamedDimsArray{(:x, :y, :coil)}(copy(maps))
    @test dimnames(normalize_sensitivity_maps(named)) == (:x, :y, :coil)
    @test unname(normalize_sensitivity_maps(named)) ≈ normalized
end

@testitem "Normalized sensitivity maps make the encoding operator a contraction" tags = [
    :preprocessing, :acquisition, :encoding,
] begin
    using LinearAlgebra, Random
    using Ristretto: get_encoding_operator
    import Ristretto.AbstractOperators as AbstractOperators

    Random.seed!(3)
    nx, ny, nc = 16, 16, 4
    maps = NamedDimsArray{(:x, :y, :coil)}(randn(ComplexF32, nx, ny, nc))
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(randn(ComplexF32, nx, ny, nc))
    info = AcquisitionInfo(ksp; sensitivity_maps = maps, image_size = (nx, ny))

    normalized = normalize_sensitivity_maps(info)
    @test info.sensitivity_maps === maps   # the input is left untouched

    # Fully sampled Cartesian SENSE over normalized maps has ‖𝒜‖ = 1 exactly, which is what lets
    # the step size come from a closed-form bound instead of a power iteration.
    E = get_encoding_operator(normalized; threaded = false)
    @test AbstractOperators.estimate_opnorm(E; maxit = 500) ≈ 1 atol = 1.0e-5
    @test AbstractOperators.estimate_opnorm(
        get_encoding_operator(info; threaded = false); maxit = 500
    ) > 1.5

    # An acquisition without maps is returned as it is, not rejected.
    plain = AcquisitionInfo(ksp[:, :, 1]; image_size = (nx, ny))
    @test normalize_sensitivity_maps(plain) === plain
end
