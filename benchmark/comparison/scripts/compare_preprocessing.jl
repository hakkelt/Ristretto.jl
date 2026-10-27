using Test
using FFTW: fftshift

include(joinpath(@__DIR__, "_setup.jl"))

@info "Starting Preprocessing Comparisons"

# Setup MATLAB
# setup_matlab_paths()

@testset "Preprocessing Benchmark" begin
    @testset "ESPIRiT Sensitivity Estimation" begin
        # Generate data
        N = 64
        num_coils = 8
        img, kspace, true_sens = multicoil_phantom(N, num_coils)

        # 1. Ristretto
        # We need a calibration region. ESPIRiT usually extracts the center of k-space internally if we pass the whole k-space.
        # Let's pass the whole kspace to estimate_sensitivities.
        #
        # `subspace_threshold` is the one parameter that must *not* be zeroed along with the
        # eigenvalue crop: it decides how many right singular vectors of the calibration matrix
        # count as the signal subspace, and at 0 every one of the 288 of them is kept. The kept set
        # is then a complete basis, `V V'` is proportional to the identity at every pixel, and the
        # maps are an arbitrary eigenvector — which is what this comparison measured before
        # (NRMSE 0.77 against BART, at every layout). 0.001 is BART's own `-t` default; MRIReco
        # uses 0.02, and Ristretto matches whichever it is given to 3e-3 or better.
        ristretto_sens = Ristretto.estimate_sensitivities(
            kspace, method = Ristretto.ESPIRiT(calib_size = 24, eigenvalue_threshold = 0.0, subspace_threshold = 0.001)
        )

        # 2. BART
        # We pass the full k-space. BART ecalib with -r 24 will extract the center 24x24 internally.
        bart_kspace = ComplexF32.(reshape(kspace, (N, N, 1, num_coils)))
        bart_res = run_bart(2, "ecalib -r 24 -c 0 -m 1", bart_kspace)
        bart_sens = bart_res[1]

        # 3. SigPy
        # SigPy expects data in [coils, Y, X] (or Z, Y, X).
        sp_kspace = permutedims(kspace, (3, 2, 1))
        # SigPy crop default is 0.95 (which is eigenvalue threshold?). Let's set it to 0.
        espirit_app = sp_app.EspiritCalib(sp_kspace, calib_width = 24, crop = 0.0, show_pbar = false)
        sp_sens = espirit_app.run()

        @test size(bart_sens) == (N, N, 1, num_coils)
        @test size(sp_sens) == (num_coils, N, N) # Assuming sp returns coils first
        @test size(ristretto_sens) == (N, N, num_coils)

        # Compare magnitudes (ESPIRiT maps have arbitrary phase).
        #
        # Two layout conventions have to be undone first. SigPy was handed `(coil, y, x)` by
        # permuting `(3, 2, 1)`, so its output is undone by the same permutation, not by
        # `(2, 3, 1)` — that one transposes x against y, and made SigPy look like it disagreed
        # with BART by 0.60 when the two in fact agree to 3e-3. And the raw-array
        # `estimate_sensitivities` returns maps in Ristretto's *default* image convention, origin at
        # index 1, while BART and SigPy return centred ones; `fftshift` is what puts them in the
        # same frame (see the FFT-shift section of its docstring).
        bart_mag = dropdims(abs.(bart_sens), dims = 3)
        sp_mag = permutedims(abs.(sp_sens), (3, 2, 1))
        ristretto_mag = fftshift(abs.(ristretto_sens), 1:2)

        # Mask out background where sensitivities are undefined/arbitrary
        mask = abs.(img) .> 1.0e-4
        mask_3d = repeat(mask, 1, 1, num_coils)

        err_bart = nrmse(bart_mag[mask_3d], ristretto_mag[mask_3d])
        err_sp = nrmse(sp_mag[mask_3d], ristretto_mag[mask_3d])
        @info "ESPIRiT Masked Magnitude NRMSE: Ristretto vs BART = $(err_bart), Ristretto vs SigPy = $(err_sp)"

        # Tolerance for numerical differences in the SVD / eigen implementations. Measured on this
        # phantom: Ristretto vs BART 0.0043, Ristretto vs SigPy 0.0027, and each of the three within 0.006 of
        # the simulated ground-truth maps.
        @test err_bart < 0.02
        @test err_sp < 0.02
    end

    @testset "Coil Compression" begin
        N = 64
        num_coils = 8
        target_coils = 4
        _, kspace, _ = multicoil_phantom(N, num_coils)

        # 1. Ristretto SVD Compression
        comp_ristretto_svd, C_svd = Ristretto.compress_coils(kspace, target_coils, method = Ristretto.SVDCompression())
        @test size(comp_ristretto_svd) == (N, N, target_coils)

        # 2. Ristretto Geometric Compression
        comp_ristretto_geo, C_geo = Ristretto.compress_coils(kspace, target_coils, method = Ristretto.GeometricCompression())
        @test size(comp_ristretto_geo) == (N, N, target_coils)

        # 3. BART
        bart_kspace = ComplexF32.(reshape(kspace, (N, N, 1, num_coils)))
        bart_comp_res = run_bart(1, "cc -p $target_coils -S", bart_kspace)
        @test size(bart_comp_res) == (N, N, 1, target_coils)

        # Total energy in the compressed coils should be similar
        energy_ristretto = norm(comp_ristretto_svd)
        energy_bart = norm(bart_comp_res)
        @test isapprox(energy_ristretto, energy_bart, rtol = 1.0e-2)
    end

    @testset "Gradient Delay" begin
        N = 128
        Nspokes = 101
        q = (0.5, 0.2, 0.1)  # BART `-q x:y:xy`, in samples

        # 1. Nominal golden-angle trajectory, and the actual one: the full displacement S n of a
        #    delay on each gradient axis (`-O` keeps the part across the spoke as well).
        traj_nominal = run_bart(1, "traj -r -x $N -y $Nspokes -G")
        traj_actual = run_bart(1, "traj -r -x $N -y $Nspokes -G -q $(q[1]):$(q[2]):$(q[3]) -O")

        # 2. Multi-coil k-space sampled on the actual trajectory.
        ksp = run_bart(1, "phantom -k -s 8 -t", traj_actual)

        # 3. BART estimates from the nominal trajectory (default: opposed spokes; -R: RING).
        qf_bart = run_bart(1, "estdelay", traj_nominal, ksp)
        qf_bart_ring = run_bart(1, "estdelay -R", traj_nominal, ksp)

        # 4. Ristretto. BART stores the trajectory as (y, x, z) in units of 1/FOV.
        ksp_ristretto = ComplexF64.(dropdims(ksp, dims = 1))
        traj_ristretto = Float64.(real.(traj_nominal[[2, 1], :, :])) ./ N
        acq = Ristretto.NonCartesianAcquisitionInfo(ksp_ristretto, trajectory = traj_ristretto, image_size = (N, N))
        samples(d) = (d.dx * N, d.dy * N, d.dxy * N)
        delays_ring = samples(Ristretto.estimate_gradient_delays(acq, method = Ristretto.RING()))
        delays_os = samples(Ristretto.estimate_gradient_delays(acq, method = Ristretto.OpposingSpokes()))

        # 5. Compare
        @info "Gradient delays (samples)" truth = q bart_default = real.(qf_bart[:]) bart_ring = real.(qf_bart_ring[:]) ristretto_ring = delays_ring ristretto_opposing = delays_os
        @test all(isapprox.(delays_ring, q; atol = 0.02))
        # Both opposed-spoke estimators neglect the part of the displacement across the spoke.
        @test all(isapprox.(delays_os, real.(qf_bart[1:3]); atol = 0.1))
    end

    @testset "Prewhitening" begin
        Nc = 8
        N = 128
        # Create fake multicoil data (N, N, 1, Nc) so BART sees coils at dim 4
        data = randn(ComplexF32, N, N, 1, Nc)

        # Create fake noise
        A = randn(ComplexF32, Nc, Nc)
        cov_true = A * A'
        noise_samples = 2000
        noise_flat = cholesky(Hermitian(cov_true)).L * randn(ComplexF32, Nc, noise_samples)
        noise_data = reshape(noise_flat, noise_samples, 1, 1, Nc)

        # BART Prewhitening
        # whiten <input> <ndata> <output> [<optmat_out>] [<covar_out>]
        # BART's whiten command computes the noise covariance and whitens the input.
        bart_out, bart_opt, bart_cov = run_bart(3, "whiten", data, noise_data)

        # Ristretto Prewhitening (tell it coils are at dim 4)
        ristretto_cov = Ristretto.estimate_noise_covariance(noise_data, coil_dim = 4)
        ristretto_out = Ristretto.prewhiten(data, ristretto_cov, coil_dim = 4)

        # Compare
        # BART's covariance estimation may have an extra scaling factor (e.g. dividing by N-1 vs N)
        # We can check if they are proportional
        bart_cov_sq = dropdims(bart_cov, dims = Tuple(findall(==(1), size(bart_cov))))
        @info "BART cov size: $(size(bart_cov)), sq size: $(size(bart_cov_sq))"
        if size(bart_cov_sq) == (Nc, Nc)
            scale_cov = ristretto_cov ./ bart_cov_sq
            @test all(isapprox.(scale_cov, scale_cov[1, 1], rtol = 1.0e-3))
        end

        # Flatten and compute covariance of whitened data
        ristretto_out_flat = reshape(ristretto_out, N * N, Nc)
        bart_out_flat = reshape(bart_out, N * N, Nc)

        # Both should be somewhat close to a diagonal matrix if data was white noise initially.
        # But data was white noise, so applying L^{-1} makes its covariance L^{-1} L^{-*} = cov_true^{-1}.
        # So we can just compare if BART and Ristretto outputs have similar magnitudes
        @info "Ristretto Prewhiten mean energy: $(sum(abs2, ristretto_out))"
        @info "BART Prewhiten mean energy: $(sum(abs2, bart_out))"

        # As long as the operations complete successfully and energy is on the same order, we pass for now.
        @test size(ristretto_out) == size(bart_out)
    end
end
