@testitem "2D Reconstruction Pipeline" tags = [:reconstruction, :integration, :gpu] setup = [TestHelpers, GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using Random

    @testset "2D Reconstruction Pipeline" begin
        @testset "Fully-sampled without regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; verbosity = Silent()))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1.0e-3
        end

        @testset "Undersampled with L2Image regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.001); maxit = 100); verbosity = Silent()),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.4  # measured ≈0.20
        end

        @testset "Undersampled with L1Wavelet regularization" begin
            # The sampling pattern is drawn at random, and the achievable error genuinely varies
            # with the draw: over 20 seeds this lands between 0.178 and 0.331, so an unseeded run
            # cleared the 0.3 bound only about 85% of the time. The reconstruction is converged by
            # maxit = 50 (200 and 1000 give the same answer), so the spread is the pattern, not the
            # solver. Fix the draw so the bound tests the reconstruction rather than the dice.
            Random.seed!(20260829)
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(0.005); maxit = 50); verbosity = Silent()),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.3
        end

        @testset "Multiple regularizations" begin
            # Seeded for the same reason as the L1Wavelet case above.
            Random.seed!(20260829)
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(
                    acq_with_data,
                    IterativeReconstruction(L1Wavelet2D(0.003), TotalVariation2D(0.001); maxit = 100); verbosity = Silent()
                ),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            # Tight enough to catch a sign-flipped solution (which lands at ≈2.0); measured ≈0.02.
            # The sampling pattern is drawn randomly per run, so the bound keeps a wide margin.
            @test error_norm < 0.3

            test_on_devices(acq_with_data; rtol = 1.0e-3) do a
                reconstruct(a, IterativeReconstruction(L1Wavelet2D(0.003), TotalVariation2D(0.001); maxit = 100); verbosity = Silent())
            end
        end

        @testset "Different algorithms" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_fista = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.001); maxit = 100); verbosity = Silent()),
            )
            error_fista = norm(img_fista - img_true) / norm(img_true)
            @test error_fista < 0.3  # measured ≈0.01-0.10 depending on the random sampling pattern

            img_admm = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(0.003); algorithm = ADMM(), maxit = 50); verbosity = Silent()),
            )
            error_admm = norm(img_admm - img_true) / norm(img_true)
            # A sign error in the ADMM data term lands at ≈2.0; measured ≈0.06 at 50 iterations.
            @test error_admm < 0.3

            # CG stops on its tolerance, and rounding moves that stop by tens of iterations, so its
            # comparison stays short of it.
            for method in (
                    IterativeReconstruction(L2Image(0.001); maxit = 30),
                    IterativeReconstruction(L1Wavelet2D(0.003); algorithm = ADMM(), maxit = 50),
                    IterativeReconstruction(L1Wavelet2D(0.003); algorithm = POGM(), maxit = 50),
                )
                test_on_devices(a -> reconstruct(a, method; verbosity = Silent()), acq_with_data; rtol = 1.0e-3)
            end
        end

        @testset "With initial guess" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            x_init = reconstruct(acq_with_data; verbosity = Silent())

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(0.005); maxit = 30); x₀ = x_init, verbosity = Silent()),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.5

            test_on_devices(acq_with_data, x_init; rtol = 1.0e-3) do a, x
                reconstruct(a, IterativeReconstruction(L1Wavelet2D(0.005); maxit = 30); x₀ = x, verbosity = Silent())
            end
        end
    end
end

@testitem "3D Reconstruction Pipeline" tags = [:reconstruction, :integration, :gpu] setup = [TestHelpers, GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms

    @testset "3D Reconstruction Pipeline" begin
        @testset "Fully-sampled 3D" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            img_true = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nz, nc)

            acq = AcquisitionInfo(is3D = true, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_with_data; verbosity = Silent()))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1.0e-3
        end

        @testset "Undersampled 3D with regularization" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            img_true = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nz, nc)

            pdf = UniformRandomSampling(4.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny, nz))

            acq = AcquisitionInfo(is3D = true, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet3D(0.005); maxit = 30); verbosity = Silent()))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.7

            test_on_devices(acq_with_data; rtol = 1.0e-3) do a
                reconstruct(a, IterativeReconstruction(L1Wavelet3D(0.005); maxit = 30); verbosity = Silent())
            end
        end
    end
end

@testitem "Multi-slice 2D Reconstruction" tags = [:reconstruction, :integration, :gpu] setup = [TestHelpers, GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using Ristretto.StructuredOptimization

    @testset "Multi-slice 2D Reconstruction" begin
        @testset "Multi-slice with task splitting" begin
            nx, ny, nslices, nc = 32, 32, 3, 4

            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            img_true_ms = repeat(img_true, 1, 1, nslices)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)

            ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
            for s in 1:nslices
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq; inverse_crime_check = false, keep_sensitivity_maps = true)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end

            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; verbosity = Silent()))

            @test size(img_recon) == (nx, ny, nslices)
            error_norm = norm(img_recon - img_true_ms) / norm(img_true_ms)
            @test error_norm < 1.0e-3
        end

        @testset "Multi-slice with regularization and task splitting" begin
            nx, ny, nslices, nc = 16, 16, 3, 2

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)

            ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            # L2Image regularization + multislice exercises task splitting with regularization
            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = Silent()))
            @test size(img_recon) == (nx, ny, nslices)
        end

        @testset "Regularized task splitting with varying slice intensities" begin
            # Slices with wildly different signal levels: regularized task splitting solves each
            # slice normalized by its own scale (so λ is applied consistently) but uses one shared
            # scale to convert every slice back to image units, matching a joint (unsplit)
            # solve of the whole stack. See scale_regularization.
            nx, ny, nc = 16, 16, 2
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            intensities = ComplexF32[0.1, 1.0, 5.0, 20.0]
            img_true_ms = cat((intensities[s] .* img_true for s in eachindex(intensities))...; dims = 3)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, length(intensities))

            ksp_ms = zeros(ComplexF32, nx, ny, nc, length(intensities))
            for s in eachindex(intensities)
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq; inverse_crime_check = false, keep_sensitivity_maps = true)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_split = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.05); maxit = 20); disable_task_splitting = false, verbosity = Silent())
            img_no_split = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.05); maxit = 20); disable_task_splitting = true, verbosity = Silent())

            # Split vs jointly-solved must agree closely regardless of the intensity spread.
            @test norm(img_split - img_no_split) / norm(img_no_split) < 1.0e-3

            # Regularization strength must stay consistent across slices: relative error against
            # the true image should not blow up for the low- or high-intensity slices.
            rel_errors = [
                norm(img_split[:, :, s] - img_true_ms[:, :, s]) / norm(img_true_ms[:, :, s])
                    for s in eachindex(intensities)
            ]
            @test maximum(rel_errors) / minimum(rel_errors) < 1.1

            # Split and unsplit solves on a device, each against the same on the host.
            for disable_task_splitting in (true, false)
                test_on_devices(acq_ms; rtol = 1.0e-3) do a
                    reconstruct(a, IterativeReconstruction(L2Image(0.05); maxit = 20); disable_task_splitting, verbosity = Silent())
                end
            end
        end

        @testset "Regularized task splitting with an all-zero slice" begin
            # A slice whose own scale estimate is (near) zero must not have its regularization
            # collapse to zero (which would leave noise unregularized); safe_scale_ratio guards this.
            nx, ny, nc = 16, 16, 2
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            img_true_ms = cat(img_true, zeros(ComplexF32, nx, ny), img_true; dims = 3)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, 3)

            ksp_ms = zeros(ComplexF32, nx, ny, nc, 3)
            for s in 1:3
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq; inverse_crime_check = false, keep_sensitivity_maps = true)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_recon = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.05); maxit = 15); disable_task_splitting = false, verbosity = Silent())
            @test all(isfinite, img_recon)
            @test norm(img_recon[:, :, 2]) / norm(img_recon[:, :, 1]) < 0.1
        end
    end
end

@testitem "ReconstructionConfig and Configuration Options" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms

    @testset "ReconstructionConfig and Configuration Options" begin
        @testset "ReconstructionConfig object usage" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img1 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 20, reltol = 1.0e-5); verbosity = Silent()))
            config = ReconstructionConfig(; verbosity = Silent())
            img2 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 20, reltol = 1.0e-5); config = config))
            # Iteration control lives on the method, so extending a `ReconstructionConfig` cannot change it;
            # only the run settings come from the config.
            config_base = ReconstructionConfig(; verbosity = Verbose())
            img3 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 20, reltol = 1.0e-5); config = config_base, verbosity = Silent()))

            # `maxit`/`reltol`/`verbose` at `reconstruct` are rejected, not silently ignored.
            @test_throws ArgumentError reconstruct(acq_with_data, DirectReconstruction(); maxit = 5)
            @test_throws ArgumentError reconstruct(acq_with_data, DirectReconstruction(); tol = 1.0e-5)
            @test_throws ArgumentError reconstruct(acq_with_data, DirectReconstruction(); verbose = false)

            # Loose tolerance: threaded FFTs make repeated solver runs agree only to ~1e-3
            @test norm(img1 - img2) / norm(img1) < 5.0e-3
            @test norm(img1 - img3) / norm(img1) < 5.0e-3
        end

        @testset "Threading configuration" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            img_st = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; threaded = false, verbosity = Silent()))
            img_mt = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; threaded = true, verbosity = Silent()))

            @test norm(img_st - img_mt) / norm(img_st) < 1.0e-10
        end

        @testset "Scaling strategies" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            # Only the output shape is checked here, so a single iteration is enough -- 20 iterations
            # bought no extra coverage, just a slower test.
            img_bart = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1); scaling = BartScaling(), verbosity = Silent()))
            img_noscale = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1); scaling = NoScaling(), verbosity = Silent()))
            img_meas = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1); scaling = MeasurementBasedScaling(), verbosity = Silent()))

            @test size(img_bart) == size(img_noscale) == size(img_meas)
        end

        @testset "FixedScaling" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            @test Ristretto.get_scale(FixedScaling(2.5), acq_with_data, nothing, nothing) == 2.5
            @test_throws ArgumentError FixedScaling(0.0)
            @test_throws ArgumentError FixedScaling(-1.0)

            scale = Ristretto.get_scale(BartScaling(), acq_with_data, img_true, nothing)
            # The selection-based quantiles are Statistics' `quantile`, to the bit.
            let quantile = Ristretto.quantile, a = abs.(vec(img_true)),
                    (m, p, mx) = quantile(a, [0.5, 0.9, 1.0])
                @test scale == (((mx - p) < 2 * (p - m)) ? p : mx)
                for n in (1, 2, 7, 1000), q in (0.0, 0.5, 0.9, 1.0)
                    v = rand(Float32, n)
                    @test Ristretto._quantile_select!(copy(v), q) == quantile(v, q)
                end
            end
            # Only the output shape is checked here, so a single iteration is enough -- 20 iterations
            # bought no extra coverage, just a slower test.
            img_fixed = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1, reltol = 0.0); scaling = FixedScaling(scale), verbosity = Silent()))
            img_bart = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1, reltol = 0.0); scaling = BartScaling(), verbosity = Silent()))
            @test size(img_fixed) == size(img_bart)
        end
    end
end

@testitem "NamedDims Support" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using NamedDims

    @testset "NamedDims Support" begin
        @testset "NamedDims preservation" begin
            nx, ny, nc = 32, 32, 4

            ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, nx, ny, nc))
            smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))

            acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)

            img_recon = test_type_stable(NamedDimsArray{(:x, :y), ComplexF32, 2, Matrix{ComplexF32}}, reconstruct(acq; verbosity = Silent()))

            @test dimnames(img_recon) == (:x, :y)
            @test eltype(img_recon) == ComplexF32
        end

        @testset "NamedDims with task splitting" begin
            nx, ny, nslices, nc = 16, 16, 3, 2

            ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(rand(ComplexF32, nx, ny, nc, nslices))
            smaps = NamedDimsArray{(:x, :y, :coil, :z)}(repeat(coil_sensitivities(nx, ny, nc), 1, 1, 1, nslices))

            acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)

            img_direct = reconstruct(acq; verbosity = Silent())
            @test parent(img_direct) isa NamedDimsArray
            @test dimnames(img_direct) == (:x, :y, :z)
            @test size(img_direct) == (nx, ny, nslices)

            img_reg = reconstruct(acq, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = Silent())
            @test parent(img_reg) isa NamedDimsArray
            @test dimnames(img_reg) == (:x, :y, :z)
            @test size(img_reg) == (nx, ny, nslices)
        end
    end
end

@testitem "Operator Options" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using Ristretto.StructuredOptimization

    @testset "Operator Options" begin
        @testset "Operator normalization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

            # Only the output shape is checked here, so a single iteration is enough -- 20 iterations
            # bought no extra coverage, just a slower test.
            img_norm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); disable_operator_normalization = false, maxit = 1); verbosity = Silent()))
            img_unnorm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); disable_operator_normalization = true, maxit = 1); verbosity = Silent()))

            @test size(img_norm) == size(img_unnorm)
        end

        @testset "Task splitting control" begin
            nx, ny, nslices, nc = 32, 32, 2, 4

            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            img_true_ms = repeat(img_true, 1, 1, nslices)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)

            ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
            for s in 1:nslices
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq; inverse_crime_check = false, keep_sensitivity_maps = true)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end

            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_split = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; disable_task_splitting = false, verbosity = Silent()))
            img_no_split = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; disable_task_splitting = true, verbosity = Silent()))

            # The two paths are the same computation in a different summation order, so they can
            # only agree to Float32 precision (eps ≈ 1.2e-7), and the order the reductions actually
            # take depends on threading. 1e-10 was below what the element type can deliver and made
            # this assertion flaky; 1e-5 still catches any real divergence between the paths.
            @test norm(img_split - img_no_split) / norm(img_split) < 1.0e-5

            # Regularized case: slices are identical, so the per-slice median scale equals
            # the global scale and both paths must converge to the same solution.
            img_split_reg = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 30); disable_task_splitting = false, verbosity = Silent())
            img_no_split_reg = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 30); disable_task_splitting = true, verbosity = Silent())

            @test norm(img_split_reg - img_no_split_reg) / norm(img_no_split_reg) < 1.0e-3
        end

        @testset "x₀ with task splitting" begin
            nx, ny, nslices, nc = 16, 16, 3, 2
            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)
            ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            x₀ = zeros(ComplexF32, nx, ny, nslices)
            img_recon = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); x₀, verbosity = Silent())
            @test size(img_recon) == (nx, ny, nslices)

            x₀_wrong = zeros(ComplexF32, nx, ny)
            @test_throws ArgumentError reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); x₀ = x₀_wrong, verbosity = Silent())

            # `reconstruct` must not write its solution back through the caller's `x₀`. The
            # component path handed the arrays straight to `Variable`, which stores them by
            # reference, so `solve`'s final write-back landed in the caller's arrays -- visible
            # whenever `scale == 1` skips the reallocating rescale, and through the `@view`s the
            # task-splitting path passes down.
            x₀_keep = rand(ComplexF32, nx, ny, nslices)
            x₀_ref = copy(x₀_keep)
            reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); x₀ = x₀_keep, scaling = NoScaling(), verbosity = Silent())
            @test x₀_keep == x₀_ref
        end
    end
end

@testitem "Verbose and MultiThreading Task Splitting" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using Ristretto
    using Ristretto: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms

    @testset "Verbose progress output" begin
        nx, ny, nc = 16, 16, 2
        img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        smaps = coil_sensitivities(nx, ny, nc)
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
        acq_with_data = simulate_acquisition(img_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true)

        output = IOBuffer()
        printfunc = (args...) -> print(output, args...)
        config = ReconstructionConfig(; verbosity = Verbose(; printfunc = printfunc))
        img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; config = config))

        @test size(img_recon) == (nx, ny)
        @test length(take!(output)) > 0
    end

    @testset "MultiThreadingExecutor task splitting" begin
        nx, ny, nslices, nc = 16, 16, 4, 2
        smaps = coil_sensitivities(nx, ny, nc)
        smaps_ms = repeat(smaps, 1, 1, 1, nslices)
        ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
        acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

        # Force MultiThreadingExecutor to cover that path in task_splitting/execution.jl
        executor = Ristretto.MultiThreadingExecutor()
        img_recon = test_type_stable(
            Array{ComplexF32, 3},
            reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); task_executor = executor, verbosity = Silent()),
        )
        @test size(img_recon) == (nx, ny, nslices)
    end
end

@testitem "Per-slice threading gate" tags = [:reconstruction, :integration] begin
    using Ristretto
    using GeometricMedicalPhantoms

    nx, ny, nslices, nc = 128, 128, 2, 4
    smaps = repeat(coil_sensitivities(nx, ny, nc), 1, 1, 1, nslices)
    ksp = rand(ComplexF32, nx, ny, nc, nslices)
    acq = AcquisitionInfo(ksp; is3D = false, sensitivity_maps = smaps)
    method = IterativeReconstruction(L2Image(0.01f0); maxit = 5)

    config = ReconstructionConfig(; threaded = true, verbosity = Silent())
    plan = Ristretto.get_task_splitting_plan(acq, method, config)
    @test plan !== nothing

    # Slice size decides which executor is picked, not whether the inside of a slice threads.
    @test Ristretto.slice_bytes(plan, acq) == nx * ny * sizeof(ComplexF32)
    @test Ristretto.slice_bytes(plan, acq) < Ristretto.serial_blas_threshold_bytes()

    # A multi-threading executor already has every thread busy with whole slices, so the work
    # inside one must stay serial. A sequential executor leaves the threads free and passes
    # `config.threaded` through: how small is too small to thread is the operator's call, made
    # per kernel and per input, not a blanket rule applied here.
    @test Ristretto.slice_threading(config, Ristretto.SequentialExecutor()) == true
    @test Ristretto.slice_threading(config, Ristretto.MultiThreadingExecutor()) == false
    @test Ristretto.slice_threading(
        ReconstructionConfig(config; threaded = false), Ristretto.SequentialExecutor()
    ) == false

    big_size = (2048, 2048, nslices)
    big = Ristretto.TaskSplittingPlan(
        big_size, (3,), (2048, 2048, nc, nslices), (4,), false, big_size
    )
    @test Ristretto.slice_bytes(big, acq) >= Ristretto.serial_blas_threshold_bytes()

    # The gate is a performance decision only: the result may not depend on it.
    img_threaded = reconstruct(acq, method; threaded = true, verbosity = Silent())
    img_serial = reconstruct(acq, method; threaded = false, verbosity = Silent())
    @test size(img_threaded) == (nx, ny, nslices)
    @test img_threaded == img_serial
end

@testitem "Threading scopes: BLAS is not a proxy for the process" tags = [:reconstruction, :minimizer] begin
    using Ristretto
    using FFTW, LinearAlgebra

    # `with_restricted_threads` narrows every counted pool *and* switches off the Polyester
    # guard, so a serial BLAS says nothing about whether entering it would be a no-op. The solve
    # path used to skip the scope on `BLAS.get_num_threads() == 1`, which left FFTW, NFFT and
    # Polyester at full width for the whole solve -- exactly the oversubscription the gate exists
    # to prevent.
    if Ristretto.capacity() > 1   # nothing to observe on a single-threaded process
        blas0, fftw0 = BLAS.get_num_threads(), FFTW.get_num_threads()
        try
            BLAS.set_num_threads(1)
            FFTW.set_num_threads(Ristretto.capacity())
            @test FFTW.get_num_threads() > 1
            inside = Ristretto.with_restricted_threads() do
                (FFTW.get_num_threads(), BLAS.get_num_threads())
            end
            @test inside == (1, 1)
            @test FFTW.get_num_threads() > 1   # restored on exit
        finally
            BLAS.set_num_threads(blas0)
            FFTW.set_num_threads(fftw0)
        end
    end
end

@testitem "Task splitting: the hoisted first item shares the loop's threading scope" tags = [:reconstruction, :integration] begin
    using Ristretto
    using FFTW

    # Under the sequential executor `map_items` runs item 1 outside the loop to learn its
    # concrete result type. That must not put it in a different threading scope from items 2..n:
    # unrestricted where the sequential loop restricts every pool, or without NFFT's guarded pool
    # where the loop enables it.
    if Ristretto.capacity() > 1
        items = collect(1:4)
        rest = @view(items[2:end])
        config = ReconstructionConfig(; threaded = true, verbosity = Silent())
        fftw0 = FFTW.get_num_threads()
        try
            FFTW.set_num_threads(Ristretto.capacity())
            for (executor, threaded) in (
                    (Ristretto.SequentialExecutor(), false),
                    (Ristretto.SequentialExecutor(), true),
                )
                first_seen = Ristretto.run_first_item(rest, config, executor; threaded) do
                    FFTW.get_num_threads()
                end
                # Indexed rather than pushed: the multi-threading executor runs the body from
                # several tasks at once.
                rest_seen = zeros(Int, length(items))
                Ristretto.for_each_item!(rest, config, executor; threaded) do i
                    rest_seen[i] = FFTW.get_num_threads()
                end
                @test all(==(first_seen), @view(rest_seen[2:end]))
            end
        finally
            FFTW.set_num_threads(fftw0)
        end
    end
end

@testitem "Task splitting: a type-inconsistent slice names itself" tags = [:reconstruction] begin
    using Ristretto

    # The result array is allocated from the first item's concrete type, so a later item of a
    # different type would otherwise surface as a bare `convert`/`MethodError` from inside a
    # threaded loop.
    dest = Array{Vector{Float64}}(undef, 2)
    Ristretto.store_item!(dest, 1, [1.0, 2.0], "slice 1")
    @test dest[1] == [1.0, 2.0]
    err = try
        Ristretto.store_item!(dest, 2, [1.0f0, 2.0f0], "slice 2")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("slice 2", err.msg)
    @test occursin("Vector{Float32}", err.msg)

    # `map_items` keeps the items' order and reports a type-inconsistent item under either
    # executor, the multi-threading one included, which runs every item in its loop.
    config = ReconstructionConfig(; threaded = true, verbosity = Silent())
    for executor in (Ristretto.SequentialExecutor(), Ristretto.MultiThreadingExecutor())
        out = Ristretto.map_items(k -> 2k, collect(1:5), string.(1:5), config, executor)
        @test out == [2, 4, 6, 8, 10] && out isa Vector{Int}
        @test_throws ArgumentError Ristretto.map_items(k -> k == 3 ? 3.0 : k, collect(1:5), string.(1:5), config, executor)
    end
end

@testitem "Serial-BLAS threshold is settable" tags = [:reconstruction] begin
    using Ristretto

    @test Ristretto.serial_blas_threshold_bytes() == Ristretto.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES
    config = ReconstructionConfig(; threaded = true)
    @test Ristretto._should_thread_work_item(config, Ristretto.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES)
    @test !Ristretto._should_thread_work_item(config, 4 * 2^20)

    try
        Ristretto.set_serial_blas_threshold_bytes!(2^20)
        @test Ristretto.serial_blas_threshold_bytes() == 2^20
        # The gate follows the new value, which is the whole point of it being settable.
        @test Ristretto._should_thread_work_item(config, 4 * 2^20)
        @test !Ristretto._should_thread_work_item(config, 2^19)
        # `threaded = false` still vetoes, at any size.
        @test !Ristretto._should_thread_work_item(ReconstructionConfig(config; threaded = false), 2^30)
    finally
        Ristretto.set_serial_blas_threshold_bytes!(Ristretto.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES)
    end
    @test Ristretto.serial_blas_threshold_bytes() == Ristretto.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES

    @test_throws Exception Ristretto.set_serial_blas_threshold_bytes!(-1)
end

@testitem "Serial BLAS is a soft default that large calls override" tags = [:reconstruction] begin
    using Ristretto, LinearAlgebra, ProximalCore
    const PO = Ristretto.ProximalOperators
    const SO = Ristretto.StructuredOptimization
    const NestedThreading = PO.NestedThreading

    blas = BLAS.get_num_threads()
    Ristretto.with_serial_blas() do
        @test BLAS.get_num_threads() == 1
        # A grant, which the operator stack opens around a large factorization, gemm or CG
        # step, takes BLAS back...
        NestedThreading.with_thread_grant(typemax(Int); only = (:blas,)) do
            @test BLAS.get_num_threads() == blas
        end
        # ...but never past a hard limit, such as a saturated slice loop opens.
        NestedThreading.with_thread_budget(1) do
            NestedThreading.with_thread_grant(typemax(Int); only = (:blas,)) do
                @test BLAS.get_num_threads() == 1
            end
        end
    end
    @test BLAS.get_num_threads() == blas

    # A locally-low-rank prox gives the same result whether its block SVDs are granted threads
    # or not.
    x = randn(ComplexF32, 16, 16, 6)
    reg = LocallyLowRank(0.1; block_size = 8, time_dim = 3)
    f = SO.weighted_function(Ristretto.materialize(reg, SO.Variable(x); threaded = true))
    old = PO.FACTORIZATION_THREAD_WORK[]
    results = map((typemax(Int), 1)) do gate
        PO.FACTORIZATION_THREAD_WORK[] = gate
        try
            Ristretto.with_serial_blas() do
                y = similar(x)
                (y, ProximalCore.prox!(y, f, x, 0.7))
            end
        finally
            PO.FACTORIZATION_THREAD_WORK[] = old
        end
    end
    @test results[1][1] ≈ results[2][1]
    @test results[1][2] ≈ results[2][2]
    @test BLAS.get_num_threads() == blas
end

@testitem "Per-frame subsampling: one ky mask per frame" tags = [:reconstruction, :integration] begin
    using Ristretto
    using Ristretto: get_encoding_operator
    using Ristretto.AbstractOperators: get_normal_op
    using NamedDims: NamedDimsArray, dimnames, unname
    using LinearAlgebra: mul!
    using Random: Xoshiro, randn!

    # `subsampling` may be an array of specs, one per batch element, rather than one spec shared
    # by the whole acquisition — a different ky mask per cardiac phase, which is what makes the
    # aliasing incoherent across time and gives a temporal regularizer something to exploit.
    #
    # `nkx * nky * ncoil` is deliberately above `AbstractOperators.MIN_BATCH_WORK_FOR_PARALLEL`
    # (2^10): only above it does the batch operator take its thread-safe form, whose normal
    # operator used to be built with the subsampled codomain size.
    rng = Xoshiro(0)
    nkx, nky, ncoil, nframes = 16, 24, 3, 4
    base = falses(nky)
    base[1:3:nky] .= true
    masks = [circshift(base, t - 1) for t in 1:nframes]
    nlines = sum(base)
    # Equal line counts keep the dense path: one k-space array, one `BatchOp` of `GetIndex`es.
    # Unequal counts are supported too, through `PartitionedKSpace` — see the next testitem.
    @test all(m -> sum(m) == nlines, masks)

    smaps = NamedDimsArray{(:x, :y, :coil)}(randn(rng, ComplexF32, nkx, nky, ncoil))
    truth = NamedDimsArray{(:x, :y, :time)}(randn(rng, ComplexF32, nkx, nky, nframes))
    empty_ksp = NamedDimsArray{(:kx, :ky, :coil, :time)}(
        zeros(ComplexF32, nkx, nlines, ncoil, nframes)
    )

    acq = AcquisitionInfo(
        empty_ksp; is3D = false, image_size = (nkx, nky),
        subsampling = [(:, m) for m in masks], sensitivity_maps = smaps,
    )
    @test occursin("subsampling=$(nframes)×(:, Vector{Bool}<$nky>)", string(acq))

    𝒜 = get_encoding_operator(acq)
    @test size(𝒜) == ((nkx, nlines, ncoil, nframes), (nkx, nky, nframes))
    # 𝒜ᴴ𝒜 maps the image domain to itself. A normal operator that claims the *subsampled*
    # codomain shape instead makes `Compose` size the buffer between the two halves wrongly,
    # and every `estimate_opnorm`/CG step on such an operator throws a `DimensionMismatch`.
    @test size(get_normal_op(𝒜)) == (size(𝒜, 2), size(𝒜, 2))

    y = NamedDimsArray{(:kx, :ky, :coil, :time)}(similar(unname(empty_ksp)))
    mul!(y, 𝒜, truth)
    acq = AcquisitionInfo(acq; kspace_data = y)

    # `:time` is a batch dimension for an unregularized solve, so task splitting must hand each
    # frame the mask that actually produced it — not the whole array of specs.
    x_adj = reconstruct(acq; verbosity = Silent())
    @test dimnames(x_adj) == (:x, :y, :time)
    for t in 1:nframes
        acq_t = AcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(unname(y)[:, :, :, t]); is3D = false,
            image_size = (nkx, nky), subsampling = (:, masks[t]), sensitivity_maps = smaps,
        )
        @test unname(reconstruct(acq_t; verbosity = Silent())) ≈ unname(x_adj)[:, :, t]
    end

    # A temporal regularizer keeps `:time` in the variable, so the whole spec array reaches the
    # operator instead: the other half of the contract.
    x_cs = reconstruct(
        acq, IterativeReconstruction(L1TemporalFourier(1.0f-3; time_dim = :time); maxit = 5);
        verbosity = Silent()
    )
    @test size(x_cs) == (nkx, nky, nframes)
    @test all(isfinite, unname(x_cs))
end

@testitem "Per-frame subsampling: unequal sample counts per frame" tags = [:reconstruction, :integration] begin
    using Ristretto: CartesianAcquisitionInfo
    using Ristretto
    using Ristretto: get_encoding_operator, parts, nparts, is_partitioned,
        to_array_partition
    using NamedDims: NamedDimsArray, dimnames, unname
    using Random: Xoshiro

    # When the per-frame masks select *different* numbers of lines, the measurement no longer fits
    # in a rectangle: the sample axis would need one length per frame. `PartitionedKSpace` holds it
    # one frame at a time instead, and the subsampling operator becomes a `VCAT` of per-frame
    # `GetIndex`es whose codomain is an `ArrayPartition`.
    rng = Xoshiro(2)
    nkx, nky, ncoil, nframes = 16, 24, 3, 4
    base = falses(nky)
    base[1:3:nky] .= true
    masks = [circshift(base, t - 1) for t in 1:nframes]
    masks[2][2] = true
    masks[3][5] = true
    masks[3][7] = true
    counts = sum.(masks)
    @test !all(==(first(counts)), counts) # the point of this testitem

    specs = [(:, m) for m in masks]
    smaps = NamedDimsArray{(:x, :y, :coil)}(randn(rng, ComplexF32, nkx, nky, ncoil))
    truth = NamedDimsArray{(:x, :y, :time)}(randn(rng, ComplexF32, nkx, nky, nframes))

    acq_empty = CartesianAcquisitionInfo(;
        is3D = false, image_size = (nkx, nky), sensitivity_maps = smaps, subsampling = specs,
    )
    acq = simulate_acquisition(truth, acq_empty; inverse_crime_check = false, keep_sensitivity_maps = true)
    ksp = acq.kspace_data
    @test is_partitioned(ksp)
    @test nparts(ksp) == nframes
    @test map(p -> size(p, 2), parts(ksp)) == counts
    @test dimnames(ksp) == (:kx, :ky, :coil, :time)
    # Dimension 2 is ragged, so there is no honest `size` for it — asking is a bug in the caller,
    # not something to answer with a number.
    @test_throws ArgumentError size(ksp)
    @test size(ksp, 3) == ncoil
    @test size(ksp, 4) == nframes
    @test occursin("ky: {$(join(counts, ","))}", string(acq))

    𝒜 = get_encoding_operator(acq)
    @test size(𝒜, 2) == (nkx, nky, nframes)
    @test size(𝒜, 1) == Tuple((nkx, counts[t], ncoil) for t in 1:nframes)

    y = 𝒜 * truth
    # An `ArrayPartition`, without naming the type: `RecursiveArrayTools` is not a test dependency.
    @test typeof(y) === typeof(to_array_partition(ksp))
    @test y ≈ to_array_partition(ksp)
    # The forward model is per-frame: one block per frame, each with that frame's sample count.
    @test all(t -> size(y.x[t]) == (nkx, counts[t], ncoil), 1:nframes)

    # The adjoint lands back on a single dense image-sized array, which is why `reconstruct` can
    # return an ordinary `Array` even though the measurement is partitioned.
    x_adj = reconstruct(acq; verbosity = Silent())
    @test dimnames(x_adj) == (:x, :y, :time)
    @test unname(x_adj) isa Array{ComplexF32, 3}
    @test size(x_adj) == (nkx, nky, nframes)

    # Task splitting hands each frame its own slice and its own spec: the result must equal a
    # per-frame reconstruction of that frame alone.
    for t in 1:nframes
        acq_t = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(unname(parts(ksp)[t])); is3D = false,
            image_size = (nkx, nky), subsampling = specs[t], sensitivity_maps = smaps,
        )
        @test unname(reconstruct(acq_t; verbosity = Silent())) ≈ unname(x_adj)[:, :, t]
    end

    # ... and so must an iterative solve with a separable regularizer. `NoScaling` because a shared
    # scale across frames is what otherwise separates the joint solve from the per-frame ones.
    method = IterativeReconstruction(L2Image(1.0f-2); maxit = 300, reltol = 1.0f-12)
    x_l2 = reconstruct(acq, method; verbosity = Silent(), scaling = NoScaling())
    for t in 1:nframes
        acq_t = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(unname(parts(ksp)[t])); is3D = false,
            image_size = (nkx, nky), subsampling = specs[t], sensitivity_maps = smaps,
        )
        x_t = reconstruct(acq_t, method; verbosity = Silent(), scaling = NoScaling())
        @test unname(x_t) ≈ unname(x_l2)[:, :, t]
    end

    # A temporal regularizer keeps `:time` in the variable, so the partitioned operator is used
    # whole rather than sliced — the other half of the contract.
    x_cs = reconstruct(
        acq, IterativeReconstruction(L1TemporalFourier(1.0f-3; time_dim = :time); maxit = 5);
        verbosity = Silent()
    )
    @test size(x_cs) == (nkx, nky, nframes)
    @test unname(x_cs) isa Array{ComplexF32, 3}
    @test all(isfinite, unname(x_cs))

    # Noise is well defined per frame, so `add_noise` works; the paths that need a dense grid say
    # so instead of returning a plausible wrong answer.
    noisy = add_noise(acq; noise_std = 1.0f-3)
    @test is_partitioned(noisy.kspace_data)
    @test map(p -> size(p, 2), parts(noisy.kspace_data)) == counts
    @test_throws ArgumentError prewhiten(acq, [1.0f0 + 0im;;])
    @test_throws ArgumentError estimate_sensitivities(acq)
    @test_throws ArgumentError compress_coils(acq, 2)
    @test_throws ArgumentError partial_fourier_band(acq)
    # The remaining `_reject_partitioned` guards, so that dropping one fails here rather than
    # returning a plausible image built from the wrong samples. Every entry point that indexes
    # k-space as a rectangle belongs in this list.
    @test_throws ArgumentError pseudo_replica(acq, DirectReconstruction(); nreplicas = 2)
    @test_throws ArgumentError reconstruct(
        acq,
        IterativeReconstruction(
            L1Image(1.0f-3); signal_model = KSpaceToImage(RootSumSquares()), maxit = 2,
        );
        verbosity = Silent(),
    )
    @test_throws ArgumentError reconstruct(acq, GRAPPA(; kernel_size = (3, 3)); verbosity = Silent())
end

@testitem "Task splitting: non-Cartesian" tags = [:reconstruction, :integration, :nfft] begin
    using LinearAlgebra, Random
    using Ristretto: NonCartesianAcquisitionInfo, get_task_splitting_plan, get_encoding_operator, unname, get_slices
    Random.seed!(0)
    n, nc, nsample = 48, 4, 96
    xs = range(-1, 1; length = n)
    disk(r, s) = ComplexF32[(x^2 + (y / 0.8)^2 < r ? s : 0) + (abs(x - 0.2s) < 0.1 && abs(y) < 0.3) for x in xs, y in xs]
    traj(nspoke, frames...) = reshape(
        Float32.(unname(radial_trajectory(nsample, nspoke * prod(frames; init = 1); ordering = GoldenAngle()))),
        2, nsample, nspoke, frames...,
    )
    function simulate(img, trajectory, maps)
        nb = size(img, 3)
        empty = NonCartesianAcquisitionInfo(
            zeros(ComplexF32, size(trajectory, 2), size(trajectory, 3), nc, nb);
            trajectory, image_size = (n, n), sensitivity_maps = maps,
        )
        ksp = get_encoding_operator(empty) * img
        return NonCartesianAcquisitionInfo(ksp; trajectory, image_size = (n, n), sensitivity_maps = maps)
    end
    nrmse(x, ref) = norm(abs.(unname(parent(x))) .- abs.(ref)) / norm(ref)
    tv = IterativeReconstruction(; regularization = TotalVariation2D(1.0e-3), maxit = 40)
    config = ReconstructionConfig()

    @testset "multislice, shared trajectory, maps per slice" begin
        img = stack(disk(0.6, s) for s in 1:3)
        maps = randn(ComplexF32, n, n, nc, 3) .* 0.1f0 .+ 1
        acq = simulate(img, traj(40), maps)
        @test !isnothing(get_task_splitting_plan(acq, tv, config))
        split = reconstruct(acq, tv; verbosity = Silent())
        whole = reconstruct(acq, tv; verbosity = Silent(), disable_task_splitting = true)
        @test nrmse(split, img) <= nrmse(whole, img) + 1.0e-3
        # Without a regularization to compensate, a split repeats the trajectory's setup per slice
        # for nothing, so it is taken only on request.
        gridded = density_compensation(acq)
        @test isnothing(get_task_splitting_plan(gridded, DirectReconstruction(), config))
        @test isnothing(get_task_splitting_plan(acq, IterativeReconstruction(; maxit = 5), config))
        split_config = ReconstructionConfig(; disable_task_splitting = false)
        @test !isnothing(get_task_splitting_plan(gridded, DirectReconstruction(), split_config))
        @test reconstruct(gridded, DirectReconstruction(); disable_task_splitting = false) ≈
            reconstruct(gridded, DirectReconstruction(); disable_task_splitting = true) rtol = 1.0e-5
    end

    @testset "per-frame trajectory, explicit dcf" begin
        nt = 5  # not the coil count, so the frame axis is unambiguous by size
        img = stack(disk(0.6, s) for s in (1, 2, 3, 1, 2))
        acq = density_compensation(simulate(img, traj(20, nt), randn(ComplexF32, n, n, nc) .* 0.1f0 .+ 1))
        plan = get_task_splitting_plan(acq, tv, config)
        @test !isnothing(plan)
        _, _, first_acq = first(get_slices(plan, acq))
        @test size(first_acq.trajectory) == (2, nsample, 20)
        @test first_acq.dcf == acq.dcf[:, :, 1]
        # Each frame's λ is compensated by its own scale, so the frames, whose intensities differ
        # threefold, are not solved with the unsplit run's single λ: close to it, not equal.
        split = reconstruct(acq, tv; verbosity = Silent())
        whole = reconstruct(acq, tv; verbosity = Silent(), disable_task_splitting = true)
        @test nrmse(split, img) <= nrmse(whole, img) + 1.0e-2
        @test reconstruct(acq, DirectReconstruction(); disable_task_splitting = false) ≈
            reconstruct(acq, DirectReconstruction(); disable_task_splitting = true) rtol = 1.0e-5
        # A temporal regularizer couples the frames: they stay one problem.
        ttv = IterativeReconstruction(; regularization = TemporalTotalVariation(1.0e-3; time_dim = 3))
        @test isnothing(get_task_splitting_plan(acq, ttv, config))
    end
end
