using TestItems

@testitem "Regularization terms in reconstruction" tags = [:regularization, :integration, :gpu] setup = [TestHelpers, GpuEnvSetup, GpuHelpers] begin
    using Test
    using LinearAlgebra
    using Ristretto
    using Ristretto: Regularization
    using NamedDims
    using Random
    using Ristretto.AbstractOperators

    Random.seed!(42)
    nx, ny, nt = 16, 16, 6
    img = zeros(ComplexF32, nx, ny, nt)
    for t in 1:nt
        img[4:12, 4:12, t] .= 1 + 0.2f0 * t
        img[6:8, 6:8, t] .= 2
    end

    ksp = zeros(ComplexF32, nx, ny, nt)
    for t in 1:nt
        acq_t = AcquisitionInfo(is3D = false, image_size = (nx, ny))
        ksp[:, :, t] = simulate_acquisition(img[:, :, t], acq_t; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data
    end
    acq = AcquisitionInfo(NamedDimsArray{(:kx, :ky, :time)}(ksp); is3D = false)

    relerr(x) = relative_error(Array(x), img)

    @testset "LocallyLowRank" begin
        @test relerr(
            reconstruct(
                acq, IterativeReconstruction(LocallyLowRank(0.02f0; block_size = 4, time_dim = 3), maxit = 60); verbosity = Silent()
            )
        ) < 0.1
    end

    @testset "TemporalTotalVariation" begin
        @test relerr(
            reconstruct(
                acq, IterativeReconstruction(TemporalTotalVariation(0.02f0; time_dim = 3), maxit = 100); verbosity = Silent()
            )
        ) < 0.1
    end

    @testset "L+S with LocallyLowRank and TemporalTotalVariation components" begin
        components = (
            Component(:lowrank, LocallyLowRank(0.02f0; block_size = 4, time_dim = 3)),
            Component(:sparse, TemporalTotalVariation(0.02f0; time_dim = 3)),
        )
        img_recon = reconstruct(acq, IterativeReconstruction(components...; maxit = 100); verbosity = Silent())
        @test img_recon isa ReconImage
        @test relerr(img_recon) < 0.2
    end

    @testset "L+S with LowRank and JointSparsity components" begin
        components = (
            Component(:lowrank, LowRank(0.02f0; time_dim = 3)),
            Component(:sparse, L1Image(0.02f0)),
        )
        img_recon = reconstruct(acq, IterativeReconstruction(components...; maxit = 100); verbosity = Silent())
        @test img_recon isa ReconImage
        @test relerr(img_recon) < 0.2
    end

    # Regression: two components whose operators are both non-tight (here `LowRank`'s reshape and
    # the finite difference of temporal TV) used to fail to prepare for any algorithm, because
    # `Reshape` claimed `is_AAc_diagonal == true` without a `diag_AAc` method to back it up, so any
    # code path that trusted the trait crashed instead of falling back to a different assumption.
    @testset "LowRank + TemporalTotalVariation components" begin
        components = (
            Component(:lowrank, LowRank(0.02f0; time_dim = 3)),
            Component(:sparse, TemporalTotalVariation(0.02f0; time_dim = 3)),
        )
        img_recon = reconstruct(acq, IterativeReconstruction(components...; maxit = 100); verbosity = Silent())
        @test img_recon isa ReconImage
        @test relerr(img_recon) < 0.2
    end

    @testset "on a device: $(nameof(typeof(reg)))" for reg in (
            LocallyLowRank(0.02f0; block_size = 4, time_dim = 3),
            MultiScaleLowRank(0.02f0; block_sizes = (2, 4), time_dim = 3),
            LowRank(0.02f0; time_dim = 3),
            TemporalTotalVariation(0.02f0; time_dim = 3),
            L1TemporalFourier(0.02f0; time_dim = 3),
            JointSparsity(0.02f0; dim = 3),
            TotalVariation2D(0.02f0),
            AnisotropicTotalVariation2D(0.02f0),
            SecondOrderTotalVariation2D(0.02f0),
            TotalGeneralizedVariation2D(0.02f0),
            EdgePreservingRoughness2D(0.02f0),
            L1Wavelet2D(0.02f0),
            L0Wavelet2D(; threshold = 1.0e-3),
            L0Image(; threshold = 1.0e-3),
            NonNegative(; complex_handling = :real),
        )
        # Unsplit on both sides: the host would otherwise split a spatial term over the frames,
        # and each frame's own scale changes the problem it solves.
        test_on_devices(acq; rtol = 1.0e-3) do a
            reconstruct(a, IterativeReconstruction(reg; maxit = 30); disable_task_splitting = true, verbosity = Silent())
        end
    end
    # A rank projection of a nearly degenerate spectrum is not continuous in its input, so the
    # device's full SVD and the host's truncated one part ways by more than rounding.
    @testset "on a device: RankLimit" begin
        test_on_devices(acq; rtol = 5.0e-2) do a
            reconstruct(a, IterativeReconstruction(RankLimit(3; time_dim = 3); maxit = 30); disable_task_splitting = true, verbosity = Silent())
        end
    end
end
