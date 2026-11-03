@testitem "Aqua" tags = [:quality, :aqua] begin
    using Aqua
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate
    Aqua.test_all(
        Ristretto;
        ambiguities = false,
        piracies = false,
        persistent_tasks = false,
        stale_deps = false,
        unbound_args = false,
        undefined_exports = false,
    )
    # The vendored NFFT carries NFFT.jl's own unbound type parameters (`precomputation.jl`), and
    # re-exports `ndct_transpose`, which AbstractNFFTs exports without defining. Both are those
    # packages' defects; everything else is checked.
    within(m, root) = m === root || (parentmodule(m) !== m && within(parentmodule(m), root))
    unbound = Aqua.detect_unbound_args_recursively(Ristretto)
    @test isempty(filter(m -> !within(m.module, Ristretto.NFFT), unbound))
    undefined = Aqua.undefined_exports(Ristretto)
    @test isempty(filter(s -> !startswith(string(s), "Ristretto.NFFT."), undefined))
    # The vendored dependencies live in submodules of this package, so a recursive ambiguity
    # check would report their ambiguities as ours. They are checked -- and, where they are
    # unavoidable consequences of the operator syntax, tolerated -- by their own test suites,
    # so only this package's own methods are checked here.
    Aqua.test_ambiguities(Ristretto; recursive = false)
end

@testitem "JET test_package" tags = [:quality, :jet] begin
    using JET
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate
    # `LastFrameModuleExact`, not the bare module: the vendored dependencies are submodules of
    # this one, and a plain module target reports their inference problems as ours.
    JET.test_package(
        Ristretto;
        target_modules = (JET.LastFrameModuleExact(Ristretto),),
    )
end

@testitem "JET exported API @test_opt" tags = [:quality, :jet] begin
    using Ristretto: CartesianAcquisitionInfo
    using JET
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate, NonCartesianAcquisitionInfo
    using Ristretto.AbstractOperators
    using NamedDims

    # The target of every `@test_call`/`@test_opt` below: this package's own frames only. A bare
    # module would also match the vendored dependencies, which are submodules of it, and report
    # their inference problems -- such as the keyword splat in `override_parameters` -- as ours.
    const OWN_FRAMES = JET.LastFrameModuleExact(Ristretto)
    nx, ny, nc = 8, 8, 2

    img = rand(ComplexF32, nx, ny)
    ksp = rand(ComplexF32, nx, ny)
    ksp_coil = rand(ComplexF32, nx, ny, nc)
    smaps = coil_sensitivities(nx, ny, nc)
    wrapped_img = NamedDimsArray{(:x, :y)}(img)
    wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
    wrapped_ksp_coil = NamedDimsArray{(:kx, :ky, :coil)}(ksp_coil)
    wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
    mask = trues(nx, ny)
    cart_info = AcquisitionInfo(ksp_coil; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps)
    cart_info_named = AcquisitionInfo(wrapped_ksp_coil; image_size = (nx, ny), sensitivity_maps = wrapped_smaps)
    trajectory = rand(Float32, 2, 12, 2) .- 0.5f0
    dcf = rand(Float32, 12, 2)
    noncart_ksp = rand(ComplexF32, 12, 2)
    noncart_info = AcquisitionInfo(noncart_ksp; trajectory, dcf, image_size = (nx, ny))

    @test_opt target_modules = (OWN_FRAMES,) AcquisitionInfo(ksp; is3D = false, image_size = (nx, ny))
    @test_opt target_modules = (OWN_FRAMES,) CartesianAcquisitionInfo(ksp; is3D = false, image_size = (nx, ny), subsampling = mask)
    @test_opt target_modules = (OWN_FRAMES,) NonCartesianAcquisitionInfo(noncart_ksp; trajectory, dcf, image_size = (nx, ny))

    @test_opt target_modules = (OWN_FRAMES,) get_subsampling_operator(rand(ComplexF32, count(mask)), (nx, ny), mask)

    @test_opt target_modules = (OWN_FRAMES,) L2Image(0.1)
    @test_opt target_modules = (OWN_FRAMES,) L1Image(0.1)
    @test_opt target_modules = (OWN_FRAMES,) L1Wavelet2D(0.1)
    @test_opt target_modules = (OWN_FRAMES,) L1Wavelet3D(0.1)
    @test_opt target_modules = (OWN_FRAMES,) TotalVariation2D(0.1)
    @test_opt target_modules = (OWN_FRAMES,) TotalVariation3D(0.1)
    @test_opt target_modules = (OWN_FRAMES,) L1TemporalFourier(0.1; time_dim = 2)
    @test_opt target_modules = (OWN_FRAMES,) LowRank(0.1; time_dim = 2)
    @test_opt target_modules = (OWN_FRAMES,) RankLimit(2; time_dim = 2)

    @test_opt target_modules = (OWN_FRAMES,) UniformRandomSampling(2.0, 0.1)
    @test_opt target_modules = (OWN_FRAMES,) VariableDensitySampling(GaussianDistribution(), 2.0)
    @test_opt target_modules = (OWN_FRAMES,) VariableDensitySampling(PolynomialDistribution(), 2.0)
    @test_opt target_modules = (OWN_FRAMES,) PoissonDiskSampling(2.0)
    @test_opt target_modules = (OWN_FRAMES,) to_displayable_mask((:, trues(ny)), (nx, ny))
    @test_opt target_modules = (OWN_FRAMES,) coil_sensitivities(nx, ny, nc)
    @test_opt target_modules = (OWN_FRAMES,) ReconstructionConfig(; verbosity = Silent())
    @test_opt target_modules = (OWN_FRAMES,) SequentialExecutor()
    @test_opt target_modules = (OWN_FRAMES,) MultiThreadingExecutor()
    @test_opt target_modules = (OWN_FRAMES,) BartScaling()
    @test_opt target_modules = (OWN_FRAMES,) MeasurementBasedScaling()
    @test_opt target_modules = (OWN_FRAMES,) NoScaling()
    @test_opt target_modules = (OWN_FRAMES,) QuantileScaling()
    @test_opt target_modules = (OWN_FRAMES,) KSpaceNormScaling()
    for s in (
            QuantileScaling(), BartScaling(), MaxScaling(), StdScaling(), NoiseLevelScaling(),
            MeasurementBasedScaling(), KSpaceNormScaling(),
        )
        @test_opt target_modules = (OWN_FRAMES,) Ristretto.get_scale(s, cart_info, img, nothing)
    end
end

@testitem "JET exported API @test_call" tags = [:quality, :jet] begin
    using JET
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate
    using Ristretto.AbstractOperators
    using NamedDims

    # The target of every `@test_call`/`@test_opt` below: this package's own frames only. A bare
    # module would also match the vendored dependencies, which are submodules of it, and report
    # their inference problems -- such as the keyword splat in `override_parameters` -- as ours.
    const OWN_FRAMES = JET.LastFrameModuleExact(Ristretto)
    nx, ny, nc = 8, 8, 2

    img = rand(ComplexF32, nx, ny)
    ksp = rand(ComplexF32, nx, ny)
    ksp_coil = rand(ComplexF32, nx, ny, nc)
    smaps = coil_sensitivities(nx, ny, nc)
    wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
    wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
    mask = trues(nx, ny)
    cart_info = AcquisitionInfo(ksp_coil; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps)
    trajectory = rand(Float32, 2, 12, 2) .- 0.5f0
    dcf = rand(Float32, 12, 2)
    noncart_ksp = rand(ComplexF32, 12, 2)
    noncart_info = AcquisitionInfo(noncart_ksp; trajectory, dcf, image_size = (nx, ny))

    @test_call target_modules = (OWN_FRAMES,) AcquisitionInfo(ksp; is3D = false, image_size = (nx, ny))
    @test_call target_modules = (OWN_FRAMES,) get_fourier_operator(ksp, false; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) get_fourier_operator(wrapped_ksp; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) get_fourier_operator(noncart_info; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) get_sensitivity_map_operator(smaps, false; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) get_sensitivity_map_operator(wrapped_smaps; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) get_subsampling_operator(rand(ComplexF32, count(mask)), (nx, ny), mask)
    @test_call target_modules = (OWN_FRAMES,) get_encoding_operator(cart_info; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) get_encoding_operator(noncart_info; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) calculate(L2Image(0.1), img; threaded = false)
    @test_call target_modules = (OWN_FRAMES,) build_model(Eye(img), img, L2Image(0.1); threaded = false)
    @test_call target_modules = (OWN_FRAMES,) create_sampling_pattern(UniformRandomSampling(2.0, 0.1), (nx, ny))
    @test_call target_modules = (OWN_FRAMES,) to_displayable_mask((:, trues(ny)), (nx, ny))
    @test_call target_modules = (OWN_FRAMES,) coil_sensitivities(nx, ny, nc)
    @test_call target_modules = (OWN_FRAMES,) simulate_acquisition(img, AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps); inverse_crime_check = false, keep_sensitivity_maps = true)
    @test_call target_modules = (OWN_FRAMES,) ReconstructionConfig(; verbosity = Silent())
    @test_call target_modules = (OWN_FRAMES,) DirectReconstruction()
    @test_call target_modules = (OWN_FRAMES,) IterativeReconstruction(L2Image(0.01))
    @test_call target_modules = (OWN_FRAMES,) reconstruct(cart_info, IterativeReconstruction(L2Image(0.01); maxit = 2); threaded = false, verbosity = Silent())
    @test_call target_modules = (OWN_FRAMES,) reconstruct(cart_info; threaded = false, verbosity = Silent())
    @test_call target_modules = (OWN_FRAMES,) SequentialExecutor()
    @test_call target_modules = (OWN_FRAMES,) MultiThreadingExecutor()
    @test_call target_modules = (OWN_FRAMES,) BartScaling()
    @test_call target_modules = (OWN_FRAMES,) MeasurementBasedScaling()
    @test_call target_modules = (OWN_FRAMES,) NoScaling()
    @test_call target_modules = (OWN_FRAMES,) QuantileScaling(0.95)
    @test_call target_modules = (OWN_FRAMES,) KSpaceNormScaling(10)
end

@testitem "Benchmark case catalog smoke test" tags = [:quality] begin
    using Ristretto

    # The case catalog and Ristretto's reconstruction of each method, at the reduced size, one method per
    # case family: this guards against the harness (benchmark/run.jl) and the comparison suite
    # (RistrettoBenchmarking) breaking on an API change, without paying for a benchmark run.
    include(joinpath(pkgdir(Ristretto), "benchmark", "utils", "bench_utils.jl"))
    const BU = BenchUtils
    withenv("RISTRETTO_BENCH_SMALL" => "1", "RISTRETTO_BENCH_REAL_DATA" => "0") do
        for (id, method) in (
                ("shepp_logan_2d_1ch_cartesian", :tv),
                ("shepp_logan_2d_8ch_cartesian", :cgsense),
                ("shepp_logan_2d_8ch_radial", :gridding),
                ("shepp_logan_multislice_8ch_cartesian", :adjoint),
                ("shepp_logan_3d_8ch_cartesian", :adjoint),
                ("torso_cine_8ch_cartesian", :lowrank),
                ("torso_cine_8ch_radial", :gridding),
            )
            c = BU.get_case(id)
            @test method in BU.applicable_methods(c)
            tmin, tmed, x = BU.time_run(BU.ristretto_reconstructor(c, method; maxit = 2); warmup = 0, runs = 1)
            @test tmin > 0
            @test size(parent(x)) == size(c.reference)
            @test isfinite(BU.mag_nrmse(Array(parent(x)), c.reference))
        end
    end
end
