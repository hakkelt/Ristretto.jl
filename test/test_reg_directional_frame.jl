using TestItems

@testitem "L1Shearlet Regularization" tags = [:regularization] setup = [RegTestSetup] begin
    using LinearAlgebra
    @testset "constructor" begin
        reg = L1Shearlet(0.1)
        @test reg.λ == 0.1
        @test reg.directions == (8, 8, 16)
        @test L1Shearlet(0.1; directions = (4, 8)).directions == (4, 8)
        @test_throws ArgumentError L1Shearlet(0.1; directions = (0, 4))
    end

    @testset "get_operator" for T in [Float32, ComplexF64], threaded in [false, true]
        x = rand(T, 32, 24)
        op = get_operator(L1Shearlet(0.1; directions = (4, 8)), x; threaded)
        @test size(op) == ((32, 24, 13), (32, 24))
        @test domain_type(op) == T && codomain_type(op) == T
        @test op' * (op * x) ≈ x rtol = 1.0e-5        # Parseval frame
    end

    @testset "batched input" for threaded in [false, true]
        x = rand(ComplexF64, 32, 32, 3, 2)
        op = get_operator(L1Shearlet(0.1; directions = (4, 8)), x; threaded)
        @test op isa BatchOp
        y = op * x
        @test size(y) == (32, 32, 13, 3, 2)
        @test op' * y ≈ x
        @test y[:, :, :, 2, 1] ≈ get_operator(L1Shearlet(0.1; directions = (4, 8)), x[:, :, 2, 1]) * x[:, :, 2, 1]
    end

    @testset "dimension check" begin
        @test_throws ArgumentError get_operator(L1Shearlet(0.1), rand(10))
    end

    @testset "materialize" begin
        x = rand(ComplexF64, 32, 32)
        reg = L1Shearlet(0.2; directions = (4, 8))
        @test Ristretto.calculate(reg, x; threaded = false) ≈ 0.2 * sum(abs, get_operator(reg, x) * x)
        @test Ristretto.scale_regularization(reg, 3).λ ≈ 0.6
    end
end

@testitem "L1Surfacelet Regularization" tags = [:regularization] setup = [RegTestSetup] begin
    using LinearAlgebra
    @testset "constructor" begin
        reg = L1Surfacelet(0.1)
        @test reg.levels == (1, 1, 1)
        @test_throws ArgumentError L1Surfacelet(0.1; levels = (-1,))
    end

    @testset "get_operator" for T in [Float64, ComplexF32]
        x = rand(T, 16, 16, 8)
        op = get_operator(L1Surfacelet(0.1; levels = (0, 1)), x)
        @test size(op) == ((16, 16, 8, 1 + 3 + 12), (16, 16, 8))
        @test domain_type(op) == T
        @test op' * (op * x) ≈ x rtol = 1.0e-5
    end

    @testset "batched input" begin
        x = rand(ComplexF64, 16, 16, 8, 2)
        op = get_operator(L1Surfacelet(0.1; levels = (0, 1)), x)
        @test op isa BatchOp
        @test op' * (op * x) ≈ x
    end

    @testset "dimension check" begin
        @test_throws ArgumentError get_operator(L1Surfacelet(0.1), rand(10, 10))
    end

    @testset "materialize" begin
        x = rand(ComplexF64, 16, 16, 8)
        reg = L1Surfacelet(0.2; levels = (0, 1))
        @test Ristretto.calculate(reg, x; threaded = false) ≈ 0.2 * sum(abs, get_operator(reg, x) * x)
    end
end

@testitem "L1Shearlet in a reconstruction" tags = [:regularization, :reconstruction] setup = [RegTestSetup] begin
    # The frame is redundant, so the term is not proximable on the image: the default algorithm
    # choice must fall back to ADMM, and the result must beat the direct reconstruction.
    using LinearAlgebra
    nx, ny, nc = 32, 32, 4
    img = zeros(ComplexF32, nx, ny)
    img[9:24, 9:24] .= 1.0f0
    img[13:20, 13:20] .= 0.5f0
    data = simulate_acquisition(
        img,
        AcquisitionInfo(;
            is3D = false, image_size = (nx, ny),
            sensitivity_maps = coil_sensitivities(nx, ny, nc),
            subsampling = create_sampling_pattern(RegularLatticeSampling(2; center_fraction = 0.2), (nx, ny)),
        ); inverse_crime_check = false, keep_sensitivity_maps = true
    )
    terms, _, _ = Ristretto.build_model_with_variables(
        Ristretto.get_encoding_operator(data), data.kspace_data, (L1Shearlet(1.0f-3),); x₀ = zeros(ComplexF32, nx, ny)
    )
    alg = Ristretto._select_algorithm(terms, (FISTA(), ADMM()))
    @test typeof(alg).parameters[1] <: Ristretto.ProximalAlgorithms.ADMMIteration
    x̂ = reconstruct(data, IterativeReconstruction(L1Shearlet(1.0f-3; directions = (4, 8)); maxit = 40); verbosity = Silent())
    x_direct = reconstruct(data; verbosity = Silent())
    @test size(x̂) == (nx, ny)
    @test norm(x̂ - img) / norm(img) < norm(x_direct - img) / norm(img)
end
