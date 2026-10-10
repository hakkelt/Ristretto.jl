using TestItems

@testitem "TotalGeneralizedVariation2D regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using Ristretto
    using Ristretto: get_operator, get_encoding_operator, materialize, get_affected_dims, scale_regularization
    using Ristretto.StructuredOptimization
    using Ristretto.AbstractOperators
    using NamedDims


    @testset "Constructor" begin
        reg = TotalGeneralizedVariation2D(0.1)
        @test reg.λ == 0.1
        @test reg.ratio == 2.0
        @test TotalGeneralizedVariation2D(0.1; ratio = 3.0).ratio == 3.0
        @test_throws ArgumentError TotalGeneralizedVariation2D(-0.1)
        @test_throws ArgumentError TotalGeneralizedVariation2D(0.1; ratio = 0)
    end

    @testset "get_operator is the flattened gradient" for threaded in [false, true]
        x = randn(8, 8)
        op = get_operator(TotalGeneralizedVariation2D(0.1), x; threaded)
        @test size(op, 1) == (64, 2)
        @test op * x ≈ reshape(get_operator(TotalVariation2D(0.1), x; threaded) * x, 64, 2)
        @test_throws ArgumentError get_operator(TotalGeneralizedVariation2D(0.1), randn(8); threaded)
    end

    @testset "materialize introduces one auxiliary field" begin
        x = Variable(randn(8, 8, 3))
        terms, auxiliaries = Ristretto.materialize_with_auxiliaries(
            TotalGeneralizedVariation2D(0.1), x; threaded = false
        )
        @test length(auxiliaries) == 1
        w = auxiliaries[1]
        # one vector per voxel of the whole array, in `Variation`'s layout
        @test size(~w) == (8 * 8 * 3, 2)
        @test all(iszero, ~w)   # auxiliaries start at zero
        @test terms isa Ristretto.StructuredOptimization.TermSet
        # `materialize` gives the same terms without the auxiliaries
        @test Ristretto.materialize(TotalGeneralizedVariation2D(0.1), x; threaded = false) isa
            Ristretto.StructuredOptimization.TermSet
    end

    @testset "the symmetrized-gradient operator acts per batch slice" begin
        # Folding the batch into a spatial extent would let differences run across slice boundaries; this
        # checks that they do not.
        batched = Ristretto._tgv_symmetrized_operator(Float64, (8, 8), 3; threaded = false)
        single = SymmetrizedVariation(Float64, (8, 8); threaded = false)
        w = randn(8 * 8 * 3, 2)
        result = batched * w
        unfolded = reshape(w, 64, 3, 2)
        for k in 1:3
            @test result[((k - 1) * 64 + 1):(k * 64), :] ≈ single * unfolded[:, k, :]
        end
        # and it is still a correct adjoint pair after all the reshaping and permuting
        y = randn(8 * 8 * 3, 3)
        @test dot(batched * w, y) ≈ dot(w, batched' * y)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 4)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test Ristretto.get_affected_dims(TotalGeneralizedVariation2D(0.1f0), info, 1:3) == 1:2
    end

    @testset "scale_regularization" begin
        reg = Ristretto.scale_regularization(TotalGeneralizedVariation2D(0.2; ratio = 3.0), 2.5)
        @test reg.λ ≈ 0.5
        @test reg.ratio == 3.0
    end
end

@testitem "TotalGeneralizedVariation2D denoising behaviour" tags = [:regularization, :minimizer] setup = [TestHelpers] begin
    using Test
    using LinearAlgebra
    using Random
    using Ristretto
    using Ristretto: get_operator, get_encoding_operator, materialize, get_affected_dims, scale_regularization
    using Ristretto.StructuredOptimization
    using Ristretto.AbstractOperators


    # ADMM settles within 200 iterations here: the errors below move in the fourth digit beyond.
    function denoise(reg, noisy; maxit = 200)
        model, x, _ = Ristretto.build_model_with_variables(
            Eye(noisy), noisy, (reg,); threaded = false, x₀ = copy(noisy),
        )
        solve(model, ADMM(; maxit, rho = 1.0))
        return copy(~x)
    end

    # A ramp with a jump in it: the case first-order TV gets wrong (staircasing on the ramp) and
    # second-order TV gets wrong (blurring across the jump).
    n = 32
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j for i in 1:n, j in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(2), n, n)
    error_to_truth(z) = relative_error(z, truth)

    @testset "TGV beats TV on a ramp with an edge" begin
        tv = denoise(TotalVariation2D(0.05), noisy)
        tgv = denoise(TotalGeneralizedVariation2D(0.05), noisy)
        @test error_to_truth(tgv) < error_to_truth(tv)
        @test error_to_truth(tgv) < error_to_truth(noisy)
    end

    @testset "a large ratio degenerates to total variation" begin
        # Driving the second-order weight up forces the auxiliary field to a constant, which leaves exactly
        # the total variation term -- TV is a limiting case of TGV, not a different model.
        tv = denoise(TotalVariation2D(0.05), noisy)
        tgv = denoise(TotalGeneralizedVariation2D(0.05; ratio = 1.0e4), noisy)
        @test norm(tgv .- tv) / norm(tv) < 0.05
    end

    @testset "batch dimensions are denoised independently" begin
        # Both the data term and the penalty are separable across the batch, so the minimizers coincide
        # slice by slice, and so do ADMM's iterates (to 1e-8 after 150 iterations). The second slice is a
        # different image, so a leak across the batch boundary would show up as a disagreement with the
        # slice reconstructed on its own.
        other = [0.03 * i for i in 1:n, _ in 1:n] .+ 0.05 .* randn(MersenneTwister(4), n, n)
        stacked = cat(noisy, other; dims = 3)
        result = denoise(TotalGeneralizedVariation2D(0.05), stacked; maxit = 150)
        @test result[:, :, 1] ≈ denoise(TotalGeneralizedVariation2D(0.05), noisy; maxit = 150) rtol = 1.0e-6
        @test result[:, :, 2] ≈ denoise(TotalGeneralizedVariation2D(0.05), other; maxit = 150) rtol = 1.0e-6
    end
end

@testitem "Infimal-convolution total variation via components" tags = [:regularization, :components] setup = [TestHelpers] begin
    using Test
    using LinearAlgebra
    using Random
    using Ristretto
    using Ristretto: get_operator, get_encoding_operator, materialize, get_affected_dims, scale_regularization
    using Ristretto.StructuredOptimization
    using Ristretto.AbstractOperators


    # The infimal convolution of first- and second-order TV needs no regularization type of its own: it is
    # exactly an image decomposition into a piecewise-constant component and a piecewise-linear one, which
    # the component machinery already expresses.
    n = 32
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j for i in 1:n, j in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(3), n, n)
    error_to_truth(z) = relative_error(z, truth)

    components = (
        Component(:cartoon, TotalVariation2D(0.05)),
        Component(:ramp, SecondOrderTotalVariation2D(0.05)),
    )

    model, vars, auxiliaries = Ristretto.build_model(
        Eye(noisy), noisy, components; threaded = false, x₀s = (copy(noisy), zero(noisy))
    )
    @test auxiliaries == ()
    solve(model, ADMM(maxit = 200, rho = 1.0))
    total = reduce(+, map(v -> copy(~v), vars))

    @test error_to_truth(total) < error_to_truth(noisy)

    # It must also beat plain first-order TV on this ramp-plus-edge image, which is the whole point of
    # splitting the image into a cartoon and a ramp part.
    tv_model, tv_x, _ = Ristretto.build_model_with_variables(
        Eye(noisy), noisy, (TotalVariation2D(0.05),); threaded = false, x₀ = copy(noisy),
    )
    solve(tv_model, ADMM(maxit = 200, rho = 1.0))
    @test error_to_truth(total) < error_to_truth(copy(~tv_x))
end

@testitem "TotalGeneralizedVariation2D runs through reconstruct" tags = [:regularization, :integration] begin
    using Test
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using Random
    using Ristretto
    using Ristretto: get_encoding_operator
    using Ristretto.StructuredOptimization

    # Regression: TGV was only ever exercised through hand-built models passed to `solve`, so the
    # public entry point had no coverage. The data term used to be rewritten into its
    # normal-operator form while the model was being built, before the regularizations had
    # declared their auxiliary variables, and the operator that form stored spanned the image
    # alone -- so once TGV added its auxiliary field the solver's `x0` spanned
    # (image, auxiliary) while the stored operator did not, and ADMM rejected the pair with
    # "A'b must have the same size as x0". The rewrite now happens when the problem is parsed,
    # against the variables it actually has.
    Random.seed!(20260829)
    nx, ny, nc = 32, 32, 4
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    smaps = coil_sensitivities(nx, ny, nc)
    pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15), (nx, ny))
    acq = simulate_acquisition(img_true, AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern); inverse_crime_check = false, keep_sensitivity_maps = true)

    img_recon = reconstruct(
        acq,
        IterativeReconstruction(TotalGeneralizedVariation2D(0.005); algorithm = ADMM(rho = 1.0), maxit = 50); verbosity = Silent()
    )
    @test size(img_recon) == (nx, ny)
    @test all(isfinite, img_recon)
    # A sign-flipped or diverged solve lands near 2.0; measured ≈0.29 at 50 iterations.
    @test norm(img_recon - img_true) / norm(img_true) < 0.6

    # TGV is the term that contributes an auxiliary variable; a first-order TV term is not.
    _, _, tgv_aux = Ristretto.build_model_with_variables(
        get_encoding_operator(acq), acq.kspace_data, (TotalGeneralizedVariation2D(0.005),); threaded = false,
    )
    @test length(tgv_aux) == 1
    tv_terms, _, tv_aux = Ristretto.build_model_with_variables(
        get_encoding_operator(acq), acq.kspace_data, (TotalVariation2D(0.001),); threaded = false,
    )
    @test tv_aux == ()
    # The data term itself is the plain least-squares one, in both models: the normal-operator
    # form is what the parser makes of it, and for this encoding operator it does.
    SO = Ristretto.StructuredOptimization
    PO = Ristretto.ProximalOperators
    ls_term = only(filter(t -> t.f isa PO.SqrNormL2, collect(tv_terms)))
    ls_op = SO.extract_operators(SO.extract_variables(tv_terms), ls_term)
    @test SO.with_normal_op(ls_term.f, ls_op, SO.displacement(ls_term), ls_term.lambda) isa
        SO.SqrNormL2WithNormalOp
end
@testitem "TotalGeneralizedVariation3D regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using Ristretto
    using Ristretto: get_operator, materialize, get_affected_dims, scale_regularization
    using Ristretto.StructuredOptimization
    using Ristretto.AbstractOperators


    @testset "Constructor" begin
        reg = TotalGeneralizedVariation3D(0.1)
        @test reg.λ == 0.1
        @test reg.ratio == 2.0
        @test TotalGeneralizedVariation3D(0.1; ratio = 3.0).ratio == 3.0
        @test_throws ArgumentError TotalGeneralizedVariation3D(-0.1)
        @test_throws ArgumentError TotalGeneralizedVariation3D(0.1; ratio = 0)
    end

    @testset "get_operator is the flattened 3D gradient" for threaded in [false, true]
        x = randn(6, 6, 6)
        op = get_operator(TotalGeneralizedVariation3D(0.1), x; threaded)
        @test size(op, 1) == (216, 3)
        @test op * x ≈ reshape(get_operator(TotalVariation3D(0.1), x; threaded) * x, 216, 3)
        @test_throws ArgumentError get_operator(TotalGeneralizedVariation3D(0.1), randn(6, 6); threaded)
    end

    @testset "materialize introduces one auxiliary field with three components" begin
        x = Variable(randn(4, 4, 4, 2))
        terms, auxiliaries = Ristretto.materialize_with_auxiliaries(
            TotalGeneralizedVariation3D(0.1), x; threaded = false
        )
        @test length(auxiliaries) == 1
        w = auxiliaries[1]
        @test size(~w) == (4 * 4 * 4 * 2, 3)
        @test all(iszero, ~w)
        @test terms isa Ristretto.StructuredOptimization.TermSet
    end

    @testset "the symmetrized-gradient operator acts per batch slice" begin
        # Six independent components in 3D: the entries of the symmetric 3x3 matrix ℰw.
        batched = Ristretto._tgv_symmetrized_operator(Float64, (4, 4, 4), 2; threaded = false)
        single = SymmetrizedVariation(Float64, (4, 4, 4); threaded = false)
        w = randn(64 * 2, 3)
        result = batched * w
        @test size(result) == (128, 6)
        unfolded = reshape(w, 64, 2, 3)
        for k in 1:2
            @test result[((k - 1) * 64 + 1):(k * 64), :] ≈ single * unfolded[:, k, :]
        end
        y = randn(64 * 2, 6)
        @test dot(batched * w, y) ≈ dot(w, batched' * y)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 6, 6, 6, 4)
        info = AcquisitionInfo(ksp; image_size = (6, 6, 6))
        @test Ristretto.get_affected_dims(TotalGeneralizedVariation3D(0.1f0), info, 1:4) == 1:3
    end

    @testset "scale_regularization" begin
        reg = Ristretto.scale_regularization(TotalGeneralizedVariation3D(0.2; ratio = 3.0), 2.5)
        @test reg.λ ≈ 0.5
        @test reg.ratio == 3.0
    end
end

@testitem "TotalGeneralizedVariation3D denoising behaviour" tags = [:regularization, :minimizer] setup = [TestHelpers] begin
    using Test
    using LinearAlgebra
    using Random
    using Ristretto
    using Ristretto.StructuredOptimization
    using Ristretto.AbstractOperators


    function denoise(reg, noisy; maxit = 100)
        model, x, _ = Ristretto.build_model_with_variables(
            Eye(noisy), noisy, (reg,); threaded = false, x₀ = copy(noisy),
        )
        solve(model, ADMM(; maxit, rho = 1.0))
        return copy(~x)
    end

    # A volumetric ramp with a jump: the 3D analogue of the 2D staircasing case.
    n = 16
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j + 0.01 * k for i in 1:n, j in 1:n, k in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(5), n, n, n)
    error_to_truth(z) = relative_error(z, truth)

    @test error_to_truth(denoise(TotalGeneralizedVariation3D(0.05), noisy)) <
        error_to_truth(denoise(TotalVariation3D(0.05), noisy))
end
