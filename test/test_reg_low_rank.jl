using TestItems

@testitem "LowRank regularization" tags = [:regularization] setup = [RegTestSetup] begin
    using LinearAlgebra
    using Ristretto.ProximalOperators: prox!, preallocate
    @testset "Constructor" begin
        reg = LowRank(0.1)
        @test reg.λ == 0.1
        @test isnothing(reg.time_dim)

        reg2 = LowRank(0.2; time_dim = 3)
        @test reg2.time_dim == 3

        @test_throws ArgumentError LowRank(0.1; time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float64, 8, 8, 10)
        reg = LowRank(0.1; time_dim = 3)
        op = get_operator(reg, x; threaded)
        # Should reshape to Casorati matrix: (8*8, 10)
        @test size(op, 1) == (64, 10)
        @test size(op, 2) == size(x)
        result = op * x
        @test size(result) == (64, 10)
    end

    @testset "get_operator with batch dims" for threaded in [false, true]
        x = randn(Float64, 8, 8, 10, 3)
        reg = LowRank(0.1; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 1) == (64, 10, 3)
        result = op * x
        @test size(result) == (64, 10, 3)
    end

    @testset "materialize" begin
        x = Variable(Float64, 8, 8, 10)
        reg = LowRank(0.1; time_dim = 3)
        term = Ristretto.materialize(reg, x; threaded = false)
        @test term !== nothing
    end

    # With dimensions after `time_dim` the penalty is one nuclear norm per batch index. It used to
    # hand the whole `(space, time, batch...)` array to `NuclearNorm`, which has no method for it
    # ("no `preallocate` method for NuclearNorm with an input of type Array{ComplexF64, 3}").
    @testset "batch dims: one Casorati matrix per batch index" for threaded in [false, true]
        x = randn(ComplexF64, 6, 5, 4, 3, 2)
        casorati(k...) = reshape(x[:, :, :, k...], 30, 4)
        expected = 0.1 * sum(sum(svdvals(casorati(k...))) for k in Iterators.product(1:3, 1:2))
        @test Ristretto.calculate(LowRank(0.1; time_dim = 3), x; threaded) ≈ expected
        f = Ristretto.materialize(LowRank(0.1; time_dim = 3), Variable(copy(x)); threaded).f
        X = Ristretto.get_operator(LowRank(0.1; time_dim = 3), x; threaded) * x
        # Both the allocating prox and the preallocated one a solver uses (it used to fail on the
        # buffers `NuclearNorm` sized for a slice view).
        for g in (f, preallocate(f, X))
            Y = similar(X)
            @test prox!(Y, g, X, 1.0) ≈ 0.1 * sum(sum(max.(svdvals(casorati(k...)) .- 0.1, 0)) for k in Iterators.product(1:3, 1:2))
            for k in Iterators.product(1:3, 1:2)
                F = svd(casorati(k...))
                @test Y[:, :, k...] ≈ F.U * Diagonal(max.(F.S .- 0.1, 0)) * F.Vt
            end
        end
        r = Ristretto.materialize(RankLimit(1; time_dim = 3), Variable(copy(x)); threaded).f
        R = similar(X)
        prox!(R, preallocate(r, X), X, 1.0)
        @test all(rank(R[:, :, k...]; atol = 1.0e-8) == 1 for k in Iterators.product(1:3, 1:2))
    end

    @testset "batch dims: reconstruct with a coupled batch dimension" begin
        # A temporal term along the batch dimension keeps it from being split into tasks, so the
        # low-rank term sees the batch dimension itself.
        x = rand(ComplexF32, 8, 8, 4, 3)
        acq = AcquisitionInfo(is3D = false, image_size = (8, 8), subsampling = (:, trues(8)))
        data = simulate_acquisition(x, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        regs = (LowRank(0.01; time_dim = 3), TemporalTotalVariation(0.01; time_dim = 4))
        rec = reconstruct(data, IterativeReconstruction(regs...; maxit = 5); verbosity = Silent())
        @test size(rec) == size(x)
    end

    @testset "get_affected_dims" begin
        reg = LowRank(0.1; time_dim = 3)
        @test Ristretto.get_affected_dims(reg, nothing, 1:4) == 1:3

        reg_named = LowRank(0.1; time_dim = :time)
        dims_named = Ristretto.get_affected_dims(reg_named, nothing, (:x, :y, :time, :coil))
        @test dims_named == (:x, :y, :time)
    end

    @testset "LowRank prevents task splitting across time" begin
        nx, ny, nt = 8, 8, 5
        ksp = NamedDimsArray{(:kx, :ky, :time)}(rand(ComplexF32, nx, ny, nt))
        acq = AcquisitionInfo(ksp)
        config = ReconstructionConfig(; verbosity = Silent())

        plan_noreg = Ristretto.get_task_splitting_plan(acq, DirectReconstruction(), config)
        @test !isnothing(plan_noreg) # :time is a batch dim without regularization

        reg = LowRank(0.1; time_dim = :time)
        plan = Ristretto.get_task_splitting_plan(acq, IterativeReconstruction(reg), config)
        @test isnothing(plan) # LowRank couples the time dimension
    end
end

@testitem "RankLimit regularization" tags = [:regularization] setup = [RegTestSetup] begin
    @testset "Constructor" begin
        reg = RankLimit(5)
        @test reg.max_rank == 5
        @test isnothing(reg.time_dim)

        @test_throws ArgumentError RankLimit(0)
        @test_throws ArgumentError RankLimit(-1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float64, 8, 8, 10)
        reg = RankLimit(3; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 1) == (64, 10)
        @test size(op, 2) == size(x)
    end

    @testset "materialize" begin
        x = Variable(Float64, 8, 8, 10)
        reg = RankLimit(3; time_dim = 3)
        term = Ristretto.materialize(reg, x; threaded = false)
        @test term !== nothing
    end

    @testset "reconstruct end-to-end" begin
        # `IndBallRank.is_proximable` used to be hardcoded `false` even though `prox!` was fully
        # implemented, so this term could never be parsed by any solver ("Sorry, I cannot parse
        # this problem for any of the provided solvers").
        using NamedDims
        x = NamedDimsArray{(:x, :y, :time)}(rand(ComplexF32, 16, 16, 6))
        acq = AcquisitionInfo(is3D = false, image_size = (16, 16), subsampling = (:, trues(16)))
        data = simulate_acquisition(x, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
        rec = reconstruct(data, IterativeReconstruction(RankLimit(2; time_dim = :time); maxit = 5); verbosity = Silent())
        @test size(rec) == size(x)
    end
end

@testitem "LocallyLowRank regularization" tags = [:regularization, :gpu] setup = [RegTestSetup, ProxOf, GpuEnvSetup, GpuHelpers] begin
    using LinearAlgebra
    using Random: Xoshiro

    function reference_llr(x, λ, block_size, nt)
        value = 0.0
        for i in 1:block_size[1]:size(x, 1), j in 1:block_size[2]:size(x, 2)
            irange = i:min(i + block_size[1] - 1, size(x, 1))
            jrange = j:min(j + block_size[2] - 1, size(x, 2))
            M = reshape(x[irange, jrange, :], length(irange) * length(jrange), nt)
            value += λ * sum(svdvals(M))
        end
        return value
    end

    @testset "Constructor" begin
        reg = LocallyLowRank(0.1; block_size = 4)
        @test reg.λ == 0.1
        @test reg.block_size == 4
        @test isnothing(reg.time_dim)
        @test LocallyLowRank(0.1; block_size = (4, 2), time_dim = 3).block_size == (4, 2)
        @test_throws ArgumentError LocallyLowRank(0.1; block_size = 0)
        @test_throws ArgumentError LocallyLowRank(0.1; block_size = 4, time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(8, 8, 4)
        op = get_operator(LocallyLowRank(0.1; block_size = 4, time_dim = 3), x; threaded)
        @test op isa Eye
        @test op * x ≈ x
    end

    @testset "materialize matches blockwise nuclear norms" for threaded in [false, true]
        x = randn(ComplexF64, 8, 6, 4)
        λ = 0.1
        reg = LocallyLowRank(λ; block_size = (4, 3), time_dim = 3)
        @test Ristretto.calculate(reg, x; threaded) ≈ reference_llr(x, λ, (4, 3), 4)
    end

    @testset "incomplete boundary blocks" begin
        x = randn(7, 5, 3)
        λ = 0.2
        reg = LocallyLowRank(λ; block_size = 4, time_dim = 3)
        @test Ristretto.calculate(reg, x; threaded = false) ≈ reference_llr(x, λ, (4, 4), 3)
    end

    @testset "batch dimensions are handled independently" begin
        x = randn(4, 4, 3, 2)
        λ = 0.15
        reg = LocallyLowRank(λ; block_size = 4, time_dim = 3)
        expected = reference_llr(x[:, :, :, 1], λ, (4, 4), 3) + reference_llr(x[:, :, :, 2], λ, (4, 4), 3)
        @test Ristretto.calculate(reg, x; threaded = false) ≈ expected
    end

    @testset "prox is exact blockwise singular value thresholding" for threaded in [false, true]
        x = randn(ComplexF64, 8, 4, 3)
        λ, γ = 0.1, 0.7
        reg = LocallyLowRank(λ; block_size = 4, time_dim = 3)
        var = Variable(x)
        term = Ristretto.materialize(reg, var; threaded)
        f = SO.weighted_function(term)
        y = similar(x)
        fy = PC.prox!(y, f, x, γ)

        expected = similar(x)
        expected_value = 0.0
        for i in 1:4:8, j in 1:4:4
            M = reshape(x[i:(i + 3), j:min(j + 3, 4), :], :, 3)
            F = svd(M)
            σ = max.(0.0, F.S .- λ * γ)
            expected[i:(i + 3), j:min(j + 3, 4), :] = reshape(F.U * Diagonal(σ) * F.Vt, 4, :, 3)
            expected_value += λ * sum(σ)
        end
        @test y ≈ expected
        @test fy ≈ expected_value
    end

    @testset "NamedDimsArray input" begin
        x = NamedDimsArray{(:x, :y, :time)}(randn(4, 4, 3))
        reg = LocallyLowRank(0.1; block_size = 2, time_dim = :time)
        op = get_operator(reg, x; threaded = false)
        @test op isa Ristretto.NamedDimsOp
        @test Ristretto.calculate(reg, x; threaded = false) ≈
            reference_llr(unname(x), 0.1, (2, 2), 3)
    end

    @testset "get_affected_dims couples space and time" begin
        ksp = randn(ComplexF32, 8, 8, 10)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        reg = LocallyLowRank(0.1f0; block_size = 4, time_dim = 3)
        @test Ristretto.get_affected_dims(reg, info, 1:3) == 1:3
        reg_named = LocallyLowRank(0.1f0; block_size = 4, time_dim = :time)
        @test Ristretto.get_affected_dims(reg_named, info, (:x, :y, :time)) == (:x, :y, :time)
    end

    @testset "block size validation" begin
        x = Variable(randn(4, 4, 2))
        reg = LocallyLowRank(0.1; block_size = 8, time_dim = 3)
        @test_throws ArgumentError Ristretto.materialize(reg, x; threaded = false)
        reg_wrong_rank = LocallyLowRank(0.1; block_size = (4, 4, 4), time_dim = 3)
        @test_throws ArgumentError Ristretto.materialize(reg_wrong_rank, x; threaded = false)
    end

    @testset "scale_regularization" begin
        reg = Ristretto.scale_regularization(
            LocallyLowRank(0.2; block_size = 4, time_dim = 3), 2.5
        )
        @test reg.λ ≈ 0.5
        @test reg.block_size == 4
        @test reg.time_dim == 3
        @test reg.shift == :none
    end

    # The device prox gathers every block at once and thresholds them in one batch; blocks of
    # edge 4 do not tile 12 × 10, so the zero-padded edge blocks are covered too.
    @testset "on a device" begin
        x = randn(ComplexF32, 12, 10, 6, 2)
        # Each call builds its own regularizer so that a shifted grid draws the same origin on
        # the host and on the device.
        for make_reg in (
                () -> LocallyLowRank(0.3f0; block_size = 4, time_dim = 3),
                () -> LocallyLowRank(0.3f0; block_size = (2, 5), time_dim = 3, shift = :fixed, rng = Xoshiro(3)),
                () -> LocallyLowRank(0.3f0; block_size = 2, time_dim = 3),
            )
            test_on_devices(x -> prox_of(make_reg(), x, 0.7f0), x; backends = all_backends())
        end
    end

    # A CUDA device picks its batched solver by slice shape: a full SVD of slices up to 32 × 32,
    # an eigendecomposition of the short side's Gram matrix when only one side fits, and the
    # generic device path otherwise. Every shape must agree with the host.
    @testset "batched singular value thresholding on a device: $(size(A))" for A in (
            randn(ComplexF32, 16, 6, 5), randn(Float32, 64, 6, 5), randn(ComplexF32, 6, 64, 5),
            randn(ComplexF32, 40, 36, 2),
        )
        # The thresholded slices, then the thresholded and the plain nuclear norms, on A's storage.
        function svt(A)
            B = copy(A)
            thresholded = Ristretto.ProximalOperators.batched_svt!(B, 1.0f0)
            nuclear = Ristretto.ProximalOperators.batched_nuclear_norm(A)
            return vcat(vec(B), copyto!(similar(B, 2), eltype(A)[thresholded, nuclear]))
        end
        test_on_devices(svt, A; backends = all_backends())
    end
end

@testitem "LocallyLowRank grid shifts" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    using LinearAlgebra
    using Random

    @testset "Constructor" begin
        @test LocallyLowRank(0.1; block_size = 4).shift == :none
        @test LocallyLowRank(0.1; block_size = 4, shift = :random).shift == :random
        @test_throws ArgumentError LocallyLowRank(0.1; block_size = 4, shift = :sometimes)
    end

    @testset "a shifted grid is exactly the unshifted grid on a circularly shifted image" begin
        x = randn(MersenneTwister(11), ComplexF64, 8, 8, 5)
        unshifted = LocallyLowRank(0.3; block_size = 4, time_dim = 3)
        shifted = LocallyLowRank(0.3; block_size = 4, time_dim = 3, shift = :fixed, rng = MersenneTwister(7))
        f = functions_of(shifted, x)
        offset = f.offset[]
        @test offset != (0, 0)   # the seed above must actually produce a shift for this test to mean anything

        y = similar(x)
        PC.prox!(y, f, x, 1.0)
        # The wrapped tiling is still a permutation of the voxels, so shifting the image by -offset,
        # applying the unshifted prox and shifting back must reproduce it bit for bit.
        rolled = circshift(x, (-offset[1], -offset[2], 0))
        expected = circshift(first(prox_of(unshifted, rolled)), (offset[1], offset[2], 0))
        @test y ≈ expected
    end

    @testset "a shifted grid needs a divisible image" begin
        x = Variable(randn(7, 8, 3))
        reg = LocallyLowRank(0.1; block_size = 4, time_dim = 3, shift = :fixed)
        @test_throws ArgumentError Ristretto.materialize(reg, x; threaded = false)
    end

    @testset "shift=:random redraws the origin on every prox call" begin
        x = randn(MersenneTwister(3), 8, 8, 4)
        reg = LocallyLowRank(0.3; block_size = 4, time_dim = 3, shift = :random, rng = MersenneTwister(5))
        f = functions_of(reg, x)
        y = similar(x)
        offsets = map(1:12) do _
            PC.prox!(y, f, x, 1.0)
            f.offset[]
        end
        @test length(unique(offsets)) > 1
    end

    @testset "scale_regularization keeps the shift policy" begin
        reg = Ristretto.scale_regularization(
            LocallyLowRank(0.2; block_size = 4, time_dim = 3, shift = :random), 2.0
        )
        @test reg.λ ≈ 0.4
        @test reg.shift == :random
    end
end

@testitem "MultiScaleLowRank regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    using LinearAlgebra

    @testset "Constructor" begin
        reg = MultiScaleLowRank(0.1; block_sizes = (4, 8))
        @test reg.λ == 0.1
        @test reg.block_sizes == (4, 8)
        @test reg.weights === nothing
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = ())
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = (4, 0))
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = (4, 8), weights = [1.0])
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = (4,), shift = :maybe)

        reg_vec = MultiScaleLowRank([0.1, 0.2]; block_sizes = (4, 8))
        @test reg_vec.λ == [0.1, 0.2]
        @test_throws ArgumentError MultiScaleLowRank([0.1, 0.2, 0.3]; block_sizes = (4, 8))
        @test_throws ArgumentError MultiScaleLowRank("bogus"; block_sizes = (4, 8))
    end

    @testset "per-scale λ gives each scale its own threshold" for threaded in [false, true]
        x = randn(ComplexF64, 8, 8, 5)
        scales = (4, 8)
        λs = [0.1, 0.4]
        mslr = MultiScaleLowRank(λs; block_sizes = scales, time_dim = 3)
        expected_value = sum(
            Ristretto.calculate(LocallyLowRank(λ; block_size = b, time_dim = 3), x) / 2
                for (λ, b) in zip(λs, scales)
        )
        @test Ristretto.calculate(mslr, x; threaded) ≈ expected_value

        expected_prox = sum(
            (1 / 2) .* first(prox_of(LocallyLowRank(λ; block_size = b, time_dim = 3), x, 0.7))
                for (λ, b) in zip(λs, scales)
        )
        @test first(prox_of(mslr, x, 0.7)) ≈ expected_prox

        # A uniform vector must match the scalar form exactly.
        mslr_uniform_vec = MultiScaleLowRank(fill(0.3, length(scales)); block_sizes = scales, time_dim = 3)
        mslr_scalar = MultiScaleLowRank(0.3; block_sizes = scales, time_dim = 3)
        @test Ristretto.calculate(mslr_uniform_vec, x; threaded) ≈
            Ristretto.calculate(mslr_scalar, x; threaded)
    end

    @testset "get_operator" begin
        x = randn(8, 8, 4)
        reg = MultiScaleLowRank(0.1; block_sizes = (4, 8), time_dim = 3)
        op = get_operator(reg, x; threaded = false)
        @test op isa AbstractOperators.Eye
        @test size(op) == (size(x), size(x))
    end

    @testset "a single scale reduces to LocallyLowRank" for threaded in [false, true]
        x = randn(ComplexF64, 8, 8, 5)
        mslr = MultiScaleLowRank(0.3; block_sizes = (4,), time_dim = 3)
        llr = LocallyLowRank(0.3; block_size = 4, time_dim = 3)
        @test Ristretto.calculate(mslr, x; threaded) ≈
            Ristretto.calculate(llr, x; threaded)
        @test first(prox_of(mslr, x)) ≈ first(prox_of(llr, x))
    end

    @testset "the prox is the weighted average of the per-scale proxes" begin
        x = randn(ComplexF64, 8, 8, 5)
        scales = (2, 4, 8)
        weights = [0.2, 0.5, 0.3]
        mslr = MultiScaleLowRank(0.3; block_sizes = scales, time_dim = 3, weights)
        expected = sum(
            w .* first(prox_of(LocallyLowRank(0.3; block_size = b, time_dim = 3), x, 0.7))
                for (w, b) in zip(weights, scales)
        )
        @test first(prox_of(mslr, x, 0.7)) ≈ expected
    end

    @testset "the value is the weighted average of the per-scale penalties" begin
        x = randn(8, 8, 4)
        scales = (4, 8)
        mslr = MultiScaleLowRank(0.2; block_sizes = scales, time_dim = 3)
        expected = sum(
            Ristretto.calculate(LocallyLowRank(0.2; block_size = b, time_dim = 3), x) / 2
                for b in scales
        )
        @test Ristretto.calculate(mslr, x) ≈ expected
    end

    @testset "get_affected_dims couples space and time" begin
        reg = MultiScaleLowRank(0.1; block_sizes = (2, 4), time_dim = 3)
        @test Ristretto.get_affected_dims(reg, nothing, 1:4) == 1:3
    end

    @testset "scale_regularization" begin
        reg = Ristretto.scale_regularization(
            MultiScaleLowRank(0.2; block_sizes = (4, 8), time_dim = 3), 2.5
        )
        @test reg.λ ≈ 0.5
        @test reg.block_sizes == (4, 8)

        reg_vec = Ristretto.scale_regularization(
            MultiScaleLowRank([0.2, 0.4]; block_sizes = (4, 8), time_dim = 3), 2.5
        )
        @test reg_vec.λ ≈ [0.5, 1.0]
        @test reg_vec.block_sizes == (4, 8)
    end
end
