@testitem "AffineAdd: basic operations" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 5, 6
    A = randn(n, m)
    opA = MatrixOp(A)
    d = randn(n)
    T = AffineAdd(opA, d)
    x1 = randn(m)
    @test norm(T * x1 - (A * x1 + d)) < 1.0e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1.0e-9
    @test displacement(T) == d
    @test norm(remove_displacement(T) * x1 - A * x1) < 1.0e-9

    T_neg = AffineAdd(opA, d, false)
    @test sign(T_neg) == -1
    @test norm(T_neg * x1 - (A * x1 - d)) < 1.0e-9

    T_scalar = AffineAdd(opA, pi)
    @test norm(T_scalar * x1 - (A * x1 .+ pi)) < 1.0e-9

    @test_throws DimensionMismatch AffineAdd(MatrixOp(randn(2, 5)), randn(5))
    @test_throws ErrorException AffineAdd(Eye(4), im * pi)
end

@testitem "displacement of a linear operator is zero without applying it" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using AbstractOperators
    # `NaN * 0` is `NaN`, so applying any of these to zeros would not give zero.
    P = MatrixOp(fill(NaN, 3, 4))
    @test displacement(P) === 0.0
    @test displacement(MatrixOp(fill(NaN + 0im, 3, 4))) === 0.0im
    @test displacement(P * MatrixOp(fill(NaN, 4, 2))) === 0.0
    @test displacement(VCAT(P, MatrixOp(fill(NaN, 2, 4)))) === 0.0
    @test displacement(2.0 * P') === 0.0
    d = randn(3)
    @test displacement(AffineAdd(P, d)) === d
    @test displacement(AffineAdd(P, d, false)) == -d
end

@testitem "displacement inside a combination is still computed" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using AbstractOperators
    # None of these is linear, and each has a nonzero displacement.
    A = MatrixOp(randn(3, 4))
    d = randn(3)
    T = AffineAdd(A, d)
    B = MatrixOp(randn(4, 2))
    @test !is_linear(T * B) && is_affine(T * B)
    @test displacement(T * B) ≈ d
    E = MatrixOp(ones(2, 3))
    @test displacement(E * T) ≈ E * d
    @test displacement(VCAT(T, MatrixOp(randn(2, 4)))) ≈ ArrayPartition(d, zeros(2))
    @test displacement(2.0 * T) ≈ 2 .* d
end

@testitem "AffineAdd: nonlinear and permute" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 5, 6
    A = randn(n, m)
    d = randn(n)
    opH = HCAT(Eye(n), MatrixOp(A))
    x = ArrayPartition(randn(n), randn(m))
    opHT = AffineAdd(opH, d)
    @test norm(opHT * x - (x.x[1] + A * x.x[2] .+ d)) < 1.0e-12
    p = [2; 1]
    @test norm(AbstractOperators.permute(opHT, p) * ArrayPartition(x.x[p]...) - (x.x[1] + A * x.x[2] .+ d)) < 1.0e-12

    n = 10
    d = randn(n)
    T = AffineAdd(Exp(n), d, false)
    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (exp.(x) - d)) < 1.0e-8
end

@testitem "AffineAdd equality operator" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 6
    A = MatrixOp(randn(n, m))
    d1 = randn(n)
    d2 = randn(n)
    @test AffineAdd(A, d1) == AffineAdd(A, d1)
    @test !(AffineAdd(A, d1) == AffineAdd(A, d2))
end

@testitem "AffineAdd property delegations (invertible, rank, diagonal)" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 5
    E = Eye(n)
    TE = AffineAdd(E, randn(n))
    @test is_invertible(TE) == is_invertible(E)
    D = DiagOp(randn(n))
    TD = AffineAdd(D, randn(n))
    @test is_AcA_diagonal(TD) == is_AcA_diagonal(D)
    @test is_AAc_diagonal(TD) == is_AAc_diagonal(D)
    m = 6
    A = MatrixOp(randn(n, m))
    TA = AffineAdd(A, randn(n))
    @test is_full_row_rank(TA) == is_full_row_rank(A)
    @test is_full_column_rank(TA) == is_full_column_rank(A)
    D2 = DiagOp(randn(n))
    TD2 = AffineAdd(D2, zeros(n))
    @test diag_AcA(TD2) == diag_AcA(D2)
    @test diag_AAc(TD2) == diag_AAc(D2)
end

@testitem "AffineAdd slicing property delegation" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex(Float64, (10,), 2:5)
    TG = AffineAdd(G, randn(4))
    @test is_sliced(TG) == is_sliced(G)
    @test AbstractOperators.get_slicing_expr(TG) == AbstractOperators.get_slicing_expr(G)
    @test AbstractOperators.get_slicing_mask(TG) == AbstractOperators.get_slicing_mask(G)
    @test AbstractOperators.remove_slicing(TG) isa Eye
end

@testitem "AffineAdd normal operator" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 6
    G = GetIndex(Float64, (n, m), (1:3, :))
    d = randn(3, m)
    TG = AffineAdd(G, d)
    @test AbstractOperators.has_optimized_normalop(TG) ==
        AbstractOperators.has_optimized_normalop(G)
    N = AbstractOperators.get_normal_op(TG)
    x = randn(n, m)
    @test N * x ≈ TG.A' * (TG.A * x + TG.d)
end

@testitem "AffineAdd with multi-domain (ArrayPartition) codomain displacement" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators, RecursiveArrayTools
    Random.seed!(0)

    # HCAT(A, A, ..., A) with the same operator repeated is `is_eye`-adjacent
    # via HCAT's normal-op fusion (see HCAT.jl): its normal operator has a
    # multi-domain codomain (an ArrayPartition), which AffineAdd's
    # constructor previously mis-validated via a flat `size`/`eltype` check
    # instead of the ArrayPartition-aware structural comparison.
    n, m = 8, 6
    A = MatrixOp(randn(n, m))
    H = HCAT(A, A)
    @test AbstractOperators.has_optimized_normalop(H)
    Hn = AbstractOperators.get_normal_op(H)

    d = ArrayPartition(randn(m), randn(m))
    T = AffineAdd(Hn, d)
    x = ArrayPartition(randn(m), randn(m))
    y = T * x
    expected_each = AbstractOperators.get_normal_op(A) * (x.x[1] + x.x[2])
    @test y.x[1] ≈ expected_each .+ d.x[1]
    @test y.x[2] ≈ expected_each .+ d.x[2]

    @test_throws DimensionMismatch AffineAdd(Hn, randn(m + 1, 2))
    @test_throws ErrorException AffineAdd(Hn, ArrayPartition(im * randn(m), im * randn(m)))
end

# A distinct-operator HCAT has no normal-op fusion: no false positive, and
# `is_full_column_rank` defaults safely to `false` (columns from independent
# blocks stacked into a shared codomain can always cancel).
@testitem "HCAT: no normal-op fusion or full-column-rank for distinct operators" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    H = HCAT(MatrixOp(randn(5, 10)), MatrixOp(randn(5, 10)))
    @test !AbstractOperators.has_optimized_normalop(H)
    @test !is_full_column_rank(H)
    @test !is_full_column_rank(HCAT(Eye(4), Eye(4)))
end

@testitem "AffineAdd is_thread_safe delegation" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test is_thread_safe(AffineAdd(DiagOp(randn(5)), randn(5))) == true
    C = Compose(FiniteDiff((6,)), DiagOp(randn(6)))
    @test is_thread_safe(AffineAdd(C, randn(5))) == false
end

@testitem "AffineAdd sign function" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 6
    opA = MatrixOp(randn(n, m))
    d = randn(n)
    @test sign(AffineAdd(opA, d, true)) == 1
    @test sign(AffineAdd(opA, d, false)) == -1
end

@testitem "AffineAdd (GPU)" tags = [:gpu, :calculus, :AffineAdd] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n, m = 5, 6
        A = gpu_randn(backend, n, m)
        d = gpu_randn(backend, n)
        T = AffineAdd(MatrixOp(A), d)
        x1 = gpu_randn(backend, m)
        y1 = T * x1
        y1_buf = similar(y1)
        mul!(y1_buf, T, x1)
        @test collect(y1) ≈ collect(y1_buf)

        r = gpu_randn(backend, n)
        r_adj = T' * r
        r_adj2 = similar(r_adj)
        mul!(r_adj2, T', r)
        @test collect(r_adj) ≈ collect(r_adj2)
    end
end

@testitem "AffineAdd: array type mismatch error" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using AbstractOperators
    n = 5
    op = Eye(Float64, (n,))
    # eltype(d) != codomain_type(op): ComplexF64 vs Float64 (line 39)
    @test_throws ErrorException AffineAdd(op, randn(ComplexF64, n))
    # Float32 vs Float64
    @test_throws ErrorException AffineAdd(op, Float32.(randn(n)))
end

@testitem "AffineAdd: element type mismatch error" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using AbstractOperators
    n = 4
    op = MatrixOp(randn(n, n))    # codomain_type = Float64
    d = randn(Float32, n)          # eltype = Float32 != Float64
    @test_throws ErrorException AffineAdd(op, d)
end

@testitem "AffineAdd: threading traits and copy_operator" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(1)

    n, m = 5, 6
    threaded_leaf = FiniteDiff(Float64, (1 << 16,); threaded = true)
    serial_leaf = FiniteDiff(Float64, (1 << 16,); threaded = false)
    d1 = randn((1 << 16) - 1)
    @test is_threaded(AffineAdd(threaded_leaf, d1)) == (Threads.nthreads() > 1)
    @test is_threaded(AffineAdd(serial_leaf, d1)) == false
    @test supports_threading(AffineAdd(serial_leaf, d1)) == true

    A = randn(n, m)
    opA = MatrixOp(A)
    d = randn(n)
    T = AffineAdd(opA, d)
    T2 = copy_operator(T; threaded = true)
    @test T2 isa AffineAdd
    x = randn(m)
    @test T * x ≈ T2 * x

    # storage_type request forces the (otherwise shared) displacement array to be copied.
    T3 = copy_operator(T; storage_type = Array{Float64})
    @test T3.d !== T.d
    @test T3.d == T.d

    # Scalar displacement: `_copy_displacement(::Number, ::Any)` shares rather than copies.
    T_scalar = AffineAdd(opA, pi)
    T_scalar2 = copy_operator(T_scalar; storage_type = Array{Float64})
    @test T_scalar2.d === T_scalar.d
end
