@testitem "ShearletOp" tags = [:fftw, :ShearletOp] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, LinearAlgebra, Random
    using AbstractOperators: is_linear, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_full_row_rank, is_full_column_rank

    # Odd and even sizes, real and complex: even sizes have a Nyquist row, where the windows must
    # still be even for a real input to give real coefficients.
    for (T, sz, dirs) in ((ComplexF64, (32, 24), (8, 8, 16)), (Float64, (33, 40), (4, 6)))
        op = ShearletOp(T, sz; directions = dirs)
        x = randn(T, sz)
        tol = 1.0e-12
        @test size(op) == ((sz..., 1 + sum(dirs)), sz)
        y = test_op(op, x, randn(T, size(op, 1)), verb)
        @test eltype(y) == T
        @test norm(y) ≈ norm(x) rtol = tol                 # Parseval
        @test norm(op' * y - x) <= tol * norm(x)            # W'W = I
    end

    x32 = randn(Float32, 16, 16)
    op32 = ShearletOp(Float32, (16, 16))
    @test eltype(op32 * x32) == Float32
    @test op32' * (op32 * x32) ≈ x32 rtol = 1.0e-5

    op = ShearletOp(ComplexF64, (32, 32))
    @test is_linear(op)
    @test is_AcA_diagonal(op) && diag_AcA(op) == 1.0
    @test !is_AAc_diagonal(op)          # redundant: W*W' is a projection, not the identity
    @test !is_orthogonal(op)
    @test is_full_column_rank(op) && !is_full_row_rank(op)
    @test ShearletOp(zeros(ComplexF64, 32, 32)) * ones(ComplexF64, 32, 32) ≈ op * ones(ComplexF64, 32, 32)

    # Directional selectivity: a plane wave along the first axis at half the Nyquist frequency,
    # where the second scale's radial window is 1, lands in the first window (centred on
    # ξ₂ = 0) of that scale, and nowhere else.
    n = 64
    x = [cospi(2 * 16 * i / n) for i in 0:(n - 1), _ in 1:n]
    y = ShearletOp(Float64, (n, n)) * x
    e = vec(sum(abs2, y; dims = (1, 2))) ./ sum(abs2, x)
    @test e[1 + 8 + 1] ≈ 1 atol = 1.0e-12

    # copy_operator: own scratch and plans, same result.
    op2 = AbstractOperators.copy_operator(op)
    @test op2.buf !== op.buf
    x = randn(ComplexF64, 32, 32)
    @test op2 * x ≈ op * x
end

@testitem "SurfaceletOp" tags = [:fftw, :SurfaceletOp] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, LinearAlgebra, Random

    for (T, sz, levels) in ((ComplexF64, (16, 12, 8), (1, 1, 1)), (Float64, (16, 16, 9), (0, 2)))
        op = SurfaceletOp(T, sz; levels)
        x = randn(T, sz)
        @test size(op) == ((sz..., 1 + sum(l -> 3 * 4^l, levels)), sz)
        y = test_op(op, x, randn(T, size(op, 1)), verb)
        @test eltype(y) == T
        @test norm(y) ≈ norm(x) rtol = 1.0e-12
        @test norm(op' * y - x) <= 1.0e-12 * norm(x)
    end

    # A plane wave along the third axis at half the Nyquist frequency lands in the first scale
    # (whose radial window is 1 there), in the pyramid of the third axis.
    n = 16
    x = [cospi(2 * 4 * k / n) for _ in 1:n, _ in 1:n, k in 0:(n - 1)]
    op = SurfaceletOp(Float64, (n, n, n); levels = (0, 0))
    e = vec(sum(abs2, op * x; dims = (1, 2, 3))) ./ sum(abs2, x)
    @test e[1 + 3] ≈ 1 atol = 1.0e-12
end
