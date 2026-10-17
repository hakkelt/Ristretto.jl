using LinearAlgebra
using Random
using Test

using ProximalCore: is_convex
using ProximalOperators
using ProximalOperators: is_preallocated

Random.seed!(0)

# Each entry: name => (operator, prototype input, step size).
# `mismatch` says whether the operator is expected to reject a wrongly-sized
# input (operators that recurse without buffers of their own do not record a
# signature themselves, their inner operators do).
const PREALLOC_CASES = let
    Q = Matrix(qr(randn(30, 15)).Q)'   # 15x30, Q*Q' = I
    [
        "IndBallL1 (real)" => (IndBallL1(1.0), randn(50), 0.8),
        "IndBallL1 (complex)" => (IndBallL1(1.0), randn(ComplexF64, 40), 0.8),
        "IndSimplex" => (IndSimplex(1.0), randn(50), 0.8),
        "LogisticLoss" => (LogisticLoss(randn(30), 1.5), randn(30), 0.3),
        "NuclearNorm" => (NuclearNorm(0.3), randn(20, 12), 0.7),
        "IndStiefel" => (IndStiefel(), randn(20, 5), 1.0),
        "IndBallRank" => (IndBallRank(3), randn(15, 10), 1.0),
        "Conjugate" => (Conjugate(IndBallL1(1.0)), randn(40), 0.6),
        "Precompose" => (Precompose(NormL1(0.5), Q, 1.0, randn(15)), randn(30), 0.9),
        "PrecomposeDiagonal" => (PrecomposeDiagonal(NormL1(0.4), rand(25) .+ 0.5, randn(25)), randn(25), 0.7),
        "Translate" => (Translate(NormL1(0.4), randn(25)), randn(25), 0.7),
        "Tilt" => (Tilt(NormL1(0.4), randn(25)), randn(25), 0.7),
        "Regularize" => (Regularize(NormL1(0.4), 1.5, randn(25)), randn(25), 0.7),
        "Postcompose" => (Postcompose(NormL1(0.4), 2.0), randn(25), 0.7),
        "MoreauEnvelope" => (MoreauEnvelope(IndBallL1(1.0)), randn(25), 0.7),
        "DistL2" => (DistL2(IndBallL1(1.0)), randn(25), 0.7),
        "PointwiseMinimum" => (PointwiseMinimum(IndPoint(randn(20)), IndBallL1(1.0)), randn(20), 0.7),
        "ReshapeInput" => (ReshapeInput(NuclearNorm(0.3), (10, 10)), randn(100), 0.7),
        "SlicedSeparableSum" => (SlicedSeparableSum((NormL1(0.4), NormL1(0.6)), ((1:10,), (11:20,))), randn(20), 0.7),
    ]
end

# `IndBallRank` goes through TSVD.jl's randomly-initialised Lanczos iteration,
# so repeated calls on the same input differ in the last few digits whether or
# not buffers are involved; it is the only case compared approximately.
const APPROX_ONLY = ("IndBallRank",)

@testset "preallocate: equivalence" begin
    for (name, (f, x, gamma)) in PREALLOC_CASES
        @testset "$name" begin
            fp = preallocate(f, x)
            same = name in APPROX_ONLY ? (a, b) -> isapprox(a, b, rtol=1e-8) : ==

            y_plain = similar(x)
            y_pre = similar(x)
            v_plain = prox!(y_plain, f, x, gamma)
            v_pre = prox!(y_pre, fp, x, gamma)

            @test same(y_pre, y_plain)
            @test same(v_pre, v_plain)

            # calling twice must give the same answer: buffers are reset, not
            # accumulated into
            v_pre2 = prox!(y_pre, fp, x, gamma)
            @test same(y_pre, y_plain)
            @test same(v_pre2, v_plain)

            # traits are unaffected by the extra type parameter
            @inferred (arg -> Val(is_convex(arg)))(fp)
        end
    end
end

@testset "preallocate: gradient equivalence" begin
    Q = Matrix(qr(randn(30, 15)).Q)'
    cases = [
        "LogisticLoss" => (LogisticLoss(randn(30), 1.5), randn(30)),
        "Precompose" => (Precompose(SqrNormL2(1.0), Q, 1.0, randn(15)), randn(30)),
        "PrecomposeDiagonal" => (PrecomposeDiagonal(SqrNormL2(1.0), rand(25) .+ 0.5, randn(25)), randn(25)),
        "Translate" => (Translate(SqrNormL2(1.0), randn(25)), randn(25)),
        "Regularize" => (Regularize(SqrNormL2(1.0), 1.5, randn(25)), randn(25)),
        "Postcompose" => (Postcompose(SqrNormL2(1.0), 2.0), randn(25)),
        "MoreauEnvelope" => (MoreauEnvelope(IndBallL1(1.0)), randn(25)),
        "DistL2" => (DistL2(IndBallL1(1.0)), randn(25)),
        "Sum" => (Sum(SqrNormL2(1.0), SqrNormL2(2.0)), randn(25)),
    ]
    for (name, (f, x)) in cases
        @testset "$name" begin
            fp = preallocate(f, x)
            g_plain = similar(x)
            g_pre = similar(x)
            v_plain = gradient!(g_plain, f, x)
            v_pre = gradient!(g_pre, fp, x)
            @test g_pre == g_plain
            @test v_pre == v_plain
        end
    end
end

@testset "preallocate: input mismatch" begin
    # operators that record an input signature of their own
    cases = [
        "IndBallL1" => (IndBallL1(1.0), randn(50)),
        "IndSimplex" => (IndSimplex(1.0), randn(50)),
        "LogisticLoss" => (LogisticLoss(randn(30), 1.5), randn(30)),
        "NuclearNorm" => (NuclearNorm(0.3), randn(20, 12)),
        "IndStiefel" => (IndStiefel(), randn(20, 5)),
        "Conjugate" => (Conjugate(IndBallL1(1.0)), randn(40)),
        "Precompose" => (Precompose(NormL1(0.5), Matrix(qr(randn(30, 15)).Q)', 1.0, randn(15)), randn(30)),
        "Translate" => (Translate(NormL1(0.4), randn(25)), randn(25)),
        "Tilt" => (Tilt(NormL1(0.4), randn(25)), randn(25)),
        "Regularize" => (Regularize(NormL1(0.4), 1.5, randn(25)), randn(25)),
        "MoreauEnvelope" => (MoreauEnvelope(IndBallL1(1.0)), randn(25)),
    ]
    for (name, (f, x)) in cases
        @testset "$name" begin
            fp = preallocate(f, x)
            @test is_preallocated(fp)
            @test !is_preallocated(f)

            # wrong size
            bigger = randn(size(x) .+ 1)
            @test_throws DimensionMismatch prox!(similar(bigger), fp, bigger, 0.7)

            # wrong element type
            single = randn(Float32, size(x))
            @test_throws DimensionMismatch prox!(similar(single), fp, single, 0.7f0)

            # wrong array wrapper
            padded = randn(size(x) .+ 2)
            sliced = view(padded, map(n -> 1:n, size(x))...)
            @test_throws DimensionMismatch prox!(similar(x), fp, sliced, 0.7)
        end
    end

    # the message names the operator and both shapes
    f = preallocate(IndBallL1(1.0), randn(50))
    err = try
        prox!(randn(60), f, randn(60), 0.7)
    catch e
        e
    end
    @test err isa DimensionMismatch
    @test occursin("IndBallL1", err.msg)
    @test occursin("(50,)", err.msg)
    @test occursin("60", err.msg)
end

@testset "preallocate: allocations" begin
    Q = Matrix(qr(randn(30, 15)).Q)'
    # operators whose preallocated prox! must not allocate at all
    zero_alloc = [
        "IndBallL1 (real)" => (IndBallL1(1.0), randn(50), 0.8),
        "IndBallL1 (complex)" => (IndBallL1(1.0), randn(ComplexF64, 40), 0.8),
        "IndSimplex" => (IndSimplex(1.0), randn(50), 0.8),
        "LogisticLoss" => (LogisticLoss(randn(30), 1.5), randn(30), 0.3),
        "Precompose" => (Precompose(NormL1(0.5), Q, 1.0, randn(15)), randn(30), 0.9),
        "Translate" => (Translate(NormL1(0.4), randn(25)), randn(25), 0.7),
        "Tilt" => (Tilt(NormL1(0.4), randn(25)), randn(25), 0.7),
        "Conjugate" => (Conjugate(IndBallL1(1.0)), randn(40), 0.6),
        "NormL1 (weighted)" => (NormL1(rand(25)), randn(25), 0.7),
    ]
    for (name, (f, x, gamma)) in zero_alloc
        @testset "$name" begin
            fp = preallocate(f, x)
            y = similar(x)
            prox!(y, fp, x, gamma)   # warm up
            @test (@allocated prox!(y, fp, x, gamma)) == 0
        end
    end

    # factorization-based operators: LAPACK/TSVD still allocate their factors on
    # every call, so the invariant is "strictly fewer allocations than before",
    # not zero
    fewer_alloc = [
        "NuclearNorm" => (NuclearNorm(0.3), randn(20, 12), 0.7),
        "IndStiefel" => (IndStiefel(), randn(20, 5), 1.0),
        "IndBallRank" => (IndBallRank(3), randn(15, 10), 1.0),
    ]
    for (name, (f, x, gamma)) in fewer_alloc
        @testset "$name" begin
            fp = preallocate(f, x)
            y = similar(x)
            prox!(y, f, x, gamma)
            prox!(y, fp, x, gamma)
            @test (@allocated prox!(y, fp, x, gamma)) < (@allocated prox!(y, f, x, gamma))
        end
    end
end

@testset "host round-trip of a preallocated operator" begin
    # The fallback copies its input to an `Array`, which never matches the signature of an
    # operator preallocated for anything else; it must prepare its own operator for the copy.
    X = randn(ComplexF64, 30, 8, 2)
    xv = view(X, :, :, 1)
    for f in (NuclearNorm(0.4), IndBallRank(2))
        fp = preallocate(f, xv)
        Y = similar(X); Yr = similar(xv)
        v = ProximalOperators.host_prox!(view(Y, :, :, 1), fp, xv, 0.6)
        vr = prox!(Yr, f, copy(xv), 0.6)
        @test v ≈ vr
        @test view(Y, :, :, 1) ≈ Yr
        @test ProximalOperators.host_call(fp, xv) ≈ f(copy(xv))
    end
end
