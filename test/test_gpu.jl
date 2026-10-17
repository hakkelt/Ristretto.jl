# GPU coverage.
#
# The claim this file exists to check is a universal one -- *every* operator in the package
# works on device arrays -- so the test is an enumeration, not a sample. An operator added
# later without a device story fails here by construction rather than being discovered by a
# user.
#
# The device is `JLArrays`, a reference GPU array that runs on the CPU. That is deliberate:
# it runs on CI with no GPU runner, and with `allowscalar(false)` it is *stricter* than a
# real device about the thing most likely to be wrong, namely a surviving scalar index. What
# it cannot check is the two tiers that depend on a backend's own LAPACK (`:device_lapack`)
# taking their native path; on JLArrays those correctly fall back to the host, and the
# native branch needs a real device.

using LinearAlgebra
using Random
using Test

using GPUArrays
using JLArrays

using ProximalOperators
using ProximalOperators: device_tier, supports_device, is_cpu_storage

JLArrays.allowscalar(false)
Random.seed!(0)

to_device(x::AbstractArray) = JLArray(collect(x))
to_device(x::Tuple) = map(to_device, x)
to_host(x::AbstractArray) = Array(x)
to_host(x::Tuple) = map(to_host, x)

same(a::Tuple, b::Tuple) = all(map(same, a, b))
same(a, b) = isapprox(a, b; rtol = 1e-6, atol = 1e-9)

# Every exported operator, with a prototype input. Operators carrying their own data are
# built from host arrays on purpose: that is the documented contract for the round-trip
# tier, and for the rest the data is small enough that where it lives does not matter.
gpu_cases() = begin
    Q = Matrix(qr(randn(30, 15)).Q)'
    Any[
        ("NormL1",             NormL1(0.3),                      randn(40), 0.7),
        ("NormL1 complex",     NormL1(0.3),                      randn(ComplexF64, 40), 0.7),
        ("NormL1 weighted",    NormL1(rand(40) .+ 0.1),          randn(40), 0.7),
        ("NormL0",             NormL0(0.3),                      randn(40), 0.7),
        ("NormL2",             NormL2(0.3),                      randn(40), 0.7),
        ("NormL21",            NormL21(0.3, 1),                  randn(20, 8), 0.7),
        ("NormLinf",           NormLinf(0.3),                    randn(40), 0.7),
        ("SqrNormL2",          SqrNormL2(0.7),                   randn(40), 0.7),
        ("ElasticNet",         ElasticNet(0.3, 0.7),             randn(40), 0.7),
        ("CubeNormL2",         CubeNormL2(0.5),                  randn(40), 0.7),
        ("HuberLoss",          HuberLoss(1.0, 1.0),              randn(40), 0.7),
        ("LogBarrier",         LogBarrier(1.0, 0.0, 1.0),        rand(40) .+ 0.5, 0.7),
        ("LogisticLoss",       LogisticLoss(randn(40), 1.5),     randn(40), 0.3),
        ("SqrHingeLoss",       SqrHingeLoss(sign.(randn(40))),   randn(40), 0.7),
        ("HingeLoss",          HingeLoss(sign.(randn(40))),      randn(40), 0.7),
        ("SumPositive",        SumPositive(),                    randn(40), 0.7),
        ("Linear",             Linear(randn(40)),                randn(40), 0.7),
        ("Maximum",            Maximum(0.5),                     randn(40), 0.7),
        ("IndBox",             IndBox(-0.5, 0.5),                randn(40), 0.7),
        ("IndBallLinf",        IndBallLinf(1.0),                 randn(40), 0.7),
        ("IndNonnegative",     IndNonnegative(),                 randn(40), 0.7),
        ("IndNonpositive",     IndNonpositive(),                 randn(40), 0.7),
        ("IndRealBox",         IndRealBox(-0.5, 0.5),            randn(ComplexF64, 40), 0.7),
        ("IndRealNonnegative", IndRealNonnegative(),             randn(ComplexF64, 40), 0.7),
        ("IndZero",            IndZero(),                        randn(40), 0.7),
        ("IndFree",            IndFree(),                        randn(40), 0.7),
        ("IndBinary",          IndBinary(0.0, 1.0),              randn(40), 0.7),
        ("IndPoint",           IndPoint(randn(40)),              randn(40), 0.7),
        ("IndSimplex",         IndSimplex(1.0),                  randn(40), 0.7),
        ("IndBallL0",          IndBallL0(5),                     randn(40), 0.7),
        ("IndBallL1",          IndBallL1(1.0),                   randn(40), 0.7),
        ("IndBallL1 complex",  IndBallL1(1.0),                   randn(ComplexF64, 40), 0.7),
        ("IndBallL2",          IndBallL2(1.0),                   randn(40), 0.7),
        ("IndSphereL2",        IndSphereL2(1.0),                 randn(40), 0.7),
        ("IndHalfspace",       IndHalfspace(randn(40), 0.5),     randn(40), 0.7),
        ("IndHyperslab",       IndHyperslab(-1.0, randn(40), 1.0), randn(40), 0.7),
        ("IndSOC",             IndSOC(),                         randn(10), 0.7),
        ("IndRotatedSOC",      IndRotatedSOC(),                  randn(10), 0.7),
        ("IndExpPrimal",       IndExpPrimal(),                   randn(3), 0.7),
        ("IndPSD",             IndPSD(),                         (A = randn(8, 8); Matrix(Symmetric(A'A))), 0.7),
        ("IndStiefel",         IndStiefel(),                     randn(20, 5), 1.0),
        ("IndBallRank",        IndBallRank(3),                   randn(15, 10), 1.0),
        ("NuclearNorm",        NuclearNorm(0.3),                 randn(20, 12), 0.7),
        ("BatchedNuclearNorm", BatchedNuclearNorm(0.3),          randn(12, 5, 4), 0.7),
        ("IndAffine",          IndAffine(Matrix(Q), randn(15)),  randn(30), 0.7),
        ("IndGraph",           IndGraph(randn(10, 20)),          (randn(20), randn(10)), 0.7),
        ("LeastSquares",       LeastSquares(randn(20, 30), randn(20)), randn(30), 0.7),
        ("Quadratic",          Quadratic((A = randn(30, 30); A'A + I), randn(30)), randn(30), 0.7),
        ("TotalVariation1D",   TotalVariation1D(0.3),            randn(40), 0.7),
        ("SumLargest",         ProximalOperators.SumLargest(3, 1.0), randn(40), 0.7),
        ("Conjugate",          Conjugate(IndBallL1(1.0)),        randn(40), 0.6),
        ("Precompose",         Precompose(NormL1(0.5), Matrix(Q), 1.0, randn(15)), randn(30), 0.9),
        ("PrecomposeDiagonal", PrecomposeDiagonal(NormL1(0.4), rand(25) .+ 0.5, randn(25)), randn(25), 0.7),
        ("Translate",          Translate(NormL1(0.4), randn(25)), randn(25), 0.7),
        ("Tilt",               Tilt(NormL1(0.4), randn(25)),     randn(25), 0.7),
        ("Regularize",         Regularize(NormL1(0.4), 1.5, randn(25)), randn(25), 0.7),
        ("Postcompose",        Postcompose(NormL1(0.4), 2.0),    randn(25), 0.7),
        ("MoreauEnvelope",     MoreauEnvelope(IndBallL1(1.0)),   randn(25), 0.7),
        ("DistL2",             DistL2(IndBallL1(1.0)),           randn(25), 0.7),
        ("SqrDistL2",          SqrDistL2(IndBallL1(1.0)),        randn(25), 0.7),
        ("PointwiseMinimum",   PointwiseMinimum(IndPoint(randn(20)), IndBallL1(1.0)), randn(20), 0.7),
        ("ReshapeInput",       ReshapeInput(NuclearNorm(0.3), (10, 10)), randn(100), 0.7),
        ("SlicedSeparableSum", SlicedSeparableSum((NormL1(0.4), NormL1(0.6)), ((1:10,), (11:20,))), randn(20), 0.7),
        ("SeparableSum",       SeparableSum(NormL1(0.4), NormL1(0.6)), (randn(10), randn(10)), 0.7),
        ("SeparableHuberLoss", SeparableHuberLoss(0.5, 2.0),     randn(40), 0.7),
        ("SeparableHuberLoss complex", SeparableHuberLoss(0.5, 2.0), randn(ComplexF64, 40), 0.7),
    ]
end

@testset "gpu: every operator runs on device arrays" begin
    for (name, f, x, gamma) in gpu_cases()
        @testset "$name" begin
            xd = to_device(x)
            yd = x isa Tuple ? map(similar, xd) : similar(xd)
            yh = x isa Tuple ? map(similar, x) : similar(x)

            vh = prox!(yh, f, x, gamma)
            vd = prox!(yd, f, xd, gamma)

            @test same(to_host(yd), yh)
            @test same(vd, vh)
            # the result must still live on the device: a fallback that quietly returned a
            # host array would pass the comparison above and break the next call
            if !(x isa Tuple)
                @test yd isa JLArray
            end
        end
    end
end

# The sweep above calls `prox!` and nothing else, which is how a scalar-indexing
# `gradient!` survived in `SqrNormL2` unnoticed: nothing on the device path ever asked a
# smooth function for its gradient. Every function with a `gradient!` belongs here.
gpu_gradient_cases() = Any[
    ("SqrNormL2", SqrNormL2(0.7), randn(40)),
    ("SqrNormL2 weighted", SqrNormL2(rand(40) .+ 0.1), randn(40)),
    ("SqrNormL2 complex", SqrNormL2(0.7), randn(ComplexF64, 40)),
    ("HuberLoss", HuberLoss(1.0, 1.0), randn(40)),
    ("SeparableHuberLoss", SeparableHuberLoss(0.5, 2.0), randn(40)),
    ("SeparableHuberLoss complex", SeparableHuberLoss(0.5, 2.0), randn(ComplexF64, 40)),
    ("LogisticLoss", LogisticLoss(randn(40), 1.5), randn(40)),
    ("SqrHingeLoss", SqrHingeLoss(sign.(randn(40))), randn(40)),
    ("Linear", Linear(randn(40)), randn(40)),
    ("LeastSquares", LeastSquares(randn(20, 30), randn(20)), randn(30)),
    ("Quadratic", Quadratic((A = randn(30, 30); A'A + I), randn(30)), randn(30)),
    ("Postcompose", Postcompose(SqrNormL2(0.4), 2.0), randn(25)),
]

# The sweep above runs on vectors, where a flat index range and the array have the same
# shape; an indexed kernel must also hold for a multidimensional array.
@testset "gpu: indexed kernels on a multidimensional array" begin
    x = randn(6, 5, 3)
    for (name, f) in (
            ("IndBox", IndBox(-0.5, 0.5)),
            ("NormL1 weighted", NormL1(rand(6, 5, 3) .+ 0.1)),
            ("SqrNormL2 weighted", SqrNormL2(rand(6, 5, 3) .+ 0.1)),
        )
        @testset "$name" begin
            xd = to_device(x)
            yd, yh = similar(xd), similar(x)
            @test same(prox!(yd, f, xd, 0.7), prox!(yh, f, x, 0.7))
            @test same(to_host(yd), yh)
            @test same(f(xd), f(x))
        end
    end
end

@testset "gpu: gradients run on device arrays" begin
    for (name, f, x) in gpu_gradient_cases()
        @testset "$name" begin
            xd = to_device(x)
            gd, gh = similar(xd), similar(x)

            vh = ProximalOperators.gradient!(gh, f, x)
            vd = ProximalOperators.gradient!(gd, f, xd)

            @test same(to_host(gd), gh)
            @test same(vd, vh)
            @test gd isa JLArray
        end
    end
end

@testset "gpu: bisection agrees with the exact CPU algorithms" begin
    # The selection operators gain a second algorithm on device storage -- threshold
    # bisection instead of Condat or `partialsortperm` -- so this is the one place where the
    # two paths are genuinely different code computing the same projection, and the
    # agreement is worth asserting on its own rather than only through the sweep above.
    for n in (17, 64, 1000)
        x = randn(n)
        for (name, f) in (("IndSimplex", IndSimplex(1.0)), ("IndBallL1", IndBallL1(1.0)))
            @testset "$name n=$n" begin
                yc = similar(x)
                prox!(yc, f, x, 1.0)              # Condat, exact
                xd = JLArray(x)
                yg = similar(xd)
                prox!(yg, f, xd, 1.0)             # bisection
                @test isapprox(Array(yg), yc; rtol = sqrt(eps(Float64)), atol = 1e-12)
            end
        end
        @testset "IndBallL0 n=$n" begin
            k = max(1, n ÷ 4)
            f = IndBallL0(k)
            yc = similar(x); prox!(yc, f, x, 1.0)
            xd = JLArray(x); yg = similar(xd); prox!(yg, f, xd, 1.0)
            # ties aside, the same entries survive; in all cases the constraint holds
            @test count(!iszero, Array(yg)) <= k
            @test isapprox(Array(yg), yc; rtol = sqrt(eps(Float64)), atol = 1e-12)
        end
    end
end

@testset "gpu: preallocate follows the storage" begin
    x = randn(50)
    xd = JLArray(x)

    # buffers are built with `similar`, so a device prototype yields device buffers and no
    # part of the preallocation layer needed changing for this
    fp = preallocate(LogisticLoss(JLArray(randn(50)), 1.5), xd)
    @test ProximalOperators.is_preallocated(fp)
    @test fp.buf.expyz isa JLArray

    yd = similar(xd)
    v1 = prox!(yd, fp, xd, 0.3)
    yh = similar(x)
    v2 = prox!(yh, LogisticLoss(Array(fp.y), 1.5), x, 0.3)
    @test same(Array(yd), yh)
    @test same(v1, v2)

    # and the recorded signature rejects an input from the wrong memory
    @test_throws DimensionMismatch prox!(similar(x), fp, x, 0.3)
end

@testset "gpu: mixed storage is rejected" begin
    x = randn(1000)
    xd = JLArray(x)
    f = NormL1(0.3)
    @test_throws ArgumentError prox!(similar(x), f, xd, 0.7)
    @test_throws ArgumentError prox!(similar(xd), f, x, 0.7)
end

@testset "gpu: threading is off on device storage" begin
    xd = JLArray(randn(1 << 18))
    @test is_cpu_storage(xd) == false
    # `threaded = true` is a permission the policy declines here: the work is already
    # parallel, and a Julia thread pool on top would serialise it
    @test ProximalOperators.is_threaded(NormL1(0.3), xd) == false
    @test ProximalOperators.execution_strategy(NormL1(0.3), xd) isa ProximalOperators.Simd
end

@testset "gpu: the support matrix is complete and honest" begin
    # Every case above must declare a tier, and the tier must match what it does. This is
    # what makes the "every operator works on the GPU" claim checkable instead of aspirational.
    for (name, f, x, gamma) in gpu_cases()
        @testset "$name" begin
            tier = device_tier(f)
            @test tier in (:native, :device_lapack, :host)
            @test supports_device(f) == (tier !== :host)
        end
    end

    # the tiers propagate through calculus rules rather than being claimed per wrapper
    @test device_tier(Postcompose(NuclearNorm(1.0), 2.0)) === :device_lapack
    @test device_tier(SeparableSum(NormL1(1.0), IndSOC())) === :host
    @test device_tier(SeparableSum(NormL1(1.0), NormL2(1.0))) === :native
    @test device_tier(IndExpDual()) === :host
end

using RecursiveArrayTools: ArrayPartition

@testset "gpu: inner products and updates of a device ArrayPartition" begin
    xh = ArrayPartition(randn(Float32, 5), randn(ComplexF32, 3, 2))
    yh = ArrayPartition(randn(Float32, 5), randn(ComplexF32, 3, 2))
    x, y = ArrayPartition(map(JLArray, xh.x)...), ArrayPartition(map(JLArray, yh.x)...)
    @test dot(x, y) ≈ dot(xh, yh)
    @test norm(x) ≈ norm(xh)
    for p in (1, 2, 3, 0, Inf, -Inf)
        @test norm(x, p) ≈ norm(xh, p)
    end
    @test all(Array.(axpy!(2.0f0, x, copy(y)).x) .≈ collect(axpy!(2.0f0, xh, copy(yh)).x))
    @test all(Array.(axpby!(2.0f0, x, 0.5f0, copy(y)).x) .≈ collect(axpby!(2.0f0, xh, 0.5f0, copy(yh)).x))
end
