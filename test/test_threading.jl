# Threading contract.
#
# Two things are checked here, and they are deliberately different in kind:
#
#   * the *policy* -- `is_threaded` must report what will actually happen, and a
#     `threaded = false` operator must never thread, at any size, on any input;
#   * the *kernels* -- the threaded and serial paths must agree, and the threaded path must
#     not allocate in proportion to the input.
#
# Agreement is `rtol`, not `==`: `@batch reduction` and `@tturbo` both re-associate the sum,
# so the value differs in the last couple of digits. The written array, which involves no
# reduction, is compared exactly -- if that ever differs the kernel is wrong, not merely
# reassociated.

using LinearAlgebra
using Random
using Test

using ProximalOperators
using ProximalOperators: is_threaded, supports_threading, should_thread, execution_strategy
using ProximalOperators: threading_threshold, is_cpu_storage, cost_class
using ProximalOperators: Simd, Batch, Turbo, TTurbo
using ProximalOperators: @elementwise_loop

Random.seed!(0)

# An operator, built from `(n, threaded)` so that each case can be constructed both ways and
# at whatever size its own cost class calls for. Operators whose parameter is an array need
# `n` to build at all, which is why the size comes first.
threading_cases() = [
    "NormL1"          => ((n, th) -> NormL1(0.3; threaded=th),                     :real),
    "NormL1 weighted" => ((n, th) -> NormL1(rand(n) .+ 0.1; threaded=th),          :real),
    "SqrNormL2"       => ((n, th) -> SqrNormL2(0.7; threaded=th),                  :real),
    "SqrNormL2 w"     => ((n, th) -> SqrNormL2(rand(n) .+ 0.1; threaded=th),       :real),
    "ElasticNet"      => ((n, th) -> ElasticNet(0.3, 0.7; threaded=th),            :real),
    "NormL0"          => ((n, th) -> NormL0(0.3; threaded=th),                     :real),
    "SumPositive"     => ((n, th) -> SumPositive(; threaded=th),                   :real),
    "IndBox"          => ((n, th) -> IndBox(-0.5, 0.5; threaded=th),               :real),
    "IndNonnegative"  => ((n, th) -> IndNonnegative(; threaded=th),                :real),
    "IndNonpositive"  => ((n, th) -> IndNonpositive(; threaded=th),                :real),
    "IndZero"         => ((n, th) -> IndZero(; threaded=th),                       :real),
    "IndBinary"       => ((n, th) -> IndBinary(0.0, 1.0; threaded=th),             :real),
    "NormL2"          => ((n, th) -> NormL2(0.3; threaded=th),                     :real),
    "IndBallL2"       => ((n, th) -> IndBallL2(1.0; threaded=th),                  :real),
    "IndSphereL2"     => ((n, th) -> IndSphereL2(1.0; threaded=th),                :real),
    "HuberLoss"       => ((n, th) -> HuberLoss(1.0, 1.0; threaded=th),             :real),
    "LogBarrier"      => ((n, th) -> LogBarrier(1.0, 0.0, 1.0; threaded=th),       :positive),
    "LogisticLoss"    => ((n, th) -> LogisticLoss(randn(n); threaded=th),          :real),
    "SqrHingeLoss"    => ((n, th) -> SqrHingeLoss(sign.(randn(n)); threaded=th),   :real),
    "NormL1 complex"  => ((n, th) -> NormL1(0.3; threaded=th),                     :complex),
]

# The size at which a given operator is expected to thread is its own, not a global
# constant: `MemoryBound` kernels only pay off at 2^18 while transcendental ones do at 2^10,
# and hard-coding one size here would test the wrong thing for most of the table.
big_size(f) = max(threading_threshold(f), 1 << 12)

make_input(kind, n) =
    kind === :real ? randn(n) :
    kind === :complex ? randn(ComplexF64, n) :
    rand(n) .+ 0.5

# build the pair and a matching input in one step
function build_case(build, kind, th)
    probe = build(1, th)                # cheap instance, only to read its threshold off
    n = big_size(probe)
    Random.seed!(0)
    return build(n, th), make_input(kind, n)
end

@testset "threading: kernels agree" begin
    for (name, (build, kind)) in threading_cases()
        @testset "$name" begin
            ft, x = build_case(build, kind, true)
            fs, _ = build_case(build, kind, false)
            yt = similar(x); ys = similar(x)
            vt = prox!(yt, ft, x, 0.7)
            vs = prox!(ys, fs, x, 0.7)
            # the written array must match exactly: no reduction is involved in it
            @test yt == ys
            @test isapprox(vt, vs; rtol = sqrt(eps(real(eltype(x)))))
            # calling the operator agrees too
            @test isapprox(ft(x), fs(x); rtol = sqrt(eps(real(eltype(x)))), nans = true)
        end
    end
end

@testset "threading: policy contract" begin
    x = randn(1 << 18)
    small = randn(4)
    for (name, (build, kind)) in threading_cases()
        @testset "$name" begin
            ft, xx = build_case(build, kind, true)
            fs, _ = build_case(build, kind, false)

            # the type knows it has a threaded path either way
            @test supports_threading(ft)
            @test supports_threading(fs)

            # `threaded = false` is a veto: never true, whatever the input
            @test is_threaded(fs, xx) == false
            @test is_threaded(fs, small) == false

            # `threaded = true` is a permission, not a command: below the operator's own
            # threshold it still declines
            @test is_threaded(ft, small) == false

            # and above it, it threads exactly when Julia has threads to spare
            @test is_threaded(ft, xx) == (Threads.nthreads() > 1)
        end
    end

    # an operator that was never converted has no threaded path, which is a different
    # statement from having declined one
    @test supports_threading(IndFree()) == false
    @test is_threaded(IndFree(), x) == false
end

@testset "threading: strategy selection" begin
    xr = randn(1 << 18)
    xc = randn(ComplexF64, 1 << 18)
    small = randn(4)

    # transcendental kernels go to LoopVectorization on real input, and only there
    lb = LogBarrier(1.0, 0.0, 1.0)
    @test execution_strategy(lb, small) isa Turbo
    @test execution_strategy(lb, xr) isa (Threads.nthreads() > 1 ? TTurbo : Turbo)
    @test execution_strategy(LogBarrier(1.0, 0.0, 1.0; threaded=false), xr) isa Turbo

    # arithmetic kernels go to Polyester, and complex input never reaches LoopVectorization
    n1 = NormL1(0.3)
    @test execution_strategy(n1, small) isa Simd
    @test execution_strategy(n1, xr) isa (Threads.nthreads() > 1 ? Batch : Simd)
    @test execution_strategy(LogisticLoss(randn(1 << 18)), xc) isa
        (Threads.nthreads() > 1 ? Batch : Simd)
end

# A batched loop that runs on one thread is not a correctness bug, so nothing else in this
# file notices it -- but it is the whole point of the feature, and it has been silently lost
# once already: `NestedThreading.@budgeted` reads the wrapped macro's *name* to decide
# whether to exclude the `:polyester` pool from the budget scope, so a head it cannot read
# leaves the scope restricting the pool the loop itself runs on. The symptom was every
# threaded operator running slower than serial. Check the property directly.
@testset "threading: the batched branch uses more than one thread" begin
    n = 1 << 20
    x = randn(n)
    ids = zeros(Int, n)
    strategy = Batch()
    @elementwise_loop strategy for i in eachindex(x, ids)
        ids[i] = Threads.threadid()
    end
    if Threads.nthreads() > 1
        @test length(unique(ids)) > 1
    else
        @test length(unique(ids)) == 1
    end
end

@testset "threading: storage is respected" begin
    x = randn(1 << 18)
    @test is_cpu_storage(x)
    @test is_cpu_storage(view(x, 1:10))
    @test is_cpu_storage(reshape(x, :, 1))
    # the conservative default: an array type the package knows nothing about is not
    # assumed to be CPU memory, which is what keeps device arrays off the threaded path
    @test is_cpu_storage(AbstractArray{Float64, 1}) == false
end

@testset "threading: allocations" begin
    # Polyester's reduction scratch is a fixed cost per call, so the invariant that matters
    # is that it does not grow with the input -- not that it is zero.
    f = NormL1(0.3)
    fs = NormL1(0.3; threaded=false)
    counts = map((1 << 16, 1 << 18, 1 << 20)) do n
        x = randn(n); y = similar(x)
        prox!(y, f, x, 0.7)
        @allocated prox!(y, f, x, 0.7)
    end
    @test allequal(counts)

    # the serial path allocates nothing at all, as it did before threading existed
    let x = randn(1 << 16), y = similar(x)
        prox!(y, fs, x, 0.7)
        @test (@allocated prox!(y, fs, x, 0.7)) == 0
    end
end

@testset "threading: block-parallel rules" begin
    # enough slices, and enough work, to cross both conditions at once
    k = 8
    n = 1 << 14
    idxs = Tuple((((i - 1) * n + 1):(i * n),) for i in 1:k)
    x = randn(k * n)

    gt = SlicedSeparableSum(Tuple(NormL1(0.3) for _ in 1:k), idxs)
    gs = SlicedSeparableSum(Tuple(NormL1(0.3) for _ in 1:k), idxs; threaded = false)
    yt = similar(x); ys = similar(x)
    vt = prox!(yt, gt, x, 0.7)
    vs = prox!(ys, gs, x, 0.7)
    @test yt == ys
    @test isapprox(vt, vs; rtol = sqrt(eps(Float64)))
    @test supports_threading(gt)
    @test is_threaded(gs, x) == false

    # slices whose functions were preallocated for them: the threaded path must hand each
    # one the `SubArray` it was prepared for. Under `Polyester.@batch` they arrived as
    # `PtrArray`s, which failed the signature check (and sent `NuclearNorm` to the host).
    let m = 4096, t = 8, k = 4
        idxm = Tuple((:, :, i) for i in 1:k)
        X = randn(m, t, k)
        for f in (NuclearNorm(0.5), IndBallRank(2))
            pt = preallocate(SlicedSeparableSum(f, idxm), X)
            ps = preallocate(SlicedSeparableSum(f, idxm; threaded = false), X)
            Yt = similar(X); Ys = similar(X)
            @test prox!(Yt, pt, X, 0.7) ≈ prox!(Ys, ps, X, 0.7)
            @test Yt ≈ Ys
            @test pt(Yt) ≈ ps(Ys)
        end
    end

    # a SeparableSum below the block-count floor never spawns, however large the blocks
    two = SeparableSum(NormL1(0.3), NormL1(0.4))
    xs2 = (randn(1 << 18), randn(1 << 18))
    ys2 = similar.(xs2)
    @test ProximalOperators.should_thread_blocks(two, length(xs2), sum(length, xs2)) == false
    v2 = prox!(ys2, two, xs2, 0.7)
    @test isfinite(v2)

    # above it, threaded and serial agree
    kk = 6
    xsk = Tuple(randn(1 << 14) for _ in 1:kk)
    gt2 = SeparableSum(Tuple(NormL1(0.3) for _ in 1:kk)...)
    gs2 = SeparableSum(Tuple(NormL1(0.3) for _ in 1:kk); threaded = false)
    yk1 = similar.(xsk); yk2 = similar.(xsk)
    v1 = prox!(yk1, gt2, xsk, 0.7)
    v2b = prox!(yk2, gs2, xsk, 0.7)
    @test all(map(==, yk1, yk2))
    @test isapprox(v1, v2b; rtol = sqrt(eps(Float64)))
end

@testset "threading: inference is preserved" begin
    # the threaded branch must not widen the return type: `prox_test` in the main suite
    # calls `@inferred`, and a `Union` strategy that leaked `Any` would break every one
    for (name, (build, kind)) in threading_cases()
        f, x = build_case(build, kind, true)
        y = similar(x)
        @test Base.return_types(prox!, (typeof(y), typeof(f), typeof(x), Float64))[1] ===
            real(eltype(x))
    end
end
