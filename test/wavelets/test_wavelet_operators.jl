@testitem "WaveletOp" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, WaveletOperators

    ########## WaveletOp ############
    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = dwt(x1, wavelet(WT.db4))

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n = 8
    op = WaveletOp(ComplexF64, wavelet(WT.db4), (n,))
    x1 = randn(ComplexF64, n)
    y1 = test_op(op, x1, randn(ComplexF64, n), verb)
    y2 = dwt(x1, wavelet(WT.db4))

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "WaveletOp (GPU)" tags = [:gpu, :wavelet, :WaveletOp] setup = [TestUtils, GpuEnvSetup] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 8
        op = WaveletOp(Float64, wavelet(WT.db4), (n,); array_type = gpu_wrapper(backend, Float64, n))
        x1 = gpu_randn(backend, n)
        y1 = test_op(op, x1, gpu_randn(backend, n), false)
        @test domain_array_type(op) <: backend.array_type
        @test codomain_array_type(op) <: backend.array_type

        y2 = dwt(collect(x1), wavelet(WT.db4))
        @test norm(collect(y1) .- y2) <= 1.0e-12
    end
end

@testitem "WaveletOp: normal operator is the identity for orthogonal families" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators
    Random.seed!(1)

    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))

    @test AbstractOperators.has_optimized_normalop(op)
    Wop = AbstractOperators.get_normal_op(op)
    @test Wop isa Eye

    x = randn(n)
    @test norm(op' * (op * x) .- x) <= 1.0e-12
    @test norm(Wop * x .- x) <= 1.0e-12
    # Every family `WaveletOp` accepts is an orthogonal filter bank; the lifting-scheme ones,
    # biorthogonal CDF among them, are rejected by the constructor (tested below).
end

@testitem "WaveletOp: copy_operator" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators
    Random.seed!(6)

    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))

    # No constraint: reproduces the operator exactly.
    op2 = copy_operator(op)
    @test op2 isa WaveletOp
    @test typeof(op2) === typeof(op)
    x = randn(n)
    y1, y2 = zeros(n), zeros(n)
    mul!(y1, op, x)
    mul!(y2, op2, x)
    @test y1 ≈ y2

    # A 1-D transform has no threaded path, so `threaded` changes nothing.
    op3 = copy_operator(op; threaded = false)
    @test domain_array_type(op3) <: Array{Float64}
    @test !supports_threading(op3) && !is_threaded(op3)

    # `storage_type` rebuilds the storage-tracking type parameter.
    op4 = copy_operator(op; storage_type = Array{Float64})
    @test op4 isa WaveletOp
    @test domain_array_type(op4) <: Array{Float64}
    @test codomain_array_type(op4) <: Array{Float64}
end

@testitem "WaveletOp: the threaded 2-D and 3-D transforms equal Wavelets.jl's" tags = [:wavelet, :WaveletOp, :Threading] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators
    Random.seed!(7)

    # Sizes above the threading threshold, including a dimension that does not split evenly
    # across threads; levels up to the maximum.
    for (T, dims, w, L) in (
            (Float64, (512, 512), wavelet(WT.db2), 3),
            (ComplexF32, (512, 1024), wavelet(WT.haar), 9),
            (ComplexF32, (64, 64, 64), wavelet(WT.db2), 3),
            (Float64, (64, 96, 64), wavelet(WT.db4), 5),
        )
        op = WaveletOp(T, w, dims, L)
        @test supports_threading(op)
        @test is_threaded(op) == (Threads.nthreads() > 1)
        serial = copy_operator(op; threaded = false)
        @test !is_threaded(serial)
        x = randn(T, dims)
        @test op * x == dwt(x, w, L)
        @test op' * x == idwt(x, w, L)
        @test serial * x == op * x
        @test op' * (op * x) ≈ x
    end

    # Below the threshold the policy keeps it serial.
    @test !is_threaded(WaveletOp(Float64, wavelet(WT.db2), (32, 32), 2))

    # A serial batch hands each frame to a threaded transform as a view.
    for (dims, w, L) in (((512, 512), wavelet(WT.db2), 3), ((64, 64, 64), wavelet(WT.db2), 3))
        op = WaveletOp(ComplexF32, w, dims, L)
        x = randn(ComplexF32, dims..., 2)
        y = similar(x)
        for k in 1:2
            mul!(view(y, ntuple(_ -> :, length(dims))..., k), op, view(x, ntuple(_ -> :, length(dims))..., k))
            @test selectdim(y, length(dims) + 1, k) == dwt(selectdim(x, length(dims) + 1, k), w, L)
        end
        xa = similar(x)
        mul!(view(xa, ntuple(_ -> :, length(dims))..., 1), op', view(y, ntuple(_ -> :, length(dims))..., 1))
        @test selectdim(xa, length(dims) + 1, 1) ≈ selectdim(x, length(dims) + 1, 1)
        b = BatchOp(op, 2; threaded = false)
        @test b * x == cat((dwt(selectdim(x, length(dims) + 1, k), w, L) for k in 1:2)...; dims = length(dims) + 1)
    end
end

@testitem "WaveletOp: threaded scratch survives garbage collection" tags = [:wavelet, :WaveletOp, :Threading] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, WaveletOperators
    Random.seed!(11)

    # Small transforms, threaded regardless of the size policy, applied often enough that the
    # collector runs while the line kernels still write through the per-block scratch.
    for dims in ((64, 64), (32, 32, 32))
        serial = WaveletOp(ComplexF32, wavelet(WT.db2), dims, 3; threaded = false)
        P = typeof(serial).parameters
        threaded = typeof(serial).name.wrapper{P[1:4]..., true}(serial.wavelet, serial.dim_in, serial.levels)
        x = randn(ComplexF32, dims)
        y, ys, z, zs = similar(x), similar(x), similar(x), similar(x)
        mul!(ys, serial, x)
        mul!(zs, serial', ys)
        bad = 0
        for round in 1:300
            mul!(y, threaded, x)
            mul!(z, threaded', y)
            (y == ys && z == zs) || (bad += 1)
            round % 50 == 0 && GC.gc(false)
        end
        @test bad == 0
    end
end

@testitem "WaveletOp constructor errors" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, WaveletOperators
    wt = wavelet(WT.db4)

    # 1D: odd dimension
    @test_throws ArgumentError WaveletOp(Float64, wt, 5)
    # 1D: too many levels
    @test_throws ArgumentError WaveletOp(Float64, wt, 8, 100)

    # ND: odd dimension in tuple
    @test_throws ArgumentError WaveletOp(Float64, wt, (5, 8))
    # ND: too many levels
    @test_throws ArgumentError WaveletOp(Float64, wt, (8, 8), 100)
end

@testitem "WaveletOp rejects lifting-scheme wavelets" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, WaveletOperators, AbstractOperators, LinearAlgebra

    # Wavelets.jl defines the level-taking `dwt!`/`idwt!` only for an `OrthoFilter`, so a `GLS`
    # operator could be constructed but never applied -- it raised a `MethodError` from inside
    # Wavelets on first use. Reject it where the message can explain why.
    for class in (WT.db2, WT.haar, WT.cdf97)
        gls = wavelet(class, WT.Lifting)
        @test gls isa WT.GLS
        @test_throws ArgumentError WaveletOp(Float64, gls, (8, 8))
    end

    # The filter-bank form of the same orthogonal class is unaffected, and still gets the
    # identity fast paths.
    op = WaveletOp(Float64, wavelet(WT.db2), (8, 8))
    @test AbstractOperators.has_fast_opnorm(op)
    @test opnorm(op) == 1
    @test AbstractOperators.get_normal_op(op) isa Eye
    @test is_AcA_diagonal(op)
    x = randn(8, 8)
    @test op' * (op * x) ≈ x
end
