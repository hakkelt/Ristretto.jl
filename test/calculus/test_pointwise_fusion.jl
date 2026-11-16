@testitem "Pointwise fusion: fused runs equal the operators applied one by one" tags = [:calculus, :Compose, :pointwise] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    const AO = AbstractOperators
    Random.seed!(0)

    # The unfused result: every operator writes to its own buffer.
    function sequential(C, x)
        y = AO.allocate_in_codomain(C)
        AO._pw_sequential!(y, C.A, C.buf, x)
        return y
    end
    function sequential_adjoint(C, r)
        x = AO.allocate_in_domain(C)
        AO._pw_sequential!(x, map(adjoint, reverse(C.A)), reverse(C.buf), r)
        return x
    end

    for T in (Float64, ComplexF32), threaded in (false, true)
        n = threaded ? (128, 64, 16) : (6, 5, 4)
        img = n[1:2]
        maps = [
            DiagOp(randn(T, n); threaded),
            DiagOp(randn(T, n); threaded)',
            AO.get_normal_op(GetIndex(T, n, (:, rand(n[2], n[3]) .> 0.5))),
            AO.get_normal_op(GetIndex(T, n, (rand(n...) .> 0.5,))),
            Scale(T(2.5), AO.get_normal_op(GetIndex(T, n, (:, :, rand(n[3]) .> 0.5))); threaded),
        ]
        # An expansion along the last axis and one along a middle axis.
        expands = [
            BroadCast(Eye(T, img), n; threaded),
            AO.NoOperatorBroadCast(T, Array{T}, (n[1], n[3]), (n[1], 1, n[3]), n; threaded),
        ]
        chains = Any[]
        for A in maps, B in maps
            push!(chains, B * A)
        end
        for E in expands, A in maps
            push!(chains, A * E)
            push!(chains, E' * A * E)
            push!(chains, E' * maps[2] * A * E)
        end
        nfused = 0
        for C in chains
            C isa Compose || continue
            nfused += AO._pw_run_length(C.A) > 1
            x = randn(T, size(C, 2))
            r = randn(T, size(C, 1))
            @test C * x == sequential(C, x)
            @test C' * r == sequential_adjoint(C, r)
        end
        @test nfused > length(chains) ÷ 2
    end
end

@testitem "Pointwise fusion: grouping" tags = [:calculus, :Compose, :pointwise] setup = [TestUtils] begin
    using AbstractOperators
    const AO = AbstractOperators
    n = (6, 5, 4)
    E = BroadCast(Eye(Float64, n[1:2]), n)
    D = DiagOp(randn(n))
    G = AO.get_normal_op(GetIndex(Float64, n, (:, rand(5, 4) .> 0.5)))
    F = FiniteDiff(n, 1)
    @test AO._pw_run_length((E, D, G, F)) == 3
    @test AO._pw_run_length((D, G, E', F)) == 3
    @test AO._pw_run_length((D, E', F)) == 2
    @test AO._pw_run_length((F, D)) == 1
    @test AO._pw_run_length((E', D)) == 1
    @test AO._pw_run_length((D, E)) == 1
    # A mask that is not over trailing axes is not fused.
    Gmid = AO.get_normal_op(GetIndex(Float64, n, (:, 2:3, :)))
    @test AO._pw_kind(Gmid) isa AO.PwNoneKind
end

@testitem "Pointwise fusion: layouts of a broadcast" tags = [:calculus, :pointwise] setup = [TestUtils] begin
    using AbstractOperators
    const AO = AbstractOperators
    T = Float64
    @test AO._pw_layout(AO.NoOperatorBroadCast(T, Array{T}, (4, 5), (4, 5, 1), (4, 5, 3))) == (20, 3)
    @test AO._pw_layout(AO.NoOperatorBroadCast(T, Array{T}, (4, 5), (4, 1, 5), (4, 3, 5))) == (4, 3)
    @test AO._pw_layout(AO.NoOperatorBroadCast(T, Array{T}, (5,), (1, 1, 5), (2, 3, 5))) == (1, 6)
    B = AO.NoOperatorBroadCast(T, Array{T}, (4,), (1, 4, 1), (2, 4, 3))
    @test AO._pw_layout(B) === nothing
    # Broadcast axes that are not adjacent fall back to the unfused operators.
    D = DiagOp(randn(2, 4, 3))
    C = D * B
    x = randn(4)
    @test C * x == D * (B * x)
    r = randn(2, 4, 3)
    @test C' * r == B' * (D' * r)
end

@testitem "Pointwise fusion: arrays that are not Arrays fall back" tags = [:calculus, :pointwise] setup = [TestUtils] begin
    using LinearAlgebra, AbstractOperators
    const AO = AbstractOperators
    n = (6, 5, 4)
    E = BroadCast(Eye(Float64, n[1:2]), n)
    D = DiagOp(randn(n))
    C = D * E
    x = randn(n[1:2])
    y = zeros(n)
    yv = view(zeros(n .+ 1), 1:n[1], 1:n[2], 1:n[3])
    mul!(y, C, x)
    mul!(yv, C, view(copy(x), :, :))
    @test yv == y == D * (E * x)
end

@testitem "Pointwise fusion: threaded and serial kernels agree" tags = [:calculus, :pointwise] setup = [TestUtils] begin
    using Random, AbstractOperators
    const AO = AbstractOperators
    Random.seed!(1)
    inner, K, outer = 37, 3, 11
    d = randn(ComplexF64, inner * K * outer)
    m = rand(K * outer) .> 0.5
    steps = (AO.PwLeftMul{ComplexF64}(d), AO.PwTrailingMask{ComplexF64}(m, inner))
    x = randn(ComplexF64, inner * outer)
    y1 = zeros(ComplexF64, inner * K * outer)
    y2 = similar(y1)
    AO._pw_kernel!(y1, x, steps, AO.PwExpand(), (inner, K), false)
    AO._pw_kernel!(y2, x, steps, AO.PwExpand(), (inner, K), true)
    @test y1 == y2
    ref = reshape(d, inner, K, outer) .* reshape(x, inner, 1, outer) .* reshape(repeat(m; inner = inner), inner, K, outer)
    @test y1 == vec(ref)
    z1 = zeros(ComplexF64, inner * outer)
    z2 = similar(z1)
    AO._pw_kernel!(z1, y1, steps, AO.PwReduce(), (inner, K), false)
    AO._pw_kernel!(z2, y1, steps, AO.PwReduce(), (inner, K), true)
    @test z1 == z2
    @test z1 ≈ vec(sum(reshape(d .* y1 .* repeat(m; inner = inner), inner, K, outer); dims = 2))
end

@testitem "Pointwise fusion: no allocation" tags = [:calculus, :pointwise] setup = [TestUtils] begin
    using LinearAlgebra, AbstractOperators
    const AO = AbstractOperators
    n = (16, 8, 4)
    E = BroadCast(Eye(ComplexF64, n[1:2]), n; threaded = false)
    D = DiagOp(randn(ComplexF64, n); threaded = false)
    G = AO.get_normal_op(GetIndex(ComplexF64, n, (:, rand(8, 4) .> 0.5)))
    C = G * D * E
    @test AO._pw_run_length(C.A) == 3
    x = randn(ComplexF64, n[1:2])
    y = C * x
    allocated(y, C, x) = @allocated mul!(y, C, x)
    Ca = C'
    allocated(x, Ca, y)
    allocated(y, C, x)
    @test allocated(y, C, x) == 0
    @test allocated(x, Ca, y) == 0
end

@testitem "Pointwise fusion: device arrays" tags = [:gpu, :calculus, :pointwise] setup = [TestUtils, GpuEnvSetup] begin
    using GPUEnv, Random, LinearAlgebra, AbstractOperators
    const AO = AbstractOperators

    for backend in gpu_backends()
        Random.seed!(0)
        T = ComplexF32
        n = (6, 5, 4)
        dev(a) = to_gpu(backend, a)
        S = Base.typename(typeof(dev(zeros(T, 1)))).wrapper
        d1, d2 = randn(T, n), randn(T, n)
        D1, D2 = DiagOp(dev(d1)), DiagOp(dev(d2))
        hD1, hD2 = DiagOp(d1), DiagOp(d2)
        # An expansion along the last axis and one along a middle axis.
        expands = [
            (BroadCast(Eye(dev(zeros(T, n[1:2]))), n), BroadCast(Eye(T, n[1:2]), n)),
            (
                AO.NoOperatorBroadCast(T, S{T}, (n[1], n[3]), (n[1], 1, n[3]), n),
                AO.NoOperatorBroadCast(T, Array{T}, (n[1], n[3]), (n[1], 1, n[3]), n),
            ),
        ]
        @test AO._pw_kind(D1) isa AO.PwMapKind
        @test AO._pw_kind(D1') isa AO.PwMapKind
        @test AO._pw_kind(first(expands[1])) isa AO.PwExpandKind
        @test AO._pw_kind(first(expands[1])') isa AO.PwReduceKind
        pairs = Any[(D2 * D1, hD2 * hD1), (D2' * D1, hD2' * hD1), (Scale(T(2), D1) * D2, Scale(T(2), hD1) * hD2)]
        for (E, hE) in expands
            push!(pairs, (D1 * E, hD1 * hE))
            push!(pairs, (E' * D1, hE' * hD1))
            push!(pairs, (E' * D2' * D1 * E, hE' * hD2' * hD1 * hE))
        end
        for (C, H) in pairs
            # Two diagonals combine into one, which is no `Compose`.
            C isa Compose && @test AO._pw_run_length(C.A) > 1
            x = randn(T, size(C, 2))
            r = randn(T, size(C, 1))
            y = C * dev(x)
            @test y isa S
            @test collect(y) ≈ H * x rtol = 1.0e-5
            @test collect(C' * dev(r)) ≈ H' * r rtol = 1.0e-5
        end
    end
end

@testitem "Pointwise fusion: POINTWISE_FUSION switches it off at run time" tags = [:calculus, :Compose, :pointwise] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    const AO = AbstractOperators
    Random.seed!(1)
    for T in (Float64, ComplexF32), threaded in (false, true)
        n = threaded ? (128, 64, 16) : (6, 5, 4)
        E = BroadCast(Eye(T, n[1:2]), n; threaded)
        D1 = DiagOp(randn(T, n); threaded)
        D2 = DiagOp(randn(T, n); threaded)
        M = AO.get_normal_op(GetIndex(T, n, (rand(n...) .> 0.5,)))
        S = Scale(T(2.5), AO.get_normal_op(GetIndex(T, n, (:, :, rand(n[3]) .> 0.5))); threaded)
        candidates = (M * D1, S * D1', M * D1 * E, D2' * S * E, E' * D2 * D1, E' * S * M)
        # chains that stay a `Compose` of one fused run, so that every buffer lies inside it
        chains = filter(C -> C isa Compose && AO._pw_run_length(C.A) == length(C.A), candidates)
        @test length(chains) >= 3
        for C in chains
            x = randn(T, size(C, 2))
            r = randn(T, size(C, 1))
            # A fused run leaves the buffers between its operators untouched; the unfused path
            # writes every one of them.
            sentinel() = foreach(b -> fill!(b, T(NaN)), C.buf)
            sentinel()
            y_on = C * x
            @test all(b -> all(isnan, b), C.buf)
            try
                AO.POINTWISE_FUSION[] = false
                sentinel()
                y_off = C * x
                @test !any(b -> any(isnan, b), C.buf)
                @test y_off == y_on
                r_off = C' * r
                AO.POINTWISE_FUSION[] = true
                @test r_off == C' * r
            finally
                AO.POINTWISE_FUSION[] = true
            end
        end
    end
end
