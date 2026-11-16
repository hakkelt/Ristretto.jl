@testitem "FFTShift/IFFTShift Operators" tags = [:fftw, :FFTShift] setup = [TestUtils] begin
    using AbstractOperators
    using LinearAlgebra, Random, FFTWOperators
    # 1D even length
    n = 4
    x = collect(1.0:n)
    A = FFTShift((n,), (1,))
    B = IFFTShift((n,), (1,))

    @test A * x == [3.0, 4.0, 1.0, 2.0]
    @test B * (A * x) == x
    @test A' * (A * x) == x  # orthogonal permutation
    @test B' * (B * x) == x

    # 1D odd length
    n = 5
    x = collect(1.0:n)
    A = FFTShift((n,), (1,))
    B = IFFTShift((n,), (1,))

    @test A * x == [4.0, 5.0, 1.0, 2.0, 3.0]
    @test B * (A * x) == x

    # 2D, both dims
    n, m = 2, 3
    X = collect(reshape(1.0:(n * m), n, m))
    A = FFTShift((n, m), (1, 2))
    B = IFFTShift((n, m), (1, 2))
    Y = A * X
    @test size(Y) == size(X)
    @test B * Y == X

    # Properties
    @test is_orthogonal(A)
    @test is_invertible(A)
    @test is_full_row_rank(A)
    @test is_full_column_rank(A)
    @test diag_AAc(A) == 1.0
    @test diag_AcA(A) == 1.0
end

@testitem "SignAlternation Operator" tags = [:fftw, :SignAlternation] setup = [TestUtils] begin
    using AbstractOperators
    using LinearAlgebra, Random, FFTWOperators
    # 1D
    n = 6
    x = ones(n)
    S = SignAlternation((n,), (1,))
    y = S * x
    @test y == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    @test S' * (S * x) == x
    @test AbstractOperators.is_symmetric(S)
    @test is_orthogonal(S)

    # 2D across both dims
    X = ones(2, 2)
    S2 = SignAlternation((2, 2), (1, 2))
    @test S2 * X == [1.0 -1.0; -1.0 1.0]

    # mul! variants
    y1 = similar(x)
    mul!(y1, S, x)
    @test y1 == y

    y2 = similar(X)
    mul!(y2, S2, X)
    @test y2 == S2 * X
end

@testitem "alternate_sign helpers" tags = [:fftw, :SignAlternation] setup = [TestUtils] begin
    using LinearAlgebra, Random, FFTWOperators
    # in-place
    v = collect(1.0:4.0)
    alternate_sign!(v, 1)
    @test v == [1.0, -2.0, 3.0, -4.0]

    # out-of-place with dest
    x = collect(reshape(1.0:4.0, 2, 2))
    y = similar(x)
    alternate_sign!(y, x, 1, 2)
    @test y == [1.0 -3.0; -2.0 4.0]

    # functional copy
    u = alternate_sign(collect(reshape(1.0:4.0, 2, 2)), 1, 2)
    @test u == [1.0 -3.0; -2.0 4.0]

    # no dirs → identity
    z = collect(1.0:3.0)
    @test_throws ArgumentError alternate_sign!(z)

    y = similar(z)
    @test_throws ArgumentError alternate_sign!(y, z)

    # too many dirs (N < M)
    m2d = ones(2, 2)
    @test_throws ArgumentError alternate_sign!(m2d, (1, 2, 3))
    y2d = similar(m2d)
    @test_throws ArgumentError alternate_sign!(y2d, m2d, (1, 2, 3))

    # unsorted dirs
    @test_throws ArgumentError alternate_sign!(m2d, (2, 1))
    @test_throws ArgumentError alternate_sign!(y2d, m2d, (2, 1))

    # out-of-range dirs
    @test_throws ArgumentError alternate_sign!(z, 0)
    @test_throws ArgumentError alternate_sign!(z, 2)  # 1D array, dir=2 > ndims

    # non-threaded paths (threaded=false)
    v2 = collect(1.0:4.0)
    alternate_sign!(v2, 1; threaded = false)
    @test v2 == [1.0, -2.0, 3.0, -4.0]

    x2 = collect(reshape(1.0:4.0, 2, 2))
    y2 = similar(x2)
    alternate_sign!(y2, x2, 1, 2; threaded = false)
    @test y2 == [1.0 -3.0; -2.0 4.0]
end

@testitem "alternate_sign!: hoisted kernel matches the naive per-element formula" tags = [:fftw, :SignAlternation] setup = [TestUtils] begin
    using LinearAlgebra, Random, FFTWOperators
    Random.seed!(3)

    # Reference: the naive per-element formula the kernel was rewritten from.
    function _ref_alternate_sign!(x::AbstractArray, dirs::NTuple)
        for I in CartesianIndices(x)
            flips = sum(iseven(I[d]) ? 1 : 0 for d in dirs)
            if isodd(flips)
                x[I] = -x[I]
            end
        end
        return x
    end

    # Every non-empty subset of `1:N`, sorted ascending (the shape `alternate_sign!` requires).
    function _dirs_subsets(N::Int)
        return [Tuple(d for d in 1:N if (mask >> (d - 1)) & 1 == 1) for mask in 1:(2^N - 1)]
    end

    # Mixed even/odd extents, including an axis of length exactly 2.
    sizes = [(4, 3, 2), (2, 5, 4), (6, 2, 3), (8,), (2, 2), (2, 3, 2)]
    for sz in sizes, threaded in (true, false)
        x = randn(ComplexF64, sz)
        N = length(sz)
        for dirs in _dirs_subsets(N)
            expected = _ref_alternate_sign!(copy(x), dirs)

            y = copy(x)
            alternate_sign!(y, dirs...; threaded)
            @test y ≈ expected

            yo = similar(x)
            alternate_sign!(yo, x, dirs...; threaded)
            @test yo ≈ expected

            # Self-inverse.
            @test alternate_sign!(copy(y), dirs...; threaded) ≈ x
        end
    end
end

@testitem "alternate_sign!: kernel state stays on the stack" tags = [:fftw, :SignAlternation] setup = [TestUtils] begin
    using Random, FFTWOperators
    using LinearAlgebra: mul!
    Random.seed!(5)

    # The per-column sign vector and the trailing-dimension mask used to be heap `Vector`s built
    # on every call, in a kernel that runs once per FFT-shift per operator application. `N` is a
    # static type parameter, so neither needs to allocate.
    #
    # Measured from inside a function, not at test-item top level: `@allocated` on a call whose
    # arguments are globals also counts the boxing of those globals, which is not the kernel's
    # doing (the same call measured at top level reports 128 B for a 3-element `dirs`).
    alloc_in_place(x, dirs) = @allocated alternate_sign!(x, dirs...; threaded = false)
    alloc_out_of_place(y, x, dirs) = @allocated alternate_sign!(y, x, dirs...; threaded = false)

    x = randn(ComplexF64, 32, 16, 4)
    y = similar(x)
    for dirs in ((1,), (2, 3), (1, 2, 3))
        alternate_sign!(copy(x), dirs...; threaded = false)      # warm up / compile
        alternate_sign!(y, x, dirs...; threaded = false)
        @test alloc_in_place(x, dirs) == 0
        @test alloc_out_of_place(y, x, dirs) == 0
    end

    # The same claim where it actually matters: the operator's own `mul!`, which runs once per
    # FFT-shift per operator application.
    mul_alloc(y, S, x) = @allocated mul!(y, S, x)
    S = SignAlternation(ComplexF64, size(x), (1, 2, 3); threaded = false)
    mul!(y, S, x)
    @test mul_alloc(y, S, x) == 0
end

@testitem "alternate_sign!: a single trailing column threads dimension 1" tags = [:fftw, :SignAlternation] setup = [TestUtils] begin
    using Random, FFTWOperators
    Random.seed!(5)

    # A single trailing column (a vector, or an `n x 1`) leaves the column loop with one item, so
    # the threaded path spreads dimension 1 instead. Both paths must agree with the naive formula.
    for sz in ((64,), (64, 1), (8192,))
        v = randn(ComplexF64, sz...)
        expected = [v[I] * (iseven(I[1]) ? -1 : 1) for I in CartesianIndices(v)]
        @test alternate_sign!(copy(v), 1; threaded = true) ≈ expected
        @test alternate_sign!(copy(v), 1; threaded = false) ≈ expected
    end
end

@testitem "alternate_sign!: the Cartesian fallback matches the linear kernel" tags = [:fftw, :SignAlternation] setup = [TestUtils] begin
    using Random, FFTWOperators
    using LinearAlgebra: mul!
    Random.seed!(7)

    # The kernel addresses a column by its linear index range, which only holds for an array
    # that indexes linearly and starts at 1. A strided view does neither, and takes the
    # Cartesian fallback instead; both must produce the same thing.
    for sz in ((8, 6), (8, 6, 3), (7, 4))
        N = length(sz)
        parent = randn(ComplexF64, (2 .* sz)...)
        idx = ntuple(k -> 1:2:(2 * sz[k]), N)
        dense = Array(@view parent[idx...])
        for dirs in ((1,), (2,), (1, 2)), threaded in (false, true)
            maximum(dirs) <= N || continue
            expected = alternate_sign!(copy(dense), dirs...; threaded = false)

            v = @view parent[idx...]
            v .= dense
            alternate_sign!(v, dirs...; threaded)
            @test Array(v) == expected

            out = @view parent[idx...]
            alternate_sign!(out, dense, dirs...; threaded)
            @test Array(out) == expected
        end
    end

    # An aliased `mul!` is routed to the in-place kernel, which only touches the elements that
    # flip; it must still be the same operator.
    x = randn(ComplexF64, 16, 8)
    S = SignAlternation(ComplexF64, size(x), (1, 2); threaded = false)
    expected = S * x
    y = copy(x)
    mul!(y, S, y)
    @test y == expected
end

@testitem "fftshift/ifftshift wrappers" tags = [:fftw, :FFTShift] setup = [TestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators, AbstractOperators
    # Even length
    n = 4
    A = DFT(n)
    x = randn(n)

    # Codomain shift: should equal shifting output of A*x
    T = fftshift_op(A; codomain_shifts = (1,))
    y1 = T * x
    y2 = FFTW.fftshift(A * x, (1,))
    @test y1 ≈ y2

    # Domain shift: should equal A applied to shifted input
    T2 = fftshift_op(A; domain_shifts = (1,))
    y1 = T2 * x
    y2 = A * FFTW.fftshift(x, (1,))
    @test y1 ≈ y2

    # ifftshift variants
    T3 = ifftshift_op(A; codomain_shifts = (1,))
    y1 = T3 * x
    y2 = FFTW.ifftshift(A * x, (1,))
    @test y1 ≈ y2

    T4 = ifftshift_op(A; domain_shifts = (1,))
    y1 = T4 * x
    y2 = A * FFTW.ifftshift(x, (1,))
    @test y1 ≈ y2

    # Odd length
    n = 5
    A = DFT(n)
    x = randn(n)

    T5 = fftshift_op(A; codomain_shifts = (1,))
    @test (T5 * x) ≈ FFTW.fftshift(A * x, (1,))

    T6 = ifftshift_op(A; domain_shifts = (1,))
    @test (T6 * x) ≈ (A * FFTW.ifftshift(x, (1,)))

    # Compose operators: DiagOp * DFT (all-diagonal/DFT → _is_dft_op true from all() branch)
    n = 8
    Random.seed!(42)
    dft_c = DFT(ComplexF64, n)
    d = randn(ComplexF64, n)
    diag_op = DiagOp(d)
    composed1 = diag_op * dft_c  # Compose: DFT applied first, then DiagOp
    xc = randn(ComplexF64, n)
    T7 = fftshift_op(composed1; domain_shifts = (1,))
    @test T7 * xc ≈ composed1 * FFTW.fftshift(xc, (1,))
    T8 = fftshift_op(composed1; codomain_shifts = (1,))
    @test T8 * xc ≈ FFTW.fftshift(composed1 * xc, (1,))

    # MatrixOp * DFT (else branch: not all-diagonal/DFT, but first subop is DFT)
    mat_op = MatrixOp(randn(ComplexF64, n, n))
    composed2 = mat_op * dft_c  # Compose: DFT applied first, then MatrixOp
    T9 = fftshift_op(composed2; domain_shifts = (1,))
    @test T9 * xc ≈ composed2 * FFTW.fftshift(xc, (1,))
    T10 = fftshift_op(composed2; codomain_shifts = (1,))
    @test T10 * xc ≈ FFTW.fftshift(composed2 * xc, (1,))
end

@testitem "Combination rules: FFTShift/IFFTShift with DFT/IDFT" tags = [:fftw, :CombinationRules] setup = [TestUtils] begin
    using AbstractOperators
    using LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: can_be_combined, combine

    # Even length → can be combined; replacement with SignAlternation
    n = 8
    x = randn(ComplexF64, n)
    dft = DFT(ComplexF64, n)
    sh = FFTShift(ComplexF64, (n,), (1,))
    ish = IFFTShift(ComplexF64, (n,), (1,))

    @test can_be_combined(dft, sh)
    @test can_be_combined(sh, dft)
    @test can_be_combined(dft', sh)
    @test can_be_combined(sh, dft')

    c1 = combine(dft, sh)
    @test c1 isa AbstractOperator
    @test (c1 * x) ≈ (SignAlternation(codomain_type(dft), size(dft, 1), (1,)) * (dft * x))

    c2 = combine(sh, dft)
    @test (c2 * x) ≈ (dft * (SignAlternation(domain_type(dft), size(dft, 2), (1,)) * x))

    # IDFT variants
    idft = IDFT(ComplexF64, n)
    @test can_be_combined(idft, sh)
    @test can_be_combined(sh, idft)

    c3 = combine(idft, sh)
    @test (c3 * x) ≈ (SignAlternation(codomain_type(idft), size(idft, 1), (1,)) * (idft * x))

    c4 = combine(sh, idft)
    @test (c4 * x) ≈ (idft * (SignAlternation(domain_type(idft), size(idft, 2), (1,)) * x))

    # Odd length → combined into a DFT and a phase-ramp DiagOp
    n = 7
    x = randn(ComplexF64, n)
    dft = DFT(ComplexF64, n)
    sh = FFTShift(ComplexF64, (n,), (1,))
    @test can_be_combined(dft, sh)
    @test can_be_combined(sh, dft)
    @test combine(dft, sh) * x ≈ dft * (sh * x)
    @test combine(sh, dft) * x ≈ sh * (dft * x)
    # a real-input DFT cannot take the complex ramp on its input
    @test !can_be_combined(sh, DFT(Float64, n))

    # Multi-dim dirs
    n, m = 6, 10
    dft2 = DFT(ComplexF64, n, m)
    sh2 = FFTShift(ComplexF64, (n, m), (1, 2))
    @test can_be_combined(dft2, sh2)  # both even

    c5 = combine(dft2, sh2)
    X = randn(ComplexF64, n, m)
    @test (c5 * X) ≈ ((dft2 * sh2) * X)
end

@testitem "Shifts and their combinations keep the threading they were built with" tags = [:fftw, :CombinationRules, :threading] setup = [TestUtils] begin
    using AbstractOperators
    using LinearAlgebra, FFTWOperators
    using AbstractOperators: combine, is_threaded, get_operators

    # Large enough that an element-wise kernel built with `threaded = true` does thread (256²
    # is still below the memory-bound threshold), so a serial result is not the size gate.
    sz = (512, 512)
    x = randn(ComplexF64, sz)
    sign_alternations(op) = filter(o -> o isa SignAlternation, collect(get_operators(op)))

    for threaded in (false, true)
        dft = DFT(zeros(ComplexF64, sz); threaded)
        # A shift that becomes a sign alternation threads exactly when the transform does.
        for shifted in (
                ifftshift_op(dft; domain_shifts = (1, 2), codomain_shifts = (1, 2)),
                fftshift_op(dft; domain_shifts = (1, 2), codomain_shifts = (1, 2)),
            )
            @test length(sign_alternations(shifted)) == 2
            @test all(s -> is_threaded(s) == is_threaded(dft), sign_alternations(shifted))
        end
        for sh in (FFTShift(ComplexF64, sz, (1, 2)), IFFTShift(ComplexF64, sz, (1, 2)))
            @test is_threaded(only(sign_alternations(combine(dft, sh)))) == is_threaded(dft)
            @test is_threaded(only(sign_alternations(combine(sh, dft)))) == is_threaded(dft)
        end

        # Folding a sign alternation into a diagonal keeps the diagonal's threading, whatever
        # the sign alternation's.
        d = randn(ComplexF64, sz)
        D = DiagOp(d; threaded)
        S = SignAlternation(ComplexF64, sz, (1, 2); threaded = !threaded)
        @test is_threaded(combine(S, D)) == is_threaded(D)
        @test is_threaded(combine(D, S)) == is_threaded(D)
        @test combine(S, D) * x ≈ S * (D * x)
        @test combine(D, S) * x ≈ D * (S * x)
    end
    @test !is_threaded(DiagOp(randn(ComplexF64, sz); threaded = false) * ifftshift_op(DFT(zeros(ComplexF64, sz); threaded = false); codomain_shifts = (1, 2)))

    # Two sign alternations merge into one that threads if either did.
    S1 = SignAlternation(ComplexF64, sz, (1,); threaded = false)
    S2 = SignAlternation(ComplexF64, sz, (2,); threaded = true)
    @test is_threaded(combine(S1, S2)) == (is_threaded(S1) || is_threaded(S2))
    @test !is_threaded(combine(S1, SignAlternation(ComplexF64, sz, (2,); threaded = false)))
end

@testitem "Odd-length shifts next to a DFT become phase ramps" tags = [:fftw, :FFTShift, :CombinationRules] setup = [TestUtils] begin
    using AbstractOperators, FFTW, FFTWOperators, LinearAlgebra, Random
    Random.seed!(3)
    has_shift(op) = op isa FFTWOperators.ShiftOp ||
        (op isa AbstractOperators.Compose && any(has_shift, AbstractOperators.get_operators(op)))
    shifted(f, x, dims) = isempty(dims) ? x : f(x, dims)
    inverse_of(f) = f === fftshift ? ifftshift : fftshift
    for dims in ((5,), (7,), (15,), (5, 6), (7, 9), (4, 5)),
            make in (d -> DFT(ComplexF64, d), d -> IDFT(ComplexF64, d), d -> DFT(ComplexF64, d)'),
            normalization in (FFTWOperators.UNNORMALIZED, FFTWOperators.ORTHO)
        F = make(dims)
        nd = length(dims)
        for (shift_op, f) in ((fftshift_op, fftshift), (ifftshift_op, ifftshift)),
                ds in ((), Tuple(1:nd), (nd,)), cs in ((), Tuple(1:nd), (1,))
            isempty(ds) && isempty(cs) && continue
            all(iseven, dims[collect((ds..., cs...))]) && continue
            T = shift_op(F; domain_shifts = ds, codomain_shifts = cs)
            x, y = randn(ComplexF64, dims), randn(ComplexF64, dims)
            @test T * x ≈ shifted(f, F * shifted(f, x, ds), cs)
            @test T' * y ≈ shifted(inverse_of(f), F' * shifted(inverse_of(f), y, cs), ds)
            @test !has_shift(T)
        end
    end

    # the ramp may thread only when the transform it sits next to does; `threaded = true` leaves
    # the decision to DiagOp's own size threshold
    for threaded in (false, true)
        F = DFT(ComplexF64, (301, 301); threaded)
        ramps = filter(op -> op isa DiagOp, collect(AbstractOperators.get_operators(fftshift_op(F; domain_shifts = (1, 2)))))
        @test length(ramps) == 1
        expected = AbstractOperators.is_threaded(DiagOp(ComplexF64, (301, 301), ones(ComplexF64, 301, 301); threaded = AbstractOperators.is_threaded(F)))
        @test AbstractOperators.is_threaded(only(ramps)) == expected
    end

    # the ramp merges with a diagonal next to it
    n = 9
    F = DFT(ComplexF64, n)
    d = randn(ComplexF64, n)
    T = fftshift_op(F; domain_shifts = (1,)) |> op -> DiagOp(d) * op
    x = randn(ComplexF64, n)
    @test T * x ≈ d .* (F * fftshift(x))

    # the DFT does not transform the shifted dimension: the shift stays
    F = DFT(ComplexF64, (5, 4), (2,))
    T = fftshift_op(F; domain_shifts = (1,))
    x = randn(ComplexF64, 5, 4)
    @test has_shift(T)
    @test T * x ≈ F * fftshift(x, 1)

    # real input: an output shift cannot become a ramp on the input
    F = DFT(Float64, 7)
    T = fftshift_op(F; codomain_shifts = (1,))
    x = randn(7)
    @test has_shift(T)
    @test T * x ≈ fftshift(F * x)
    T = fftshift_op(F; domain_shifts = (1,))
    @test !has_shift(T)
    @test T * x ≈ F * fftshift(x)
end

@testitem "Even-length shifts on both sides of a DFT" tags = [:fftw, :FFTShift, :CombinationRules] setup = [TestUtils] begin
    using AbstractOperators, FFTW, FFTWOperators, Random
    Random.seed!(4)
    # a shift by N/2 and a sign alternation along a dimension of length N anticommute when N/2 is
    # odd, so these sizes are where a dropped sign shows
    for dims in ((4,), (6,), (10,), (6, 8), (6, 6), (4, 6)), (shift_op, f) in ((fftshift_op, fftshift), (ifftshift_op, ifftshift))
        nd = length(dims)
        F = DFT(ComplexF64, dims)
        T = shift_op(F; domain_shifts = Tuple(1:nd), codomain_shifts = Tuple(1:nd))
        x = randn(ComplexF64, dims)
        @test T * x ≈ f(F * f(x))
        @test T' * x ≈ (f === fftshift ? ifftshift : fftshift)(F' * (f === fftshift ? ifftshift : fftshift)(x))
    end
    # the DFT does not transform the shifted (even) dimension: the shift stays
    F = DFT(ComplexF64, (6, 4), (2,))
    x = randn(ComplexF64, 6, 4)
    @test fftshift_op(F; domain_shifts = (1,)) * x ≈ F * fftshift(x, 1)
    @test fftshift_op(F; codomain_shifts = (1,)) * x ≈ fftshift(F * x, 1)
end

@testitem "FFTShift/IFFTShift (GPU)" tags = [:gpu, :fftw, :FFTShift] setup = [TestUtils, GpuEnvSetup] begin
    using FFTWOperators, GPUEnv, LinearAlgebra

    for backend in gpu_backends()
        n = 4
        x = to_gpu(backend, collect(1.0:n))
        A = FFTShift((n,), (1,); array_type = typeof(x))
        y = A * x
        @test collect(y) == [3.0, 4.0, 1.0, 2.0]
        @test collect(A' * y) ≈ collect(x)

        n2, m2 = 2, 4
        X = gpu_ones(backend, Float64, n2, m2)
        A2 = FFTShift((n2, m2), (1, 2); array_type = typeof(X))
        Y = A2 * X
        @test collect(Y) ≈ ones(n2, m2)
    end
end

@testitem "SignAlternation (GPU)" tags = [:gpu, :fftw, :SignAlternation] setup = [TestUtils, GpuEnvSetup] begin
    using FFTWOperators, GPUEnv, LinearAlgebra

    for backend in gpu_backends()
        n = 6
        x = gpu_ones(backend, Float64, n)
        S = SignAlternation((n,), (1,); array_type = typeof(x))
        y = S * x
        @test collect(y) == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        @test collect(S * y) ≈ collect(x)

        X2 = gpu_ones(backend, Float64, 2, 2)
        S2 = SignAlternation((2, 2), (1, 2); array_type = typeof(X2))
        Y2 = S2 * X2
        @test collect(Y2) == [1.0 -1.0; -1.0 1.0]
    end
end
