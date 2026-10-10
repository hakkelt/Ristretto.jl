using TestItems

@testsnippet RegTestSetup begin
    using Test
    using Ristretto
    using Ristretto: Regularization, get_operator, get_affected_dims,
        materialize, materialize_with_auxiliaries, materialize_all,
        scale_regularization, bind_dimensions, calculate
    using Ristretto.AbstractOperators
    using Ristretto.StructuredOptimization
    using NamedDims
    using Wavelets
end

@testmodule ProxOf begin
    using Ristretto
    using Ristretto.StructuredOptimization: Variable

    export SO, PC, functions_of, prox_of

    const SO = Ristretto.StructuredOptimization
    const PC = Ristretto.ProximalCore

    function functions_of(reg, x)
        term = Ristretto.materialize(reg, Variable(x); threaded = false)
        return SO.weighted_function(term)
    end

    function prox_of(reg, x, γ = 1.0)
        y = similar(x)
        value = PC.prox!(y, functions_of(reg, x), x, γ)
        return y, value
    end
end

@testmodule TestHelpers begin
    using Test: @test
    using LinearAlgebra: norm
    using Ristretto: ReconImage

    export relative_error, test_type_stable

    relative_error(z, truth) = norm(z .- truth) / norm(truth)
    # `reconstruct` returns a `ReconImage`; the type checked is that of the image it holds.
    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)
    test_type_stable(::Type{T}, value::ReconImage) where {T} = (@test typeof(parent(value)) == T; value)
end

@testmodule FiniteDiff begin
    export manual_gradient

    # Forward difference at the first index along dimension `d`, backward difference elsewhere --
    # matches the boundary convention `get_operator` uses for the (Second)TotalVariation family.
    function manual_gradient(x::AbstractArray, ndims_spatial::Int)
        step(d) = CartesianIndex(ntuple(k -> k == d ? 1 : 0, ndims(x)))
        manual = zeros(eltype(x), size(x)..., ndims_spatial)
        for d in 1:ndims_spatial, idx in CartesianIndices(x)
            manual[idx, d] = if idx[d] == first(axes(x, d))
                x[idx + step(d)] - x[idx]
            else
                x[idx] - x[idx - step(d)]
            end
        end
        return manual
    end
end

@testsnippet IterationCallbackSetup begin
    using Test
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo
    using Random

    # Fully sampled and single-coil, so the encoding operator is square and every algorithm in
    # `DEFAULT_ALGORITHMS` -- CG included -- can be run against the same problem.
    function square_acquisition(nx = 16, ny = 16; seed = 5)
        Random.seed!(seed)
        x_true = rand(ComplexF32, nx, ny)
        acq = CartesianAcquisitionInfo(; is3D = false, image_size = (nx, ny))
        return simulate_acquisition(x_true, acq; inverse_crime_check = false, keep_sensitivity_maps = true), x_true
    end

    # Multi-slice, so `get_task_splitting_plan` splits it into one task per slice.
    function multislice_acquisition(nx = 16, ny = 16, nslices = 4, nc = 2; seed = 7)
        Random.seed!(seed)
        smaps = repeat(coil_sensitivities(nx, ny, nc), 1, 1, 1, nslices)
        kspace = rand(ComplexF32, nx, ny, nc, nslices)
        return AcquisitionInfo(kspace; is3D = false, sensitivity_maps = smaps)
    end
end

@testmodule SyntheticCoils begin
    export synthetic_sensitivities

    # A smooth, complex-valued coil pattern: a Gaussian blob offset around a ring per coil, with a
    # linear phase ramp, normalized so the coils combine to unit magnitude (root-sum-of-squares).
    function synthetic_sensitivities(::Type{T}, Nx, Ny, Nc; phase_scale = 0.5) where {T}
        X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
        Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
        sens = zeros(T, Nx, Ny, Nc)
        for c in 1:Nc
            a = (c - 1) * 2π / Nc
            sens[:, :, c] = exp.(-((X .- cos(a) / 2) .^ 2 .+ (Y .- sin(a) / 2) .^ 2)) .*
                cis.(phase_scale .* (X .* cos(a) .+ Y .* sin(a)))
        end
        sens ./= sqrt.(sum(abs2, sens; dims = 3)) .+ 1.0e-8
        return sens
    end
end

@testmodule RadialCalibration begin
    using Ristretto
    using Ristretto: NonCartesianAcquisitionInfo
    using NamedDims
    using LinearAlgebra: dot, norm

    export radial_case, map_alignment

    # A radial acquisition of a block phantom through known coil sensitivities: everything the
    # non-Cartesian sensitivity-estimation tests calibrate from.
    function radial_case(; N = 64, ncoil = 4, nsamp = 128, nspokes = 96)
        img = zeros(ComplexF32, N, N)
        img[16:48, 20:44] .= 1
        smaps = ComplexF32.(coil_sensitivities(N, N, ncoil))
        traj = Float32.(radial_trajectory(nsamp, nspokes))
        sim = simulate_acquisition(
            NamedDimsArray{(:x, :y)}(img),
            NonCartesianAcquisitionInfo(
                nothing; trajectory = traj, image_size = (N, N),
                sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps),
            ); inverse_crime_check = false, keep_sensitivity_maps = true
        )
        return (; img, smaps, traj, kspace = sim.kspace_data, mask = abs.(img) .> 0.5, N, ncoil)
    end

    # Sensitivity maps are defined only up to a common phase per pixel, so maps are compared by
    # the direction of the coil vector, not by its phase: 1 is a perfect match.
    function map_alignment(a, b, mask)
        a, b = unname(a), unname(b)
        vals = [
            abs(dot(a[i, j, :], b[i, j, :])) / (norm(a[i, j, :]) * norm(b[i, j, :]) + eps(Float32))
                for i in axes(a, 1), j in axes(a, 2)
        ]
        return sum(vals[mask]) / count(mask)
    end
end

@testsnippet WaveletHelpers begin
    using Ristretto.WaveletOperators: WaveletOp

    # The inverse must recover the original signal, whether or not the forward pass padded it.
    check_wavelet_roundtrip(op, x, result) = (Test.@test op' * result ≈ x rtol = 1.0e-10)
end

@testmodule ModelEval begin
    using Ristretto
    using Ristretto.StructuredOptimization

    export eval_term

    function eval_term(terms)
        vars = StructuredOptimization.extract_variables(terms)
        @assert length(vars) == 1
        xvar = vars[1]
        # `weighted_function` is λ·f and nothing else, so the displacement has to come from the
        # *affine* operator rather than the bare linear one.
        f = StructuredOptimization.weighted_function(terms)
        op = StructuredOptimization.extract_affines((xvar,), terms)
        xval = ~xvar
        return f(op * xval)
    end
end

@testmodule GpuEnvSetup begin
    # Loads every GPU backend the machine has (JLArrays always, CUDA and friends where a device
    # is present) into this test process, which makes `RistrettoGPUExt` load too.
    using GPUEnv
    GPUEnv.activate(; include_jlarrays = true, persist = true)
end

@testsnippet GpuHelpers begin
    using GPUEnv: gpu_backends
    using Ristretto: AcquisitionInfo, _is_device
    using NamedDims: NamedDimsArray, unname
    using LinearAlgebra: norm
    const Adapt = Ristretto.Adapt

    # The backends a case can run on: anything with an FFT (and an NFFT) needs a real device, as
    # JLArrays has no FFT; the rest runs on JLArrays too, which is also what CI has.
    fft_backends() = gpu_backends(; include_jlarrays = false, supports_fftw = true)
    all_backends() = gpu_backends(; include_jlarrays = true)

    to_device(backend, x) = Adapt.adapt(backend.array_type, x)

    _values(x::AbstractArray) = Array(unname(x))
    _values(x::AcquisitionInfo) = (_values(x.kspace_data), isnothing(x.sensitivity_maps) ? nothing : _values(x.sensitivity_maps))
    _values(x::Tuple) = map(_values, x)
    _values(::Nothing) = nothing
    _values(x::Number) = x

    _relerr(a::AbstractArray, b::AbstractArray) = norm(a - b) / norm(b)
    _relerr(a::Tuple, b::Tuple) = maximum(map(_relerr, a, b))
    _relerr(::Nothing, ::Nothing) = 0.0
    _relerr(a::Number, b::Number) = abs(a - b) / abs(b)

    _on_device(x) = _is_device(x)
    _on_device(x::Tuple) = _on_device(first(x))

    """
        test_on_devices(f, args...; rtol, backends = fft_backends())

    `f(args...)` on the host and on every backend's device copy of `args`: the device result must
    be on the device and agree with the host one to `rtol`. Without a backend nothing runs, not
    even the host case, which the item checks on its own.
    """
    function test_on_devices(f, args...; rtol = 1.0e-4, backends = fft_backends())
        isempty(backends) && return nothing
        ref = f(args...)
        for backend in backends
            @testset "$(backend.name)" begin
                out = f(map(a -> to_device(backend, a), args)...)
                @test _on_device(out)
                @test _relerr(_values(out), _values(ref)) < rtol
            end
        end
        return nothing
    end
end
