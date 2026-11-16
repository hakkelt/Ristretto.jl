# Task splitting on a GPU: split (one solve per slice, run one after another) against unsplit
# (one solve over the whole stack), for the cases task splitting applies to. The two variants of
# a case are timed round-robin so that drift on a shared node hits both alike.
#
#   julia --project=test benchmark/gpu_task_splitting.jl [reps]
#
# Needs a CUDA device; run it through benchmark/slurm/gpu_task_splitting.sh on an A100.

using GPUEnv
GPUEnv.activate(; include_cpu_backends = false)
using CUDA, NamedDims, Printf, Statistics, Random
using Ristretto
const Adapt = Ristretto.Adapt

CUDA.functional() || error("no functional CUDA device")
const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5

# A stack of `nb` `n × n` images, `nc` coils, 2x uniformly undersampled: the slices of a
# multislice scan (`:z`, one set of maps per slice) or the frames of a cine solved frame by frame
# (`:time`, one set of maps for all frames).
function stack_acquisition(n, nc, nb, name)
    Random.seed!(0)
    img = [ComplexF32(exp(-((x - n / 2)^2 + (y - n / 2)^2) / (n / 4)^2)) for x in 1:n, y in 1:n]
    smaps = ComplexF32.(coil_sensitivities(n, n, nc))
    pattern = create_sampling_pattern(UniformRandomSampling(2.0, 0.1), (n, n))
    one_slice = AcquisitionInfo(is3D = false, image_size = (n, n), sensitivity_maps = smaps, subsampling = pattern)
    ksp = stack(unname(simulate_acquisition((1 + 0.1f0 * b) .* img, one_slice; inverse_crime_check = false, keep_sensitivity_maps = true).kspace_data) for b in 1:nb)
    maps = name === :z ?
        NamedDimsArray{(:x, :y, :coil, :z)}(repeat(smaps, 1, 1, 1, nb)) :
        NamedDimsArray{(:x, :y, :coil)}(smaps)
    return AcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil, name)}(ksp); is3D = false, image_size = (n, n),
        sensitivity_maps = maps, subsampling = pattern,
    )
end

function timed(acq, method; disable_task_splitting)
    CUDA.synchronize()
    t = @elapsed begin
        reconstruct(acq, method; disable_task_splitting, verbosity = Silent())
        CUDA.synchronize()
    end
    return t
end

methods = (
    "CG-SENSE (20 it)" => IterativeReconstruction(L2Image(0.01f0); maxit = 20),
    "TV (20 it)" => IterativeReconstruction(TotalVariation2D(0.001f0); maxit = 20),
)
cases = [
    ("multislice", 64, 8, 16, :z),
    ("multislice", 128, 8, 16, :z),
    ("multislice", 256, 8, 16, :z),
    ("multislice", 384, 16, 8, :z),
    ("cine", 128, 8, 30, :time),
]

println("device: ", CUDA.name(CUDA.device()), ", reps: ", REPS)
@printf("%-10s %5s %3s %3s  %-18s %12s %12s %8s\n", "case", "n", "nc", "nb", "method", "split [ms]", "unsplit [ms]", "ratio")
for (label, n, nc, nb, name) in cases
    acq = Adapt.adapt(CuArray, stack_acquisition(n, nc, nb, name))
    for (mname, method) in methods
        # Warm-up: compilation, FFT plans.
        timed(acq, method; disable_task_splitting = false)
        timed(acq, method; disable_task_splitting = true)
        split, whole = Float64[], Float64[]
        for _ in 1:REPS
            push!(split, timed(acq, method; disable_task_splitting = false))
            push!(whole, timed(acq, method; disable_task_splitting = true))
        end
        s, w = 1.0e3 * median(split), 1.0e3 * median(whole)
        @printf("%-10s %5d %3d %3d  %-18s %12.1f %12.1f %8.2f\n", label, n, nc, nb, mname, s, w, s / w)
        flush(stdout)
    end
end
