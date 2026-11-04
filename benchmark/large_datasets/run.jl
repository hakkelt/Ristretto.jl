# Ristretto on two full-size scanner datasets, one Cartesian and one non-Cartesian, at a given host thread
# count or on a CUDA device:
#
#   julia --project=benchmark -t N benchmark/large_datasets/run.jl [options]
#
# Options:
#   --device=cuda          reconstruct on the first CUDA device (host data in, host image out)
#   --cases=knee,breast    substrings of the case ids (default: both)
#   --methods=cgsense,...  methods (default: cgsense, wavelet, tv, atv_pd)
#   --prepare              only prepare (and cache) the cases, time nothing
#
# The cases (prepared once, then loaded from `<RISTRETTO_BENCH_WORK_DIR>/large_*.jls`):
#
# | id | data | size |
# |---|---|---|
# | `large_3d_knee_cartesian` | MRIDATA knee `52c2fd53-…`, the whole volume | 320×320×256, 8 coils, ky-kz variable density, R = 6 |
# | `large_multislice_breast_radial` | fastMRI breast `fastMRI_breast_006_2`, every partition | 83 slices of 320², 16 coils, 144 of 288 golden-angle spokes × 320 samples |
#
# Maps are `SelfCalibrating` (the central 24-wide calibration region normalised by its
# root-sum-of-squares), from the fully sampled data. The knee's reference is the SENSE combination
# of the fully sampled volume. The breast is a stack of stars: the partitions are
# inverse-Fourier transformed along kz into slices, the twice-oversampled readout is cropped to its
# central 320 samples (the full field of view on a 320² grid), maps come per slice from all 288
# spokes, and the reference is a 30-iteration CG-SENSE from all 288 spokes. The breast
# series is DCE, so its reference averages the contrast change.
#
# Every method runs at the fixed effort of the cross-toolkit comparison suite (`OUTER_ITERATIONS`, `CG_ITERATIONS`,
# `PDHG_ITERATIONS`, no early stop) and λ of the case's synthetic analogue. Each row is compiled on
# the analogue first, then timed once. Results go to `results/<backend>.json` next to this script.

include(joinpath(@__DIR__, "..", "utils", "cli.jl"))

const DEVICE = _arg("device", "cpu")
DEVICE in ("cpu", "cuda") || error("--device=$DEVICE: expected cpu or cuda")
const ON_GPU = DEVICE == "cuda"

using ThreadPinning
let allowed = findall(==(1), getaffinity()) .- 1
    physical = filter(!ThreadPinning.ishyperthread, allowed)
    if length(physical) >= Threads.nthreads()
        pinthreads(physical[1:Threads.nthreads()])
    elseif !_flag("prepare")
        error("only $(length(physical)) physical cores allowed, $(Threads.nthreads()) threads requested")
    end
end

if ON_GPU
    using GPUEnv
    GPUEnv.activate(; include_jlarrays = false, only_first = true, persist = true)
    using CUDA
    CUDA.functional() || error("--device=cuda but CUDA is not functional on $(gethostname())")
end

using Ristretto, NamedDims, JSON, Printf, Random, Dates, LinearAlgebra
using Serialization: serialize, deserialize
include(joinpath(@__DIR__, "..", "utils", "bench_utils.jl"))
using .BenchUtils
const B = BenchUtils

load_site_env!()
ensure_download_path!()

# ---------------------------------------------------------------- case preparation

const LARGE_CACHE_VERSION = 1

# ESPIRiT takes 18 s per 320×256 slice and 38 s per 320² radial slice here, 1.6 h for the knee's
# 320 readout positions; the calibration-region estimate takes seconds.
const MAPS = Ristretto.SelfCalibrating(; calib_size = 24)

function _raw(source, id)
    M = B._mritestdata()
    return Base.invokelatest(M.load_raw, Base.invokelatest(M.dataset, B._source(source), id; offline = true))
end

function prepare_knee()
    id = "large_3d_knee_cartesian"
    raw = _raw("MRIDATA", "52c2fd53-d233-4444-8bfd-7c454240d314")
    nky = maximum(Int(B._idx(p).kspace_encode_step_1) for p in raw.profiles) + 1
    nkz = maximum(Int(B._idx(p).slice) for p in raw.profiles) + 1
    nx = length(B._readout_range(raw.profiles[1]))
    k = ComplexF32.(B._assemble_cartesian_3d_centre(raw, (nx, nky, nkz)))      # (kx, ky, kz, coil)
    raw = nothing
    acq = Ristretto.CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :kz, :coil)}(k); is3D = true, shifted_image_dims = (:x, :y, :z),
    )
    maps = Array(unname(Ristretto.estimate_sensitivities(acq; method = MAPS).sensitivity_maps))
    acq = nothing
    ref = B._sense_combine(centred_ifft(k, (1, 2, 3)), maps)
    seed = B._case_seed(id)
    yz = B.vd_mask_2d(MersenneTwister(seed), nky, nkz; R = 6, calib = 24)
    mask = BitArray(repeat(reshape(yz, 1, nky, nkz), nx, 1, 1))
    return B._real_cartesian(
        id, "shepp_logan_3d_8ch_cartesian", :volume, ref, maps, k, mask,
        "MRIDATA:52c2fd53-d233-4444-8bfd-7c454240d314 (whole volume)", seed; heavy = true,
    )
end

function prepare_breast(; nkeep = 320, nspokes = 144)
    id = "large_multislice_breast_radial"
    raw = _raw("FASTMRI", "fastMRI_breast_IDS_001_010/fastMRI_breast_006_2")
    prof = raw.profiles
    nsamp, ncoil = size(prof[1].data)
    nsp = maximum(Int(B._idx(p).kspace_encode_step_1) for p in prof) + 1
    nz = maximum(Int(B._idx(p).slice) for p in prof) + 1
    k = zeros(ComplexF32, nsamp, nsp, ncoil, nz)
    traj = zeros(Float32, 2, nsamp, nsp)
    for p in prof
        j, z = Int(B._idx(p).kspace_encode_step_1) + 1, Int(B._idx(p).slice) + 1
        k[:, j, :, z] .= p.data
        z == 1 && (traj[:, :, j] .= p.traj[1:2, :])
    end
    c = Int(prof[1].head.center_sample) + 1
    raw = prof = nothing
    keep = (c - nkeep ÷ 2):(c + nkeep ÷ 2 - 1)
    k = centred_ifft(k[keep, :, :, :], (4,))                                       # kz -> z
    traj = clamp.(traj[:, keep, :] .* Float32(nsamp / nkeep), -0.5f0, prevfloat(0.5f0))
    k = norm_ksp(k)
    n = nkeep
    dcf = B.ramp_dcf(traj)
    maps = Array{ComplexF32}(undef, n, n, ncoil, nz)
    ref = Array{ComplexF32}(undef, n, n, nz)
    for z in 1:nz
        full = B._noncartesian_acq(k[:, :, :, z], traj, dcf, (n, n))
        est = Ristretto.estimate_sensitivities(full; method = MAPS)
        maps[:, :, :, z] .= unname(est.sensitivity_maps)
        ref[:, :, z] .= B._cgsense_reference(k[:, :, :, z], traj, maps[:, :, :, z], (n, n))
        z % 10 == 0 && @info "breast: slice $z of $nz prepared"
    end
    sel = 1:nspokes
    tsel = traj[:, :, sel]
    return BenchCase(;
        id, family = :multislice, trajectory = :noncartesian, reference = ref, smaps = maps,
        kspace = k[:, sel, :, :], traj = tsel, dcf = B.ramp_dcf(tsel), image_size = (n, n), heavy = true,
        real = true, analogue = "shepp_logan_2d_8ch_radial", seed = B._case_seed(id),
        source = "FASTMRI:fastMRI_breast_IDS_001_010/fastMRI_breast_006_2 ($nz partitions, $nspokes of $nsp spokes)",
    )
end

const PREPARE = Dict("large_3d_knee_cartesian" => prepare_knee, "large_multislice_breast_radial" => prepare_breast)

function large_case(id)
    path = joinpath(B.work_dir(), "$(id)__v$(LARGE_CACHE_VERSION).jls")
    isfile(path) && return deserialize(path)
    @info "preparing $id (cached at $path)"
    t = @elapsed c = PREPARE[id]()
    @info "prepared $id in $(round(t; digits = 1)) s"
    mkpath(dirname(path))
    serialize(path * ".tmp$(getpid())", c)
    mv(path * ".tmp$(getpid())", path; force = true)
    return c
end

# ---------------------------------------------------------------- timing

const CASES = filter(id -> any(p -> occursin(lowercase(p), id), _list("cases", ["knee", "breast"])), sort(collect(keys(PREPARE))))
const METHOD_LIST = Symbol.(_list("methods", ["cgsense", "wavelet", "tv", "atv_pd"]))

const BACKEND = ON_GPU ? "cuda" : "openblas_$(Threads.nthreads())threads"
const RESULTS = joinpath(@__DIR__, "results", "$BACKEND.json")

function record!(row)
    rows = isfile(RESULTS) ? JSON.parsefile(RESULTS)["rows"] : Any[]
    filter!(r -> !(r["case_id"] == row["case_id"] && r["method"] == row["method"]), rows)
    push!(rows, row)
    sort!(rows; by = r -> (r["case_id"], r["method"]))
    meta = Dict(
        "backend" => BACKEND, "threads" => Threads.nthreads(), "hostname" => gethostname(),
        "gpu" => ON_GPU ? CUDA.name(CUDA.device()) : nothing, "ristretto_commit" => git_ref(joinpath(@__DIR__, "..", "..")),
        "outer_iterations" => OUTER_ITERATIONS, "cg_iterations" => CG_ITERATIONS, "pdhg_iterations" => PDHG_ITERATIONS,
        "rows" => rows,
    )
    mkpath(dirname(RESULTS))
    open(io -> JSON.print(io, meta, 2), RESULTS * ".tmp", "w")
    return mv(RESULTS * ".tmp", RESULTS; force = true)
end

function release!()
    GC.gc(true)
    return ON_GPU && CUDA.reclaim()
end

for id in CASES
    c = large_case(id)
    @info "case" c
    _flag("prepare") && continue
    analogue = get_case(c.analogue)
    for method in METHOD_LIST
        device = ON_GPU ? CuArray : nothing
        λ = default_lambda(c, method)
        # Compile on the analogue: same family, trajectory and method, so the same method instances.
        ristretto_reconstructor(analogue, method; λ, device)()
        release!()
        f = ristretto_reconstructor(c, method; λ, device)
        ON_GPU && CUDA.synchronize()
        t0 = time_ns()
        x = f()
        ON_GPU && CUDA.synchronize()
        t = (time_ns() - t0) / 1.0e9
        img = Array(unname(x))
        err = Float64(mag_nrmse(img, c.reference))
        mem = ON_GPU ? CUDA.memory_status : nothing
        @printf("%-32s %-8s %10.2f s  NRMSE %.4f\n", id, method, t, err)
        record!(
            Dict(
                "case_id" => id, "method" => String(method), "time_s" => t, "nrmse" => err, "lambda" => λ,
                "maxit" => default_maxit(method), "image_size" => collect(size(c.reference)), "coils" => ncoils(c),
                "maxrss_gb" => Sys.maxrss() / 2^30, "date" => string(now()),
            ),
        )
        x = img = f = nothing
        release!()
    end
end
