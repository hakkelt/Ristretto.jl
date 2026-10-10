# Roadmap

Working plan for the next development phase, agreed on 2026-10-08. The goal is a paper presenting
the library, so documentation weighs heavily and performance work is deferred except for
low-hanging fruit.

Item numbers are stable; refer to them in branch names, commit messages and session notes. When
you start or finish an item, update its **Status** line (`todo` / `in progress (<branch>)` /
`done (<commit or PR>)`) in the same change. Repository rules in `AGENTS.md` and `NAMING.md`
apply throughout; in particular, changes to vendored packages go on fork branches, never under
`deps/`, and re-vendoring needs the maintainer's permission.

## Priorities

| Tier | Items |
|---|---|
| **P1 — done** (PR #2, 2026-10-09) | 5 (rename), 9, 2, 11, 1 (measurement + quick wins), 3 |
| **P2 — documentation (next)** | 28, 30, 29, 31, 6, 7 + 8, 10, 23, 4 |
| **P3 — strengthens the paper** | 19, 20 (Python wrapper), 22 |
| **Deferred** | 12, 13, 14, 15, rest of 1, 16, 17, 18, 24, 25, 26, 27, 32–38 |

Ordering constraints: 5 before 4, 10 and 20 (a rename touches all of them); 9 and 7 before 10
(notebooks are rewritten only once); 9 before 22 (simulated benchmark cases change); 7 before 8;
2 before 15; 11 before 12; 28 lands on master before the P2 branch opens its PR (coverage and
benchmark comments then report on every later change); 31 before 10 and 23 (the review's done
parts and comparison move into those pages).

P1 closed on 2026-10-09 with PR #2. What is left of items 1, 3 and 5 is listed in their **Status**
lines and is not P1 work any more.

## Foundations and cleanup

### 1. Package load time
**Status:** P1 part done (PR #2); the rest deferred. **Tier:** P1 (measurement + quick wins), rest deferred.

Done 2026-10-09 (`benchmark/load_time.jl` via `benchmark/slurm/load_time.sh`, compute node
x1001c4s3b0n1, Julia 1.13.1, 16 cores, 5 fresh processes per row):

- `Distributed` is gone from the load path: dropped from `[deps]`, and ProgressMeter is taken from
  its `master` through `[sources]` (timholy/ProgressMeter.jl#366 moved Distributed to an
  extension but is unreleased). **Before registering Ristretto, replace that `[sources]` entry
  with a compat bound on the first ProgressMeter release that contains #366.** On the compute
  node this saves only ~40 ms (`using` 1.37 → 1.30–1.36 s); the 409 ms above was the login node.
- A PrecompileTools workload (16² Cartesian direct + L1-wavelet, radial TV, NamedDims cine
  low-rank, 2 iterations each, `ESTIMATE` plans, wisdom off; `src/precompile.jl`) is **kept**. It
  fails the original gate (≤ 10 % load cost), but the maintainer accepted the ~2 s load for the
  first-solve speedup (2026-10-09). `precompile_workload = false` in `LocalPreferences.toml` turns
  it off.

  | | `using` | first solve, Cartesian / radial / cine | pkgimage | precompile |
  |---|---|---|---|---|
  | no workload, 1 thread | 1.33 s | 15.1 / 18.6 / 17.6 s | 19 MB | 36 s |
  | workload, 1 thread | 2.05 s (+54 %) | 1.47 / 0.27 / 0.08 s | 118 MB | 141 s |
  | no workload, 8 threads | 1.31 s | 15.0 / 20.0 / 17.0 s | | |
  | workload, 8 threads | 2.02 s | 7.4 / 8.0 / 7.5 s | | |

  The `Ristretto` entry of `@time_imports` goes from 83 ms to 806 ms: the cost is loading the
  larger image. A Cartesian-only workload still gave an 81 MB image. At 8 threads the workload
  helps only half, as precompilation runs the serial paths; covering the threaded paths is open.

Measured 2026-10-08 (login node, `-t 1`): warm `using Ristretto` takes 3.5 s.
Largest `@time_imports` entries: Distributed 409 ms (96 % compilation), SparseArrays 384 ms,
Polynomials 274 ms (via DSP), VectorizationBase + LoopVectorization ≈ 255 ms, Ristretto 186 ms,
Contourlets 90 ms, RecursiveArrayTools 73 ms. Target: 0.5–1 s.

- Distributed arrives through ProgressMeter; Ristretto's own `[deps]` entry is unused by `src/`.
  Replace ProgressMeter (ProgressLogging.jl, ProgressBars.jl, or a minimal in-house printer).
- `src/Ristretto.jl` includes the vendored DSPOperators, which no Ristretto code uses.
- DSP is otherwise needed only by Wavelets: vendor Wavelets and inline the few DSP functions it
  uses.
- SparseArrays arrives through IterativeSolvers, StatsBase and Polynomials.
- LoopVectorization stays (a deep dependency of many packages).
- Features are switched off through Preferences.jl opt-outs, not package extensions: an extension
  cannot add new names, and it is not obvious to users which package enables which feature.
  Package extensions remain the right tool for file export (item 8).
- A split into simulation / pre-processing + reconstruction / post-processing packages over a
  shared core is possible if the measurements justify it.
- **P1 (measurement):** measure the effect of PrecompileTools on both load time and
  time-to-first-solve. It is a dependency, but no workload exists yet (`src/` and `ext/` have no
  `@setup_workload`/`@compile_workload`), so this means writing a representative workload and
  comparing with and without it.

### 2. Register NestedThreading and MRITestData
**Status:** done (PR #2). **Tier:** P1.

NestedThreading 0.1.2 (JuliaRegistries/General#171087) and MRITestData 0.1.1 (General#171085,
the USC Speech dwell-time fix the examples need) are registered. Ristretto, `examples/`,
`benchmark/`, `test/` and the notebooks take both from the registry. Vendoring does not block
registering Ristretto itself.

### 3. Comment cleanup
**Status:** done for Ristretto `src/` and `ext/` (PR #2); vendored code open. **Tier:** P1.

Done 2026-10-09: process history ("used to", "TODO 6"), measurement tables and timings moved out of
the comments into the commit messages, and what-comments dropped, in `src/reconstruction`,
`src/encoding`, `src/acquisition_data`, `src/preprocessing` and `ext/`; one comment-only commit
per directory. Why-comments, invariants and references stay.

Comments are often too verbose. Ristretto `src/` and `ext/` first; comments in vendored code are changed
on the fork branch that owns the code.

### 4. README
**Status:** done (`p2`, 2026-10-10): installation, three high-level examples, a
low-level one, a benchmark table from `docs/benchmark_tables.jl`, badges; `test/test_readme.jl`
runs its code. **Tier:** P2. **After:** 5.

Short and focused: what the package is, how to install Julia (one or two lines: juliaup), how to
install the package, two or three examples, and a small benchmark table from the committed
comparison results.

### 5. New name
**Status:** package renamed (branch `rename-ristretto`; GitHub repo renamed to
`hakkelt/Ristretto.jl`, UUID kept; the former abbreviation replaced everywhere, including
environment variables (`RISTRETTO_*`) and stored benchmark results). Still open: comments in the
vendored AbstractOperators code naming the package (fork branches), the local checkout directory
name, trademark check. **Tier:** P1, before 4, 10 and 20.

The package becomes **Ristretto** (`Ristretto.jl`) — **R**egularized **I**maging **S**olvers
**T**oolbox — **R**apid, **E**fficient, **T**hreaded, **T**unable, **O**pen. Free in the General
registry (checked 2026-10-08); a trademark check is still needed. The package is unreleased, so
rename by deleting the old name, never by deprecating it (`AGENTS.md`): module, `Project.toml`,
extensions, repository, docs, notebooks, benchmarks, `AGENTS.md`/`NAMING.md`.

### 6. Fork documentation and vendoring notes
**Status:** done (`p2`, 2026-10-10). Pages notes added; each fork's stack has `docs/fork-pages`
(AbstractOperators #55, ProximalOperators #21, ProximalAlgorithms #18, StructuredOptimization #8,
open for review), which deploys `integration` to the fork's GitHub Pages as `dev`; all four `dev`
sites are up, and the low-level pages and tutorial 12 link them. **Tier:** P2.

- Pages presenting a vendored package (`docs/src/low-level/abstract_operators.md`,
  `proximal_operators.md`, `custom_reconstruction.md`, ...) warn that it must be imported through
  Ristretto, say that vendoring is temporary, and give the reason: Ristretto needs work-in-progress versions
  of the forks that are not registered, while upstream review is slow.
- Each fork deploys its Documenter docs from its `integration` branch to GitHub Pages (none has
  Pages enabled as of 2026-10-08), and Ristretto's docs and notebooks link there.

### 29. Faster test suite
**Status:** done (`p2`, 2026-10-10). On CI (PR #4, warm package cache) the test job took 19 min 34 s,
of which the test step 17 min 53 s and the `:extension` coverage rerun 43 s: 1.1× faster than the
22 min baseline, short of the 2× target (the first run, with a cold cache, took 34 min 37 s). Baseline
2026-10-09: CI test job 22 min with coverage; locally 214 items, 3567 s summed on 4 workers.
Measured item by item in one process as CI runs them (1 thread, no GPU): 3909 s with
`--code-coverage=user`, 2393 s after (225 items), which includes ~350 s of one-off precompilation
that `--check-bounds=yes` (as `Pkg.test` sets it) causes and the baseline did not pay. Most of the
gain: coverage now counts only `src/` (and `ext/` in a second run of the `:extension` items), since
`user` also instrumented the vendored `deps/` and made solver loops 4–10× slower (TGV3D 375 s →
21 s). The rest: shorter ADMM/VuCondat budgets where the extra iterations moved nothing (TGV,
infimal convolution, preconditioned Chambolle–Pock reference), and no host solve in
`test_on_devices` when there is no device. No `:slow` tag was needed. **Tier:** P2.

The full suite takes about 25 minutes on CI and on the login node. Find where the time goes
(per-item durations from the TestItems runner) and cut it without losing coverage:
- smaller inputs where the size does not matter to what is checked;
- duplicated checks, e.g. the same property tested in several items or on several equivalent
  configurations;
- checks that no longer guard anything (superseded behaviour, or covered by a cheaper test);
- shared fixtures built once in a `@testmodule` instead of in every item;
- compilation: items that pay for a large, unique set of method instances.

Report the per-item time before and after, and keep the slow, high-value cases (real data,
integration) behind a tag if they cannot be shrunk.

### 30. Re-run the examples
**Status:** done (`p2`): all 35 scripts ran on 2026-10-09 at `0e3aabe1`; table in
`examples/README.md`. **Tier:** P2.

Run every `examples/` script (`examples/run_all.jl`; the large datasets in one SLURM job) against
the current package, fix what broke in `ext/RistrettoMRIBaseExt.jl`, preprocessing or `src/`
(with a test where a synthetic equivalent exists), and record a "Last verified" table (date,
commit, ok / failed / skipped per source) in `examples/README.md`. Sources behind expiring
credentials (fastMRI signed URLs, the CMRxRecon Synapse token) are reported as skipped when the
credential is missing, not as failures.

### 31. Literature review into the docs and the roadmap
**Status:** done (`p2`, 2026-10-10). Not implemented: items 32–38 and notes on 16, 17 and 24 (open
question 2, the non-linear solver strategy, is on item 32; 3 is on item 33; 1 is item 16; 4 no
longer applies). Implemented: a few lines and the seminal references on the page where each
feature lives (methods, preprocessing, acquisition data, analysis, regularization, theory), the
method-API design in `low-level/custom_reconstruction.md`, the feature matrix on the
related-packages page (item 23). The review file is deleted. **Tier:** P2. **Before:** 10, 23.

`comprehensive_literature_review_mri_toolboxes.md` is too long to keep as it is. Audit it against
`src/`: what is implemented is compacted into the documentation page where the feature lives (a
few lines and the seminal reference each), the toolbox feature matrix goes to the related-packages
page (item 23), and what is not implemented becomes roadmap items. Then delete the file.

## User-facing features

### 7. Metadata header and result type
**Status:** done on `p2` (2026-10-09). `Header` is a dictionary whose known keys are typed fields;
it is optional on `AcquisitionInfo`, shared by its copies, and filled from MRD by the MRIFiles
extension. `reconstruct` returns a `ReconImage` with its own copy; keyword slicing moves `offset`.
`DecomposedImage` is gone: a `ReconImage` holds the components as plain arrays, sliced with it
and released by `drop_components`. Geometry is stored in LPS. **Tier:** P2.

- `AcquisitionInfo` gets a header holding arbitrary key/value metadata.
- `reconstruct` returns a type that is a subtype of `AbstractArray`, carrying geometry (FOV, voxel
  size, orientation, position) and the header forwarded from `AcquisitionInfo`.
- Tags can be added on both `AcquisitionInfo` and the result image.
- Header contents come from the MRD header (MRIFiles extension) or other acquisition metadata, and
  feed the export formats of item 8.

The design note written before the implementation (reviewed 2026-10-09) now lives in the
`Header` and `ReconImage` docstrings and the acquisition-data, reconstruction and export pages.

### 8. Export to NIfTI, DICOM, MRD
**Status:** done on `p2` (2026-10-10). Saving follows the FileIO convention: `save("x.nii", img)`
(FileIO + NIfTI extension; RAS `sform`, BIDS-style JSON sidecar), `save("x.dcm", img)` (FileIO +
DICOM extension; magnitude MR image series, 16-bit with a rescale slope, tags in `ImageComments`)
and `save(ISMRMRDFile("x.h5"), img)` (MRIFiles extension; ISMRMRD HDF5 images, tags as meta
attributes). Each file was read back by nibabel, pydicom and the `ismrmrd` Python package with the
same geometry; docs page `high-level/export.md`. **Tier:** P2. **After:** 7.

One package extension per format. Known keys of the header map to DICOM tags, NIfTI fields and
MRD image header fields; geometry provides the affine.

### 9. Simulation without the inverse crime
**Status:** done (PR #2). **Tier:** P1.

Done 2026-10-09: `simulate_acquisition(phantom, acq; inverse_crime_check = true,
keep_sensitivity_maps = false)` simulates on the phantom's grid and keeps the reconstruction
grid's frequencies (Cartesian) or samples the same physical frequencies (non-Cartesian); maps are
made at the phantom size and dropped from the result unless kept. GeometricMedicalPhantoms 1.1.0
adds `supersample` (area sampling). Study (`benchmark/inverse_crime/study.jl`, 128² Shepp–Logan,
table in `docs/src/high-level/simulation.md`): area-sampled 202² (s = 1.58) gives 3.2 % Cartesian
k-space error, the 30 dB noise level. The TV-reconstruction SER bias against the area-sampled
truth is +1.3–1.5 dB for 1.5 ≤ s ≤ 2 (0.6 dB at 1.3 and 3), so the planned < 0.5 dB target is not
met by any ratio up to 2; part of it is the box filter of area sampling. Existing tests, docs pages
and notebook sources pass `inverse_crime_check = false, keep_sensitivity_maps = true` explicitly;
moving them to finer phantoms and estimated maps is left to item 10.

Kaipio & Somersalo, doi:10.1016/j.cam.2005.09.027. Simulating on the reconstruction grid with
the reconstruction's own forward operator makes results optimistic.

- `simulate` requires the image size and warns when it equals the ground-truth phantom size:
  simulate from a finer phantom, reconstruct on a coarser grid.
- `simulate` wipes coil maps from the returned `AcquisitionInfo` by default (a keyword keeps
  them), so users go through a coil-map estimator.
- Not pursued: analytic phantoms (would rule out GeometricalMedicalPhantoms); exact NUDFT (the
  finer grid already makes simulation and reconstruction operators differ).
- KomaMRI as an alternative simulator: item 19.

Update tests, docs and notebooks that rely on the current behaviour.

### 10. One documentation site
**Status:** done (`p2`, 2026-10-10). The twelve notebooks are Literate scripts in
`docs/literate/`, executed into `docs/src/tutorials/` on every build and written as notebooks to
download; `docs/notebooks/` (jupytext, IJulia, export and SLURM scripts) is gone. Home page cut
to installation (juliaup), a three-line example and a map of the site; each reference page links
its tutorial. Docs CI caches the MRITestData downloads and runs 2 threads. Item 9 leftover:
tutorial 01 simulates from a finer area-sampled phantom; the other tutorials keep their phantoms
(tutorial 03 explains the inverse crime on them). **Tier:** P2. **After:** 5, 7, 9.

Merge the notebooks (`docs/notebooks/`) and the Documenter pages (`docs/src/`) into a single
Documenter site in which every topic has one home: tutorials come from the notebooks; reference,
theory and API pages stay. Complete but terse, with images wherever possible. Proposed mechanism:
Literate.jl scripts as the single source, generating both the executed pages and the `.ipynb`
files. Includes a short "Installing Julia" section (juliaup), as in item 4.

### 23. Related packages page
**Status:** done (`p2`, 2026-10-10). `docs/src/related_packages.md`: feature matrix
of BART, SigPy, Gadgetron, MRIReco.jl, MIRT.jl, MRpro and Ristretto re-verified against each
project's source and docs (October 2026), benchmark tables generated at build time from the
committed snapshots by `docs/benchmark_tables.jl` (time to accuracy, matched effort, thread
scaling, GPU, hardware), the Julia ecosystem and the pipelines. Snapshot schema 3 records
dates, CPU, GPU, Julia and BLAS; the harness now stores the CPU model. **Tier:** P2. **After:** 31.

A documentation page "Related packages" with a feature comparison against other toolboxes (from
the literature review, item 31, re-verified), the Julia MRI ecosystem (KomaMRI, MRIReco,
MRIBase/MRIFiles, MIRT, GIRFReco, MriResearchTools, ...), reconstruction pipelines (Gadgetron,
OpenRecon) in brief, and the benchmarks: the latest committed cross-toolkit comparison
(`benchmark/comparison/results/benchmark_<backend>_<n>threads.json`), Ristretto against BART,
SigPy, MRIReco, MIRT and MRpro, with the methodology (λ calibrated to matched accuracy,
time-to-accuracy) and the hardware. Generated from the committed snapshot, so re-running
`export_snapshot.jl` updates it.

### 26. Image shifts other than an fftshift
**Status:** todo. **Tier:** Deferred.

`shifted_image_dims` only says whether an image axis is fftshifted, i.e. moved by `n ÷ 2` voxels.
Acquired data can need a different shift: an off-centre FOV, a scanner that places the origin
elsewhere, or a shift by a fraction of a voxel. Let the user pass the shift per axis in voxels
(`image_shift = (dx, dy[, dz])`, possibly non-integer), applied as a k-space phase ramp, so the
image needs no copy and the encoding operator stays a DFT. An integer shift of `n ÷ 2` must give
the same result as `shifted_image_dims`; the simulation's origin bookkeeping
(`_phase_origin`, `_origin_offset` in `src/simulation/simulate_acquisition.jl`) has to follow.

### 27. Surfacelet and shearlet sparsifiers
**Status:** todo. **Tier:** Deferred.

1. Finish the surfacelet transform in Contourlets.jl (`src/transforms/surfacelet3d.jl`, with its
   GPU extension).
2. Implement shearlets there as well.
3. Integrate both as sparsifying transforms, the way contourlets are: an operator in the
   AbstractOperators fork (`ContourletOperators`) and a regularizer in `src/regularization/`
   with tests and a section in `docs/src/high-level/regularization.md`.

## Performance (fork branches)

### 11. SignAlternation fusion
**Status:** done (PR #2). **Tier:** P1.

`SignAlternation` (FFTWOperators) had no `_pw_kind`, so it never joined the fused pointwise runs of
`Compose` (`src/calculus/pointwise.jl`). It was 25–43 % of a dynamic low-rank solve in earlier
profiling. It now answers `PwMapKind` on CPU (AbstractOperators `perf/fused-pointwise`, fork PR #50)
and on device arrays (`perf/gpu-fused-pointwise`, #36); fused results equal unfused ones bit for
bit.

Measured 2026-10-09 (a one-off benchmark, compute node, 16 cores booked) on
`M S F C B` (mask, sign alternation, 2-D DFT, coil weighting, coil expansion), forward, against
the same chain with the sign alternation as a pass of its own: 1.05–1.17× faster at 4 and 8
threads (128×128×8 to 256×256×16), 0.89–1.00× at one thread. The whole fused chain is 2.2–5.3×
faster than running every operator separately at 4–8 threads.

### 12. Odd-length shift next to a DFT
**Status:** todo. **Tier:** deferred. **After:** 11.

`combination_rules.jl` folds FFTShift/IFFTShift into a DFT only for even lengths (as
`SignAlternation`). For odd lengths, with 0-based indices and ω = e^{−2πi/N}:

S_a F S_b = ω^{−ab} · diag(ω^{kb}) · F · diag(ω^{−an}),

where fftshift shifts by ⌊N/2⌋ and ifftshift by ⌈N/2⌉ (they differ for odd N), the adjoint of a
shift is the opposite shift, and the adjoint DFT conjugates ω. In N-d the ramp is separable over
the dimensions that are both shifted and transformed.

- New phase-ramp operator storing 1-D ramps per dimension.
- Combine rules: with itself and `SignAlternation` (→ ramp), with `Scale` (fold the scalar), with
  `DiagOp` (→ `DiagOp`), and a ramp next to its own adjoint cancels.
- Opt into `PwMapKind` so it fuses with other elementwise operators.
- Test against dense matrices for odd and even N, both shifts, adjoints and every normalization;
  benchmark against the shift + DFT pair.

### 13. Older performance backlog
**Status:** todo (verify against current code first). **Tier:** deferred.

The k-space shift pair in 𝒜ᴴ𝒜 is not cancelled across the sampling mask; FISTA's full
`normalize_op` cost; GPU: small 2D TV launch-bound, solver scalar syncs.

### 14. Quasi-Newton benchmark
**Status:** todo. **Tier:** deferred.

Benchmark PANOC, PANOCplus and ZeroFPR (already in ProximalAlgorithms) against FISTA and POGM by
time to accuracy. They do not work with LLR or PlugAndPlay (see their docstrings).

### 15. NFFT improvements as upstream PRs
**Status:** todo. **Tier:** deferred. **After:** 2.

Open an issue proposing the series to the NFFT.jl maintainer first (not yet aware of it). Small
PRs, readability separated from behaviour changes: TestItems, FastBroadcast, NestedThreading,
KernelAbstractions; GPU NUFFT with on-the-fly kernel evaluation. NFFT.jl contains a table it says
was taken from NFFT3 (GPL); raise it with the maintainers.

### 28. CI: coverage and benchmarks
**Status:** in progress (`ci`). **Tier:** P2.

Coverage of `src/` and `ext/` reported to Codecov from the test job. A CI workflow that runs a reduced benchmark set (`benchmark/run.jl` at the small case sizes) on
pull requests and compares it with the base branch, so a performance regression shows up in
review rather than in the next manual run. Shared CI runners are noisy (see the timing caveats
in `benchmark/`), so the job reports ratios against the base run on the same runner and flags
only large changes.

## New domains (survey first)

### 16. Learned reconstruction
**Status:** todo. **Tier:** deferred (future work in the paper).

In a separate package so it never affects core load time. Order: ChainRules `rrule` for the
operators (using the true adjoint — Ristretto's Fourier `'` is Aᴴ/N); pretrained denoisers (e.g.
SNRAware, Hugging Face weights) through the existing `PlugAndPlay` and as post-processing;
MoDL/VarNet in Lux with fastMRI weight import; RAKI; implicit neural representations; diffusion
(via PythonCall); transformers (import only). Check weight licenses individually.
Unrolled networks (VarNet: Hammernik et al., MRM 2018, doi:10.1002/mrm.26977; MoDL: Aggarwal et
al., IEEE TMI 2019, doi:10.1109/TMI.2018.2865356) train the regularizer and the step sizes end to
end through the forward operator, an execution model `reconstruct` does not have; that is why
this lives in its own package.

### 17. Post-processing
**Status:** todo. **Tier:** deferred.

Phase unwrapping, geometric distortion correction, DC artifact removal, Gaussian smoothing, Dixon,
QSM, MR fingerprinting. Prefer integrating existing packages, after checking their licenses.

Water–fat separation: multi-echo signal model with multi-peak fat and field map, IDEAL (Reeder et
al., doi:10.1002/mrm.20624) and graph-cut field-map estimation (Hernando et al.,
doi:10.1002/mrm.22177). Joint water/fat/field-map estimation is non-linear; see item 32's note on
the solver strategy.

### 18. Motion correction
**Status:** todo. **Tier:** deferred.

Survey: retrospective rigid autofocusing, joint image + motion estimation, navigator-based
correction, motion-compensated dynamic reconstruction with registration in the loop.

### 19. KomaMRI extension
**Status:** todo. **Tier:** P3.

KomaMRI outputs MRIBase `RawAcquisitionData`, which Ristretto already reads, so an extension is cheap.
It needs a sequence rather than a trajectory, so it complements `simulate` rather than replacing
it.

### 20. Access for non-Julia users
**Status:** todo. **Tier:** P3 (Python wrapper), rest deferred. **After:** 1, 5, 8.

Python wrapper through juliacall; a container image with a sysimage (Apptainer on HPC, Docker
elsewhere); a CLI that hands jobs to a persistent server, since startup would otherwise dwarf a
reconstruction. MATLAB through Python or the CLI. JuliaC `--trim` is not realistic for this code.

### 24. Self-navigator extraction
**Status:** todo. **Tier:** deferred.

Extract respiratory/cardiac self-navigation signals from the acquired data (e.g. repeated k-space
centre samples) for binning and motion-resolved reconstruction; relates to item 18.

The reconstruction half of XD-GRASP (Feng et al., doi:10.1002/mrm.25665) is already expressible
(`TemporalTotalVariation` over the motion-state dimensions); what is missing is this item: the
self-gating signal, the sorting into motion states, and the per-bin trajectories that result.

### 32. Joint image and sensitivity estimation (JSENSE, NLINV)
**Status:** todo. **Tier:** deferred.

Estimate the image and the coil sensitivities together instead of calibrating the maps first:
JSENSE (Ying & Sheng, doi:10.1002/mrm.21245) alternates between the two; NLINV (Uecker et al.,
doi:10.1002/mrm.21692) solves the bilinear problem by the iteratively regularized Gauss–Newton
method. **Open:** the non-linear solver strategy (an IRGNM in Ristretto, an alternating layer over
the existing linear solvers, or a general non-linear optimizer) is to be discussed; the choice also
shapes items 17 (water–fat) and 38 (quantitative mapping).

### 33. B₀ off-resonance correction
**Status:** todo. **Tier:** deferred.

A forward operator with off-resonance (and optionally R₂* decay) during the readout, made fast by
time segmentation or multi-frequency interpolation into a short sum of NFFTs (Sutton, Noll &
Fessler, doi:10.1109/TMI.2002.808360; Fessler et al., doi:10.1109/TSP.2005.853152), supplied
through the method's `signal_model` like `TemporalBasis`. Model factors stay on the method, so one
acquisition can be reconstructed with and without correction; revisit whether field maps belong
on `AcquisitionInfo` (they describe the acquisition) once this lands.

### 34. EPI Nyquist ghost correction
**Status:** todo. **Tier:** deferred.

Even/odd echo phase correction from navigator lines, as a preprocessing step that returns a
corrected `AcquisitionInfo` (Bruder et al., doi:10.1002/mrm.1910230211); entropy-based and
reference-free variants after that. The examples include EPI data to test it on.

### 35. Virtual conjugate coils
**Status:** todo. **Tier:** deferred.

Add the conjugate of the point-reflected k-space as extra virtual channels, so SENSE and GRAPPA
exploit the phase constraint of partial Fourier without an explicit phase estimate (Blaimer et
al., doi:10.1002/mrm.21652).

### 36. Simultaneous multi-slice
**Status:** todo. **Tier:** deferred.

SMS encoding with blipped-CAIPI shifts and slice-GRAPPA / split slice-GRAPPA (Setsompop et al.,
doi:10.1002/mrm.23097; Cauley et al., doi:10.1002/mrm.24898). Slices are then no longer separable,
so task splitting must be told the slice dimension is coupled (as `get_affected_dims` does for a
regularizer).

### 37. PROPELLER / BLADE
**Status:** todo. **Tier:** deferred.

Blade-wise rotation, translation and phase estimation from the commonly sampled k-space centre,
correlation weighting or rejection of corrupted blades, then gridding (Pipe,
doi:10.1002/(SICI)1522-2594(199911)42:5<963::AID-MRM17>3.0.CO;2-L); relates to item 18.

### 38. Model-based parameter mapping
**Status:** todo. **Tier:** deferred.

T₁, T₂ and T₂* maps estimated directly from k-space through a signal equation in the forward
model, instead of fitting reconstructed contrast images (Block, Uecker & Frahm,
doi:10.1109/TMI.2009.2023119; Sumpf et al., doi:10.1002/jmri.22633; Wang et al.,
doi:10.1002/mrm.26726). Its linearized form is `TemporalBasis` (subspace reconstruction), which
exists; the non-linear form depends on item 32's solver decision. `NonNegative` and `BoxConstraint`
already cover the regularization side.

### 25. Reconstruction pipelines: OpenRecon, Gadgetron
**Status:** todo. **Tier:** deferred.

Explore OpenRecon and Gadgetron and possible integrations (e.g. Ristretto as an MRD streaming
reconstruction server); relates to item 20.

## Paper

### 22. Paper
**Status:** todo. **Tier:** P3. **After:** 9.

Re-run the cross-toolkit benchmarks after item 9 changes the simulated cases; figures; a
reproducibility package. Deferred items appear as future work.

### 21. Upstream issues
**Status:** for the maintainer to file.

FFTW `spawnloop` segfault; RecursiveArrayTools `ArrayPartition` complex broadcast and `norm`/`dot`;
slow N-d broadcast with a small leading dimension; a JET constant-propagation workaround.
