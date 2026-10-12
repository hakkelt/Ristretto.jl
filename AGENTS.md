# AGENTS.md — Ristretto

Ristretto is a Julia package for MRI image reconstruction. It provides a
modular pipeline: acquisition data → encoding operators → regularization → reconstruction via
proximal algorithms.

## Mission

Keep changes minimal and localized; avoid unrelated refactors. Never weaken a test to force it
green — if a failure reflects a real bug, fix the source. When you touch public API, update its
docstring, the relevant `docs/src/**` page, and its tests in the same change.

## Architecture

```
AcquisitionInfo → Encoding operators → Regularization → Reconstruction
   Cartesian/       FFT/NFFT +          image/transform    ISTA/FISTA/
   NonCartesian     sensitivity maps    domain terms       ADMM/CG/CGNR
```

| Module | Directory | Role |
|---|---|---|
| Acquisition data | `src/acquisition_data/` | `AcquisitionInfo` types, dimension utilities, copy constructors |
| Encoding | `src/encoding/` | Fourier (FFT/NFFT), sensitivity map, subsampling operators; `NamedDimsOp` wrapper |
| Regularization | `src/regularization/` | one file per regularizer + `regularization.jl` (abstract type, contract, fallbacks) |
| Reconstruction | `src/reconstruction/` | `config.jl`, `build_model.jl`, `task_splitting/`, `components.jl`, `reconstruct.jl` |
| Simulation | `src/simulation/` | phantom sampling patterns, coil sensitivities, full acquisition simulation |

`src/Ristretto.jl` is the authoritative list of source files (`include` order) and
exports — read it rather than trusting a tree here.

`examples/` is a workspace member holding one script per data type of every `MRITestData` source
(35 of them), each reconstructing a real dataset. When changing the raw-data path
(`ext/RistrettoMRIFilesExt/`) or preprocessing, run the affected ones —
`julia --project=examples examples/run_all.jl <source>` — since the header defects they cover
(missing dwell time, unrecorded echo position, calibration block with a different readout length,
calibration profiles overwriting the imaging k-space centre, single-partition 3D slab) have no
synthetic equivalent in the test suite.

### Task splitting over batch dimensions

The reconstruction is split into tasks over batch (non-image, non-time) dimensions: each slab is solved
independently, with and without regularization. `reconstruct.jl` merges the component and
single-variable paths, caches the encoding operator `𝒜`, and dispatches on the regularizer's
domain. When editing this path, preserve that a `NamedDimsOp` is unwrapped and rewrapped (not
reshaped in place), and that dimension symbols are resolved to integer indices *before* the image
is unnamed into a `Variable`.

### Key dependencies (custom forks, dev-pathed under `deps/`)

`AbstractOperators` (+ `FFTWOperators`, `NFFTOperators`, `WaveletOperators`, `DSPOperators`),
`StructuredOptimization`, `ProximalOperators`, `ProximalAlgorithms`, `OperatorCore`, `NFFT`,
`NestedThreading`. These are local checkouts under `deps/` — never `Pkg.add` an upstream version;
`Pkg.instantiate` the existing Manifest.

They are inlined as **submodules** of `Ristretto` (see the `include`s at the top of
`src/Ristretto.jl`), which has two consequences worth knowing before editing them:

- Every cross-package `using`/`import` inside `deps/` must be relative (`using ..AbstractOperators`).
- NFFT.jl may be loaded next to the vendored `NFFT` (MRIReco does), and both register an
  AbstractNFFTs backend. The vendored copy activates itself only when no backend is active, so
  Ristretto code names its backend explicitly: `NFFT.plan_nfft(NFFT.backend(), ...)`. Its thread switch
  is registered with NestedThreading as the pool `:ristretto_nfft` (`src/threading_utils.jl`).
- A method that extends another package's function must have that function on an `import` list, or
  it silently defines a *new* function of the same name in the submodule and the extension is never
  seen. `FFTWOperators.has_optimized_normalop`, `NFFTOperators.is_symmetric` and
  `ProximalOperators.is_positively_homogeneous` each shadowed this way at some point; if a trait
  looks ignored, compare `Pkg.Mod.trait === Owner.trait` first.
- An `ext/` directory cannot load at all: package extensions do not apply to a submodule. What a
  vendored extension provides is either inlined into `src/` by hand, as `ProximalOperators`'
  `RecursiveArrayToolsExt` is (`deps/ProximalOperators/src/recursive_array_tools.jl`), or dropped
  when Ristretto does not need it — OSQP, and with it `IndPolyhedral`, is not vendored for that reason.
  The GPU extensions (AbstractOperators', FFTWOperators' and ProximalOperators' `ext/GpuExt`,
  ProximalOperators' `GpuRecursiveArrayToolsExt`, NFFTOperators' `NFFTOperatorsGPUArraysExt`, NFFT's
  `NFFTGPUArraysExt`) are the third case: they stay under `deps/` and
  `ext/RistrettoGPUExt.jl` `include`s each one, into a module of its own where it
  imports relatively. ProximalOperators' `ProximalOperatorsCUDAExt` is included the same way by
  `ext/RistrettoCUDAExt.jl`.

Only what Ristretto compiles is vendored. Each package's own `test/`, `docs/`, `benchmark/`, CI config
and every `ext/` except the GPU ones are pruned on every sync (`prune` in `deps/vendor.toml`);
they belong to the fork and run there. `git subtree pull` only carries changes, so a file that
stops being pruned has to be restored from the fork's `integration` branch once by hand. `deps/` is therefore absent from Ristretto's own test run as well (`JuliaTestItems.toml`).

Ristretto is **ahead of** its upstreams in places (its own fixes are pushed there as branches), so a sync
is a merge, not a copy: check whether the vendored side is the newer one before overwriting it.

That merge is the thing `deps/vendor.toml` and `deps/vendor.jl` exist to remove. The manifest
declares, per package, which fork branches make up the vendored copy and how they are stacked;
`julia deps/vendor.jl rebuild` merges them into one `integration` branch per fork, and
`julia deps/vendor.jl sync` projects that branch into `deps/` as a squashed subtree, so the
vendored copy records where it came from. Two rules follow, and they are what keep the sync
one-directional:

- **Never fix a bug under `deps/`.** Fix it on the branch whose PR introduced the code, then
  rebuild and sync. `deps/` is generated output; `sync` refuses to run over uncommitted changes
  there for exactly this reason.
- **A bugfix does not get its own branch.** It is another commit on the branch that owns the
  code. Branch and PR count should track ideas, not mistakes.

Where each fork is checked out is machine state, so it is **not** in `deps/vendor.toml`: it lives
in the untracked `deps/vendor.local.toml`, one `<Package> = "/path/to/checkout"` line per fork
that exists on this machine. A package with no entry there is simply not cloned here — `check`
still reports everything GitHub can answer for it, and `rebuild`/`patch`/`sync` say which entry
they need. Nothing tracked in this repository, and nothing pushed to a fork, may name a path of
one machine; the `[sources]` block that points the vendored copy at its siblings under `deps/` is
supplied by `deps/patches/<package>.patch` and belongs nowhere else.

`julia deps/vendor.jl check` compares the manifest against GitHub and reports mis-based PRs,
branches with no PR, branches whose PR has already merged (whose code should come from upstream
instead), branches the fork does not have or whose local tip is ahead of it, branches that push a
path of one machine, and branches on the fork that no manifest entry refers to. Branch existence
is read from GitHub via `gh`, not from remote-tracking refs, so a stale fetch cannot make an
unpushed branch look pushed.

`deps/patches/<package>.patch`, which `sync` re-applies, carries **only** what vendoring itself
forces: the relative imports, the inlined extension, the OSQP removal, the vendored `[sources]`
paths. Nothing else belongs there. Work that would make sense to the upstream package goes on the
branch that owns the code; work that is about MRI rather than about the dependency belongs in
Ristretto's own `src/`. A hunk that is neither is a sign the fix was made in the wrong place —
`julia deps/vendor.jl patch` regenerates the file, so such a hunk shows up the moment it appears.

### API gotchas

- `materialize` / `materialize_with_auxiliaries` / `materialize_all` are `public` but not exported —
  call as `Ristretto.materialize(reg, x::Variable; threaded)`, or import them
  explicitly. The same goes for `get_operator`, `calculate`, `get_encoding_operator` and the rest of
  the extension surface listed in `NAMING.md` §6.2.
- `Variable(T, dims...)` — splat, do not pass a tuple.
- `Base.reshape` on an `AbstractOperator` returns `Reshape(...)`.
- `create_sampling_pattern` returns `(:, mask)` when `subsample_freq_encoding=false` (default).
- `AbstractOperators` and `ProximalOperators` both export `Sum`; resolved via an explicit
  `using AbstractOperators: Sum`.
- The package no longer reexports its dependencies. Test items and doc examples that need
  `Variable`, `Eye`, `WaveletOp` and friends must `using StructuredOptimization` /
  `using AbstractOperators` / `using WaveletOperators: WaveletOp` themselves.

## Naming and API surface

`NAMING.md` is authoritative for how things are named and what the package exposes. Read it before
adding a type, renaming anything, or touching the export list. The rules that bite most often:

- Use the field's standard name (BART / SigPy / RegularizedLeastSquares.jl / the originating paper)
  over a name that describes the implementation.
- A regularizer's name must state the penalty, not only the transform — `L1TemporalFourier`, not
  `TemporalFourier`.
- Export concrete types and verbs a non-expert constructs or calls. Mark the extension surface —
  abstract supertypes, interface functions — `public` but do not export it. `AcquisitionInfo` is
  the one exported abstract type, because it is also a constructor.
- Never `@reexport` a whole dependency; reexport the individual names a non-expert must type.
- The package is unreleased: rename by deleting the old name, never by deprecating it.

## Adding a regularizer

New file `src/regularization/<name>_reg.jl`, `include`d in `Ristretto.jl`, type(s)
exported there. A regularizer is `struct Foo{T} <: Regularization` plus:

- `get_operator(::Foo, x::AbstractArray; threaded)` — the linear operator; wrap in `NamedDimsOp`
  when `x isa NamedDimsArray`, mapping input to output dimension names.
- `materialize(reg::Foo, x::Variable; threaded)` — build the `StructuredOptimization.Term`
  (operator ∘ norm function). Default throws.
- `get_affected_dims(::Foo, dimspec, image_dims)` — which image dims the term acts on.
- `scale_regularization(reg::Foo, factor::Real)` — only if the term is homogeneous (scale `λ`).
- `bind_dimensions(reg::Foo, image_dims)` — only if parameterized by a dim (`time_dim`, `dim`,
  possibly `nothing`/`Symbol`): resolve it to a concrete index here. Generic fallback is identity.
- `materialize_with_auxiliaries` — only if the term introduces extra optimization variables
  (see `TotalGeneralizedVariation2D`).
- A new proximal function with no MRI-specific content belongs in the `deps/ProximalOperators` fork,
  not here (`NAMING.md` §7); `ProximalAverage` and `IndAffineCG` went that way.

Add a `test/test_reg_<name>.jl` (`@testitem`, `tags = [:regularization]`) and a section in
`docs/src/high-level/regularization.md`.

## Testing

- **TestItems.jl** / **TestItemRunner.jl**. Each `@testitem` does `using Ristretto`
  and any extra packages. Multiple `@testitem` blocks per file are fine (regularizer files often
  have several); keep begin/end nesting shallow.
- Tags in use: `:encoding`, `:regularization`, `:reconstruction`, `:acquisition`, `:simulation`,
  `:minimizer`, `:components`, `:integration`, `:nfft`, `:quality` (+ `:aqua`, `:jet`),
  `:operators`, `:gpu`, `:export`, `:extension` (an item that exercises a package extension in
  `ext/`). Combine as needed.
- Device coverage lives in the existing items, not in separate ones: an item that builds a case
  adds `setup = [GpuEnvSetup, GpuHelpers]`, the `:gpu` tag, and a `test_on_devices(f, args...)`
  call after its host assertions (`test/test_snippets.jl`). `GpuEnvSetup` loads every backend
  GPUEnv finds; cases with an FFT run on `fft_backends()` (a real device), FFT-free ones on
  `all_backends()`, which includes JLArrays and is what CI without a GPU exercises.
- Full suite: `julia --project=test test/runtests.jl`
- Filtered:
  ```sh
  julia --project=test -e 'using TestItemRunner; run_tests("."; filter = ti -> :regularization in ti.tags)'
  ```
- Quality: Aqua (`piracies=false`, `persistent_tasks=false`, `stale_deps=false`) and JET, in
  `test/test_quality.jl`.
- CI counts coverage with `--code-coverage=@src`, then reruns the `:extension` items
  (`RISTRETTO_TEST_TAGS=extension`) with `--code-coverage=@ext`. `--code-coverage=user` would
  instrument the vendored `deps/` too and makes the suite several times slower.
- The README's code blocks run in `test/test_readme.jl`; keep them runnable.

## Documentation

`docs/make.jl` builds one Documenter site. The tutorials are Literate.jl scripts in
`docs/literate/` (`# ` lines are prose, `## ` lines are code comments, `#-` splits a code block),
executed on every build into the gitignored `docs/src/tutorials/`, each also written as a
notebook to download. `RISTRETTO_DOCS_TUTORIALS=01,09` builds only those tutorials, `none` none.
The related-packages page reads the committed benchmark snapshots through
`docs/benchmark_tables.jl`.

## Formatting

Format with **Runic.jl** before committing (there is no `.runic.toml`; defaults apply):

```sh
julia --project=@runic -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src/ test/
```

## Commit messages

- First line: `<type>(<scope>): <summary>` in the imperative mood, ~72 chars, no trailing period
  (`type` = `feat`/`fix`/`refactor`/`test`/`docs`/`chore`; `scope` optional).
- Blank line, then a body wrapped at ~72 chars explaining *what* changed and *why* — bullets for
  multiple distinct changes, naming the files touched.
- Trailers: attribute the model that wrote the change as co-author.

Claude:

```
<type>(<scope>): <summary>

<body>

Co-Authored-By: Claude <Model> <noreply@anthropic.com>
```

`<Model>` is the exact model, e.g. `Opus 5`, `Sonnet 5`, `Fable 5`.

Gemini:

```
<type>(<scope>): <summary>

<body>

Co-Authored-By: Gemini <model> <gemini@localhost>
```

Replace `<model>` with the exact model name, e.g.:

```
Co-Authored-By: Gemini 3.7 Flash <gemini@localhost>
Co-Authored-By: Gemini 2.5 Pro <gemini@localhost>
```

## Known issues

| Issue | Status | Notes |
|---|---|---|
| Aqua `stale_deps` check | Disabled | Subprocess fails with EAGAIN under HPC load |
| Aqua `persistent_tasks` check | Disabled | False positive on Julia 1.12 HPC |

Verify any other suspected issue against current source before acting — this table is pruned when
items are fixed.
