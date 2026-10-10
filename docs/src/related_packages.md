# Related packages

## Feature comparison

What each toolbox provides itself, checked against its source and documentation in October 2026
(BART 731bfd3, SigPy 0.1.27, Gadgetron 1d14c4c, MRpro 6830010, current MRIReco.jl and MIRT.jl).
"via X" means the feature comes from another package.

| Feature | BART | SigPy | Gadgetron | MRIReco.jl | MIRT.jl | MRpro | Ristretto |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Iterative SENSE | ✅ | ✅ | ✅ (GPU) | ✅ | ✅ (operator + solver) | ✅ | ✅ |
| g-factor / noise propagation | ❌ | ❌ | ✅ (GRAPPA g-map, pseudo-replica) | ❌ | ❌ | ❌ | ✅ (pseudo-replica) |
| GRAPPA | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| SPIRiT | ❌ | ❌ | ✅ (incl. ℓ₁-SPIRiT) | ❌ | ❌ | ❌ | ✅ |
| ESPIRiT | ✅ | ✅ | via BART | ✅ (Cartesian) | ❌ | ❌ (Walsh, Inati) | ✅ |
| JSENSE / NLINV | ✅ NLINV | ✅ JSENSE | via BART | ❌ | ❌ | ❌ | ❌ |
| Calibrationless structured low-rank | ✅ SAKE | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ SAKE, LORAKS |
| Noise prewhitening | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Coil compression | ✅ SVD, GCC, ROVir | ❌ | ✅ PCA | ✅ SVD, GCC | ✅ PCA | ✅ PCA | ✅ SVD, GCC |
| Non-Cartesian (NUFFT) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Density compensation | ❌ (given weights) | ✅ Pipe–Menon | ✅ | ✅ | radial only | ✅ Voronoi | ✅ Pipe–Menon, Voronoi |
| B₀ off-resonance correction | ✅ | ✅ time-segmented | ✅ spiral | ✅ time-segmented | building blocks | ✅ | ❌ |
| Partial Fourier | homodyne | ❌ | ✅ homodyne, POCS | ❌ | ❌ | ❌ | ✅ homodyne, POCS, phase-constrained |
| Gradient delay estimation | ✅ RING | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ RING, opposing spokes |
| Total variation | ✅ | ✅ | ✅ | ✅ | edge-preserving | ✅ | ✅ |
| Wavelets | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Low-rank, locally low-rank | ✅ (+ multi-scale) | ❌ | ❌ | ✅ (via RegularizedLeastSquares) | ❌ | ❌ | ✅ (+ multi-scale) |
| Low-rank + sparse decomposition | matrix completion | ❌ | ❌ | ❌ | example script | ❌ | ✅ any regularizers |
| TGV | ✅ (TGV, ICTGV) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Subspace / temporal basis | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| Plug-and-play priors | ✅ | ❌ | ❌ | ✅ (via RegularizedLeastSquares) | ❌ | ❌ | ✅ |
| Unrolled networks | ✅ MoDL, VarNet | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Simultaneous multi-slice | model-based | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Quantitative parameter mapping | ✅ model-based | ❌ | ✅ cardiac T₁, T₂ | ❌ | ❌ | ✅ | ❌ |
| GPU | ✅ CUDA | ✅ CuPy | ✅ CUDA | ✅ GPUArrays | ❌ | ✅ PyTorch | ✅ GPUArrays (tested on CUDA) |
| Scanner streaming (MRD) | streaming import | ❌ | ✅ server | file I/O | ❌ | file I/O | MRD read and write |
| Image export | PNG, DICOM | ❌ | DICOM, MRD | NIfTI (via ImageUtils) | AVS `.fld` | NIfTI, DICOM | NIfTI, DICOM, MRD |

PROPELLER/BLADE reconstruction is in none of them. What Ristretto lacks is on its
[roadmap](https://github.com/hakkelt/Ristretto.jl/blob/master/ROADMAP.md).

## Benchmarks

The comparison harness in `benchmark/comparison/` reconstructs the same cases with Ristretto,
BART, SigPy, MRIReco.jl, MIRT.jl and MRpro, from the same prepared data, on the same node
(see its [README](https://github.com/hakkelt/Ristretto.jl/blob/master/benchmark/comparison/README.md)).

**Method.** Each toolbox solves the same problem: the same encoding model, the same regularizer,
with λ calibrated per toolbox so that all of them reach the same accuracy, since the toolboxes
scale the data and the regularizer differently. The fair number is then **time to accuracy**: how
long each toolbox needs to bring the NRMSE against the ground truth below a common target (the
*accuracy race*). Fixed-iteration timings (*matched effort*) are listed too, but an iteration
costs and achieves different things in different solvers. Times are the fastest of three runs
after a warm-up, each planning its FFTs from scratch, from data on the host to an image on the
host.

**FFT planning.** FFTW chooses how to compute a transform when it is planned. `FFTW_ESTIMATE`
guesses from the sizes and plans in well under a millisecond; `FFTW_MEASURE` times candidate
algorithms on the machine, which costs about 0.1–0.2 s per 2D transform and 1–1.5 s per 3D one
and gives a plan that runs faster. What FFTW learns by measuring ("wisdom") can be saved and
reused by a later plan of the same transform. The toolboxes differ here:

- **BART** plans with `FFTW_ESTIMATE` by default; with `BART_USE_FFTW_WISDOM=1` it plans with
  `FFTW_MEASURE` and saves the wisdom to files, which later calls read.
- **MRIReco.jl** always plans with `FFTW_MEASURE` and keeps the wisdom only in memory, so every
  new Julia session measures again.
- **Ristretto** chooses per reconstruction: `FFTW_MEASURE` when the transforms the solve will run
  repay the measuring, else `FFTW_ESTIMATE` (`fft_planning = :auto`, the default), and saves the
  wisdom to disk for later sessions.
- **SigPy and MRpro** do not use FFTW: NumPy's and PyTorch's FFTs plan without a measuring step.
  On the GPU every toolbox uses cuFFT.

The benchmarks charge every run its planning: no wisdom survives from one run to the next, neither
in memory nor on disk. MRIReco therefore pays its measuring in every run, about 0.3 s per solve on
the 128² Shepp–Logan case, most of its time there. BART is timed both ways: the
*BART (MEASURE)* column measures in every run, the *BART* column estimates.

### Start-up and warm-up

The tables below favour the toolboxes that are slow to start. They time warm solves, while a
user also pays to start the toolbox: a BART call is a new process that starts in milliseconds, a
Python toolbox has to be imported, and a Julia toolbox also compiles each solver the first time
it is called. Measured on one case (Shepp–Logan 2D, 8 coils, Cartesian; ℓ₁-wavelet / CG-SENSE),
8 threads, in fresh processes ([`benchmark/startup/`](https://github.com/hakkelt/Ristretto.jl/tree/master/benchmark/startup)):

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.startup_markdown())
```

The Julia numbers assume precompiled packages; precompiling Ristretto itself once takes about
two minutes more. For BART every call is a fresh process, so its first and warm solves are the
same and include the process start and file I/O that the tables below leave out. The first solve
in a Julia session is therefore one to two orders of magnitude slower than the warm one, which
matters for a script that reconstructs one image and hardly at all for a session that
reconstructs many.

### Time to accuracy

Target NRMSE against the ground truth in the method column; time per toolbox, the fastest in
bold, the NRMSE reached in brackets. 8 threads, OpenBLAS.

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.comparison_markdown("openblas", 8; category = "Accuracy race"))
```

### Matched effort

ℓ₁-wavelet regularized SENSE, 20 iterations of each toolbox's default solver for the problem.

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.comparison_markdown("openblas", 8; category = "Sparsity", method_filter = m -> startswith(m, "L1-Wavelet")))
```

### Thread scaling

Ristretto's time for the matched-effort ℓ₁-wavelet reconstruction, from 1 to 16 threads.
Single-coil 2D problems are too small to gain from threads and run serially by design.

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.scaling_markdown("openblas"))
```

### GPU

The matched-effort rows on an NVIDIA GPU (CUDA), copies to and from the device included.

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.comparison_markdown("cuda", 1; category = "Sparsity", method_filter = m -> startswith(m, "L1-Wavelet")))
```

### Hardware

```@eval
using Markdown
Markdown.parse(join(("- " * Main.BenchmarkTables.hardware_markdown(b, t) for (b, t) in (("openblas", 8), ("cuda", 1))), ""))
```

### Versions

The toolbox versions behind the CPU tables (8 threads, OpenBLAS):

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.versions_markdown("openblas", 8))
```

and behind the GPU table:

```@eval
using Markdown
Markdown.parse(Main.BenchmarkTables.versions_markdown("cuda", 1))
```

## The Julia MRI ecosystem

- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) simulates MRI acquisitions (Bloch
  equations, on the GPU) and writes the raw data as MRD, which Ristretto reads. A Ristretto
  extension for KomaMRI is planned.
- [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) is the established Julia
  reconstruction framework, built on RegularizedLeastSquares.jl and NFFT.jl; it is one of the
  toolboxes benchmarked above.
- [MRIBase.jl and MRIFiles.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) hold
  MRIReco's raw-data types and file readers (ISMRMRD/MRD, Bruker, ...). Ristretto builds an
  `AcquisitionInfo` from an `MRIBase.RawAcquisitionData` and writes MRD images through MRIFiles.
- [MIRT.jl](https://github.com/JeffFessler/MIRT.jl) is the Julia version of the Michigan Image
  Reconstruction Toolbox; the tutorials display images with its `MIRTjim`.
- [GIRFReco.jl](https://github.com/BRAIN-TO/GIRFReco.jl) is a spiral reconstruction pipeline with
  gradient impulse response corrected trajectories, built on MRIReco.
- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl) post-processes
  reconstructed images: phase unwrapping (ROMEO), multi-echo coil combination, bias-field
  correction and masking.
- [NFFT.jl](https://github.com/JuliaMath/NFFT.jl) is the non-uniform FFT Ristretto uses (a bundled
  copy with GPU plans).
- [MRITestData.jl](https://github.com/hakkelt/MRITestData.jl) downloads the public raw datasets the
  tutorials and examples reconstruct, and
  [GeometricMedicalPhantoms.jl](https://github.com/hakkelt/GeometricMedicalPhantoms.jl) generates
  the phantoms.

## Reconstruction pipelines

[Gadgetron](https://github.com/gadgetron/gadgetron) is a streaming reconstruction framework that
receives MRD data from the scanner and returns images, with reconstructions written as chains of
C++ or Python gadgets. Siemens' OpenRecon runs containerized reconstructions on the scanner,
exchanging MRD as well. Ristretto does not run inside either yet; since both speak MRD, which
Ristretto reads and writes, that is a matter of a wrapper, not of the data model.
