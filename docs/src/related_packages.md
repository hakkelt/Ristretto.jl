# Related packages

## Other reconstruction toolboxes

- [BART](https://github.com/mrirecon/bart) (Berkeley Advanced Reconstruction Toolbox) is a C
  library and command-line toolbox with CUDA support. It covers parallel imaging and compressed
  sensing (`pics`), ESPIRiT and NLINV calibration, structured low-rank and subspace methods, and
  learned reconstructions (MoDL, VarNet).
- [SigPy](https://github.com/mikgroup/sigpy) is a Python package for signal processing and
  iterative reconstruction on NumPy and CuPy arrays, with an MRI submodule (`sigpy.mri`) holding
  SENSE, ℓ₁-wavelet and TV apps, ESPIRiT and JSENSE.
- [Gadgetron](https://github.com/gadgetron/gadgetron) is a streaming reconstruction framework
  that runs next to the scanner; see [Reconstruction pipelines](@ref) below.
- [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) is the established Julia
  reconstruction framework, built on RegularizedLeastSquares.jl and NFFT.jl, with GPU support
  through GPUArrays.
- [MIRT.jl](https://github.com/JeffFessler/MIRT.jl) is the Julia version of the Michigan Image
  Reconstruction Toolbox: system models, regularizers and solvers as building blocks.
- [MRpro](https://github.com/PTB-MR/mrpro) is a PyTorch-based Python package for MR image
  reconstruction and quantitative parameter mapping, on the CPU and the GPU.

Benchmarks of Ristretto against BART, SigPy, MRIReco.jl, MIRT.jl and MRpro, with a
comparison of their features, are published separately in
[RistrettoBenchmarking](https://hakkelt.github.io/RistrettoBenchmarking/). Each snapshot there
records its date and the versions it measured; the toolboxes keep changing, so treat older
numbers as a record of that date rather than as a description of the current releases.

## The Julia MRI ecosystem

- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) simulates MRI acquisitions (Bloch
  equations, on the GPU) and writes the raw data as MRD, which Ristretto reads. A Ristretto
  extension for KomaMRI is planned.
- [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) is the established Julia
  reconstruction framework, built on RegularizedLeastSquares.jl and NFFT.jl; it is one of the
  toolboxes described above.
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
