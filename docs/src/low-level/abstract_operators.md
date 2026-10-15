# [AbstractOperators.jl: Matrix-Free Linear Operators](@id abstract_operators)

!!! note "Bundled package"
    Ristretto ships its own copy of AbstractOperators.jl (with FFTWOperators, NFFTOperators,
    WaveletOperators and DSPOperators). Import them through Ristretto, as `using
    Ristretto.AbstractOperators`, and do not `Pkg.add` the registered packages: the bundled
    version carries work that is not registered yet (multithreading, GPU support, new operators,
    functions and algorithms). Upstreaming it is under way, and the bundled copy goes away once
    the registered releases have it. Documentation of the bundled version:
    [AbstractOperators.jl (fork)](https://hakkelt.github.io/AbstractOperators.jl/dev/).

Every operator Ristretto builds is an `AbstractOperator`: a matrix-free linear (or nonlinear)
map that works on multi-dimensional arrays without flattening them, supports `*`, `'`, `+`,
scaling and composition like a matrix, and stores only its parameters. The encoding operator,
for instance, is a composition of a sensitivity map operator, a DFT and a `GetIndex`:

```julia
using Ristretto.AbstractOperators, Ristretto.FFTWOperators

ℱ = DFT(ComplexF32, (64, 64))
𝒫 = GetIndex(zeros(ComplexF32, 64, 64), (mask,))
𝒜 = 𝒫 * ℱ * DiagOp(smap)       # forward: 𝒜 * x, adjoint: 𝒜' * y
```

[The low-level interface](../tutorials/12_low_level_interface.md) takes these operators apart,
checks their adjoints, and writes a new operator from scratch (§7). The fork's documentation
covers [writing an operator](https://hakkelt.github.io/AbstractOperators.jl/dev/custom/) and the
[operator properties](https://hakkelt.github.io/AbstractOperators.jl/dev/properties/) solvers read.

## Operators in AbstractOperators.jl

The operators most useful in MRI reconstruction:

### Transform Operators
- **`DFT`**: Discrete Fourier Transform (via FFTWOperators.jl)
- **`DCT`**: Discrete Cosine Transform (via FFTWOperators.jl)
- **`RDFT`**: Real-to-complex FFT (via FFTWOperators.jl)
- **`WaveletOp`**: Wavelet transforms (via WaveletOperators.jl)
- **`NFFTOp`**: Non-uniform FFT (via NFFTOperators.jl)

### Linear Operators
- **`Eye`**: Identity operator
- **`DiagOp`**: Diagonal operator (element-wise multiplication)
- **`MatrixOp`**: Wraps a regular matrix
- **`FiniteDiff`**: Finite difference (gradients)
- **`Variation`**: Total variation operator
- **`GetIndex`**: Subsampling/indexing operator
- **`ZeroPad`**: Zero-padding operator

### Calculus
- **`Scale`**: Scalar multiplication
- **`Compose`**: Operator composition
- **`Sum`**: Sum outputs of different operators applying to the same input
- **`BroadCast`**: Broadcasting operator -- repeats the output of an operator across specified dimensions
- **`Reshape`**: Reshape arrays

### Nonlinear Operators
- **`Sigmoid`**, **`Tanh`**, **`Exp`**, **`Sin`**, **`Cos`**: Activation functions
- **`SoftMax`**, **`SoftPlus`**: Softmax and softplus

### Signal Processing
- **`Conv`**: Convolution (via DSPOperators.jl)
- **`Filt`**: Filtering (via DSPOperators.jl)
- **`Xcorr`**: Cross-correlation (via DSPOperators.jl)
