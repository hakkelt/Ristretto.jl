# Theoretical Background

## MRI Forward Model

The MRI forward model describes how an image transforms into observed k-space data:

```math
y = \mathcal{P} \mathcal{F} \mathcal{S} x + n = \mathcal{A} x + n
```

Where:
- ``x \in \mathbb{C}^{N_x \times N_y \times [N_z]}`` is the image to be reconstructed
- ``\mathcal{S}`` represents coil sensitivity weighting operator
- ``\mathcal{F}`` is the Fourier transform operator
- ``\mathcal{P}`` is the sampling / data-consistency operator
- ``\mathcal{A} = \mathcal{P} \mathcal{F} \mathcal{S}`` is the complete encoding operator
- ``y`` is the observed k-space data
- ``n`` is measurement noise

This is the SENSE model (Pruessmann, Weiger, Scheidegger & Boesiger, *SENSE: Sensitivity encoding
for fast MRI*, Magnetic Resonance in Medicine 42(5), 952-962, 1999,
<https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S>); solving it
iteratively, as `IterativeReconstruction` does, is CG-SENSE when no regularizer is added.
Noise correlated across coils is assumed to have been whitened first (see [Pre-processing](high-level/preprocessing.md)).

These operators are all linear maps, and they are usually represented as complex matrices in the literature. Even though, in practice, we implement them as efficient computational operators without explicitly forming large matrices, it gives theoretical background for defining adjoint operations (complex conjugate of transpose for matrices) that are essential for iterative reconstruction algorithms.

### Operator Components

#### 1. Sensitivity Map Operator (S)

Models the spatial sensitivity of receiver coils in parallel imaging:

```math
S: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{N_x \times N_y \times [N_z] \times N_c}
```

**Forward operation**: ``(Sx)_c = s_c \odot x`` (element-wise multiplication)

**Adjoint operation**: ``S^H y = \sum_{c=1}^{N_c} \bar{s}_c \odot y_c`` (coil combination)

Where ``s_c`` is the sensitivity map for coil ``c``, ``\odot`` denotes element-wise multiplication, ``N_c`` is the number of coils, and ``\bar{s}_c`` is the complex conjugate of ``s_c``.

#### 2. Fourier Transform Operator (ℱ)

Transforms between image and k-space:

```math
\mathcal{F}: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{N_x \times N_y \times [N_z]}
```

**Forward operation**: Discrete Fourier Transform (DFT)
**Adjoint operation**: Inverse DFT (scaled appropriately)

#### 3. Subsampling Operator (𝒫)

Selects observed k-space locations according to an undersampling pattern:

```math
\mathcal{P}: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{|\Omega|}
```

Where ``\Omega`` is the set of sampled k-space locations.

## Reconstruction as Inverse Problem

The most simple way of reconstructing the image ``x`` from observed data ``y`` is to apply the adjoint of the encoding operator:

```math
\mathcal{A}^H y = \mathcal{S}^H \mathcal{F}^H \mathcal{P}^H y
```

For fully sampled data without noise, this gives the least-squares solution. However, in practice, data is often undersampled and noisy, making direct inversion ill-posed. To address this, we formulate the reconstruction as a regularized inverse problem, formulated as:

```math
\hat{x} = \arg\min_x \frac{1}{2}\|\mathcal{A}x - y\|_2^2 + \lambda R(x)
```

Where:
- ``\frac{1}{2}\|Ex - y\|_2^2`` is the data fidelity term
- ``R(x)`` is a regularization term (e.g., sparsity, total variation)
- ``\lambda`` controls the regularization strength

The encoding operator ``E`` makes this optimization problem well-defined and computationally tractable.

### Regularization Techniques

The regularization term ``R(x)`` can take various forms depending on the desired image properties:
- **L2 Regularization**: ``R(x) = \|x\|_2^2`` -> Promotes smoothness
- **L1 Wavelet Regularization**: ``R(x) = \|\Psi x\|_1`` where ``\Psi`` is a wavelet transform -> Promotes sparsity in the wavelet domain
- **Total Variation (TV)**: ``R(x) = \|\nabla x\|_1`` -> Preserves edges while reducing noise

The regularization terms are represented by a (linear) operator and a norm function in general:
```math
R(x) = f(\mathcal{T} x)
```
where ``\mathcal{T}`` is some transform operator and ``f`` is a norm function (e.g., L1, L2, nuclear norm, etc.).

### Additive Image Decomposition

Some regularization strategies do not fit a single image well but do fit a *sum* of images with different properties — the canonical case being low-rank + sparse (L+S) decomposition of dynamic MRI. Instead of one variable `x`, the image is modeled as a sum of components `x = x_1 + x_2 + \dots + x_n`, each with its own regularizer:

```math
\hat{x}_1, \dots, \hat{x}_n = \arg\min_{x_1, \dots, x_n} \frac{1}{2}\left\|E\left(\sum_{i=1}^n x_i\right) - y\right\|_2^2 + \sum_{i=1}^n R_i(x_i)
```

The reconstructed image is ``\hat{x} = \sum_i \hat{x}_i``; the individual ``\hat{x}_i`` remain available as well (e.g. the low-rank background and the sparse dynamic foreground). See [Image Decomposition](high-level/image_decomposition.md) for the `Component`/`ReconImage` API implementing this model — not to be confused with [Task Splitting](high-level/task_splitting.md), which splits a *single-image* problem over independent batch dimensions rather than into additive components.

### Optimization Algorithms

Various iterative algorithms can be employed to solve the regularized inverse problem, most commonly:
- **Gradient Descent** (usually conjugate gradient): If ``R(x)`` is differentiable
- **Proximal Gradient Methods**: For non-differentiable ``R(x)``

Proximal gradient methods alternate between gradient descent on the data fidelity term and applying the proximal operator of the regularization term. The proximal operator is defined as:
```math
\text{prox}_{\alpha R}(v) = \arg\min_x \frac{1}{2}\|x - v\|_2^2 + \alpha R(x)
```
While it is inefficient to solve this minimization directly, many common regularization terms have closed-form proximal operators. For example, the proximal operator for L1 regularization is soft-thresholding:
```math
\text{prox}_{\alpha \|\cdot\|_1}(v) = \text{sign}(v) \odot \max(|v| - \alpha, 0)
```
