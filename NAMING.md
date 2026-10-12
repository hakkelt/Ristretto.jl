# NAMING.md — naming and API-surface conventions

Rules for choosing names and deciding what the package exposes. Derived from a review of BART,
RegularizedLeastSquares.jl (MRIReco), SigPy and Fessler's MIRT; the sources are cited where a rule
rests on them.

The package is **not yet released**. Renames are hard breaks: delete the old name, do not add a
deprecation shim. The only surviving second names are the aliases §2 explicitly allows.

---

## 1. Names come from the literature, not from the implementation

**Rule 1.1 — If the field has a standard name, use it.** Prefer the name the reader will already
know from BART, SigPy, RegularizedLeastSquares.jl or the originating paper over a name that
describes the code. `EdgePreservingRoughness2D` stays because "edge-preserving roughness penalty"
is Fessler's own term, even though the implementation is a Huber loss on a gradient.

Reference vocabulary, for checking a candidate name against prior art:

| Source | Names |
|---|---|
| BART `-R` (`src/grecon/optreg.c`) | `Q` l2-norm image domain, `I` l1-norm image domain, `W` l1-wavelet, `F` l1-Fourier, `T` total variation, `G` total generalized variation, `C`/`V` infimal convolution TV/TGV, `L` locally low rank, `M` multi-scale low rank, `N`/`H` NIHT image/wavelet, `POS` |
| RegularizedLeastSquares.jl | `L1Regularization`, `L2Regularization`, `L21Regularization`, `TVRegularization`, `LLRRegularization`, `NuclearRegularization`, `PositiveRegularization`, `RealRegularization`, `ProjectionRegularization`, `TransformedRegularization`, `PlugAndPlayRegularization` |
| SigPy | `L1Reg`, `L2Reg`, `L1Proj`, `L2Proj`, `LInfProj`, `BoxConstraint`, `L1WaveletRecon`, `TotalVariationRecon`, `EspiritCalib` |
| Fessler MIRT | "edge-preserving roughness penalty", potential functions (`huber`) |

**Rule 1.2 — A type name must state the penalty, not only the transform.** A transform alone is
ambiguous: `TemporalFourier` could be an ℓ₁ sparsity term or an ℓ₂ smoothness term. This is why the
name is `L1TemporalFourier` (matching BART's "l1-Fourier") and why the identity-operator pair is
`L1Image` / `L2Image` rather than `L1Image` / `Tikhonov`.

The rule is satisfied when the *penalty* is legible, by whatever means:

- by an explicit norm prefix — `L1Image`, `L2Image`, `L1Wavelet2D`, `L1Contourlet`,
  `L1TemporalFourier`, `L0Image`, `L0Wavelet2D`, `L0Wavelet3D`;
- by a penalty name that is itself standard — `TotalVariation2D`, `TotalGeneralizedVariation2D`,
  `JointSparsity`, `LowRank`, `LocallyLowRank`, `MultiScaleLowRank`, `StructuredLowRank`,
  `EdgePreservingRoughness2D`, `ReferencePrior`, `PlugAndPlay`.

A name that states only a transform (`Wavelet2D`, `TemporalFourier`) fails the rule. A qualifier in
front of a standard penalty name is fine as long as the penalty survives: `LocallyLowRank`,
`MultiScaleLowRank` and `StructuredLowRank` all say *low rank*, and only differ in what is made
low-rank.

**Rule 1.1a — Prefer the family term over a single paper's acronym when one type covers several
papers.** `StructuredLowRank` is named for "structured low-rank matrix completion", which is the
field's own umbrella term (it is the title phrase of Shin et al., MRM 2014) and which covers both
forms the type provides: `max_rank` is SAKE's hard rank constraint and `λ` is LORAKS' C-matrix
nuclear-norm penalty. Naming the type `SAKE` or `LORAKS` would break Rule 1.4's spirit rather than
follow it: those acronyms name whole reconstruction pipelines, so a type called `LORAKS` would
overclaim, and a type per paper would duplicate one operator and one prox for no gain. The same
reasoning governs the `structure` keyword: LORAKS is a *family* of matrix constructions, and its
C, S and G matrices are `structure = :c`, `:s` and `:g` of the one type rather than three types,
because they differ only in the lift. The papers are cited in the docstring and in
`docs/src/high-level/regularization.md`; the keywords tell the reader which one they are running.

**Rule 1.3 — Constraints read as constraints.** A hard constraint (an indicator function) is named
for the set, not for a norm: `RankLimit`, `NonNegative`, `BoxConstraint`, `HardConsistency`. Where a
penalty and a constraint express the same idea via two otherwise-identical types, pair them as
separate names: `LowRank` / `RankLimit`. Where the penalty and the constraint are two proximal
operators on the *same* sparsifying transform (nothing else differs — same operator, same
`get_affected_dims`/`scale_regularization` shape aside from the constraint's scale-invariance), fold
them into one type taking `threshold` (penalty) XOR `count` (constraint) as mutually-exclusive
keywords instead of minting a second type name: `L0Image(; threshold, count)`,
`L0Wavelet2D(; threshold, count)`, `L0Wavelet3D(; threshold, count)` (there is no separate
`SparsityLimit`). `StructuredLowRank(; λ, max_rank)` follows the same pattern: the lift, the
`get_affected_dims` shape and the prox pipeline are identical, and only the singular-value step
differs (soft-threshold vs truncate), so it is one type with two mutually-exclusive keywords rather
than a `StructuredLowRank` / `StructuredRankLimit` pair.

**Rule 1.4 — Acronyms are kept when they are the field's own.** `GRAPPA`, `SPIRiT`, `ESPIRiT`,
`POCS`, `RING`, `SVDCompression`, `GeometricCompression`, `PipeMenonDCF`, `VoronoiDCF`,
`RootSumSquares`, `AdaptiveCombine`. Do not spell these out and do not invent new ones.

## 2. Aliases

**Rule 2.1** — A `const Alias = CanonicalName` is allowed only when a *second* name is genuinely
established in the field and a reader is likely to reach for it first. It must be exported
alongside the canonical name and documented in the same docstring.

Approved aliases:

```julia
const Tikhonov = L2Image        # the textbook name for the ℓ₂ image-domain term
const LLR      = LocallyLowRank # BART, RegularizedLeastSquares.jl, and the literature
const PDHG     = ChambollePock  # primal-dual hybrid gradient: SigPy, MRpro, the literature
```

**Rule 2.2** — Do not add an alias merely to shorten a name, and never add one that cannot name a
single type (`TV` cannot stand for both `TotalVariation2D` and `TotalVariation3D`).

## 3. Dimensionality

**Rule 3.1** — Spatial dimensionality is expressed by a `2D` / `3D` type suffix
(`TotalVariation2D/3D`, `SecondOrderTotalVariation2D/3D`, `EdgePreservingRoughness2D/3D`,
`L1Wavelet2D/3D`, `L0Wavelet2D/3D`, `TotalGeneralizedVariation2D/3D`) **or** by a constructor field
(`PlugAndPlay(spatial_dims)`). Within one family, pick one and do not mix.

**Rule 3.2** — A family that offers a 2D member should offer the 3D member too. A missing twin is a
gap, not a design decision; if it is deliberate, say why in the docstring.

**Rule 3.3** — A dimension selected at runtime (a time axis, a coil axis) is a field, never a type
suffix: `time_dim`, `dim`. `TemporalTotalVariation` needs no suffix because the temporal axis is
one-dimensional by construction.

## 4. Abstract types

**Rule 4.1 — Bare noun, no `Abstract` prefix and no `Method` suffix.** The prefix is Julia
convention only when a concrete type would otherwise claim the name, which is not the case here.
Target: `ReconstructionMethod`, `IterativeMethod`, `DirectMethod`, `Regularization`, `Scaling`,
`CoilCombination`, `DataFidelity`, `Verbosity`, `Subsampling`, `PartialFourierFilter`,
`DensityCompensation`, `CoilCompression`, `SensitivityEstimation`, `GradientDelay`.

**Rule 4.2 — The supertype's stem must match its subtypes'.** `Scaling` over `NoScaling` /
`BartScaling` / `FixedScaling` / `MeasurementBasedScaling`, not `Normalization`. When adding a
subtype whose name does not share the family stem, either rename the subtype or reconsider the
supertype.

**Rule 4.3 — The `No*` prefix marks the null member of a family**: `NoScaling`, `NoFidelity`,
`NoCoilCombination`.

## 5. Function names

**Rule 5.1 — Verb phrases for actions, specific enough to survive `using`.** `reconstruct`,
`build_model`, `estimate_sensitivities`, `normalize_sensitivity_maps`, `compress_coils`,
`correct_gradient_delays`,
`simulate_acquisition`, `create_sampling_pattern`. A bare generic verb (`calculate`, `lower`,
`total`) is acceptable only for a name that is not exported.

**Rule 5.2 — `get_*` is reserved for operator construction**: `get_operator`,
`get_encoding_operator`, `get_fourier_operator`, `get_sensitivity_map_operator`,
`get_subsampling_operator`. Do not use `get_` for a plain field access.

## 6. What is exported, what is `public`, what is neither

The exported set targets a **non-expert user**: someone assembling a reconstruction from the
built-in pieces. Anything needed only to *extend* the package is `public` but not exported.
`public` requires `julia = "1.11"` in `Project.toml`.

**Rule 6.1 — Export** concrete types the user constructs and verbs the user calls:

- every concrete `Regularization`;
- every concrete reconstruction method (`DirectReconstruction`, `IterativeReconstruction`, `GRAPPA`,
  `SPIRiT`, `Homodyne`, `PhaseConstrained`, `POCS`);
- algorithm aliases (`ISTA`, `FISTA`, `POGM`, `ADMM`, `DouglasRachford`, `CG`, `CGNR`, `ChambollePock`,
  `PDHG`, `VuCondat`, `NCG`, `LBFGS`);
- the concrete members of each configuration family — coil combination, data fidelity, verbosity,
  scaling, executors, partial-Fourier filters, sampling patterns, preprocessing methods;
- `AcquisitionInfo` (its two concrete subtypes are `public`, see Rule 6.2);
- the top-level verbs, `ReconstructionConfig`, `Component`, `ReconImage`, `Header`, `components`,
  `total_image`, `drop_components`, `TemporalBasis`, `KSpaceToImage` and `pseudo_replica`. Export
  to a file goes through FileIO's `save`, whose methods come from package extensions.

`AcquisitionInfo` is exported despite being abstract because it is also a constructor: it dispatches
to `CartesianAcquisitionInfo` or `NonCartesianAcquisitionInfo`. That is the *only* justification for
exporting an abstract type — an abstract type with no constructor methods is `public` at most. The
two concrete subtypes are *not* exported, in either direction: `AcquisitionInfo(...)` reaches both,
so a user never has to name one, and naming one is a statement about dispatch — the extension
surface, which Rule 6.2 governs.

**Rule 6.2 — `public`** for the extension surface: everything a third party must dispatch on,
subtype, or implement.

- every abstract supertype without a constructor (Rule 4.1's list);
- `CartesianAcquisitionInfo` and `NonCartesianAcquisitionInfo`, the two concrete acquisition types
  `AcquisitionInfo(...)` dispatches to;
- the regularizer interface: `get_operator`, `materialize`, `materialize_with_auxiliaries`,
  `materialize_all`, `get_affected_dims`, `scale_regularization`, `bind_dimensions`, `calculate`;
- the method interface hook a new method may override: `check_applicable`;
- low-level operator construction, documented under `docs/src/low-level/`:
  `get_encoding_operator`, `get_fourier_operator`, `get_sensitivity_map_operator`,
  `get_subsampling_operator`, `build_encoding_operator`, `signal_model_operator`, `NamedDimsOp`,
  `DFT`, `DEFAULT_ALGORITHMS`.

Every `public` name carries a docstring and appears in the API reference. `public` is a promise of
stability, so do not mark something public merely because it happens to be useful internally.

**Rule 6.3 — Neither exported nor public** when nothing outside the package can meaningfully
override or call it. Test that by asking: *is there a caller this package does not own?* Internal
by this test: `lower`, `variable_dims`, `variable_size`, `output_dims` (overridden only on the
closed signal-model axis, which is not an open extension point), and every proximal-operator
implementation detail (`hard_consistency_prox`, `DenoiserProx`, `BlockNuclearNorm`,
`HankelLowRankProx`, `SPIRiTConsistencyOp`, `StackedNSCTOp`).

**Rule 6.4 — Do not blanket-reexport a dependency.** `@reexport using SomePackage` drops that
package's whole namespace on the user and causes real collisions (an `@reexport using
AbstractOperators` plus `ProximalOperators` clash on `Sum` was the concrete case here). Reexport
only the individual names a non-expert must type:

```julia
export NamedDimsArray, dimnames, unname   # build the input array
export WT, wavelet                        # L1Wavelet2D(λ; wavelet = WT.db4)
export ContourletParams, parabolic_levels # L1Contourlet
```

**Rule 6.5 — Avoid names that collide on `using`.** A short generic noun in the exported set is a
liability for exactly the beginners the set targets: `Config` breaks any script that also defines
one, `total` is five generic letters. Prefer `ReconstructionConfig`, `total_image`. This applies to
the exported set only — an internal or `public` name may be short and generic.

## 7. Proximal operators belong upstream

**Rule 7.1** — Before writing a `ProximalCore.prox!` in this package, check
`deps/ProximalOperators/src/functions/` and `.../calculus/` for an existing function. Ristretto already
reuses `NormL1`, `NormL0`, `NormL21`, `NuclearNorm`, `SqrNormL2`, `IndBox`, `IndNonnegative`,
`IndBallL0`, `IndBallRank`, `SeparableHuberLoss`, `Translate` and `SlicedSeparableSum`.

**Rule 7.2** — A new proximal function with **no MRI-specific content** goes into the
`ProximalOperators` fork, not into `src/regularization/`. Precedents: `SeparableHuberLoss` was added
there in commit `274b63a`; `ProximalAverage` (the proximal-average calculus rule behind
`MultiScaleLowRank`) and `IndAffineCG` (the matrix-free affine projection behind `HardConsistency`)
followed. A proximal function that only makes sense given an image layout
(spatial dims, frames, coils) stays here; where possible, split it into a generic core upstream and
a thin layout wrapper in Ristretto. `BlockNuclearNorm` is the case that stays: it is defined against the
`(spatial…, frames, batch)` image layout throughout, so there is no generic core to lift out;
`HankelLowRankProx` (behind `StructuredLowRank`) is the same case for the
`(k-space grid…, channels, batch)` layout, and `LoraksLowRankProx` doubly so: the LORAKS S- and
G-matrix lifts it wraps reflect k-space about DC, which is MRI content, and they are only
real-linear, so they are not `LinearOperator`s that rule 7.3 could send to `AbstractOperators`
either.
`hard_consistency_prox` is the case that split cleanly: the CG projection went upstream, and only
the `is_AAc_diagonal`/`diag_AAc` shortcut — which is knowledge about MRI encoding operators — stayed.

**Rule 7.3** — A **linear operator** never belongs in `ProximalOperators`. It goes to
`AbstractOperators` (or the relevant `*Operators` fork) so that both packages and any third party
can use it. Precedent: the block-Hankel lift used by `StructuredLowRank` went upstream as
`AbstractOperators.Hankel` — it is the generic sliding-window embedding, with no MRI content —
while the MRI-layout Cadzow prox around it stayed in Ristretto.
