# Noise and Reconstruction Analysis

*Tutorial: [Real Cartesian data](../tutorials/09_real_data_cartesian.md).*

`Ristretto` provides analysis tools for evaluating noise propagation, SNR maps, and g-factor geometry in MRI reconstructions.

## Pseudo-Replica Noise Propagation

The Monte Carlo pseudo-replica method (Robson et al. 2008) measures pixel-by-pixel noise variance and parallel imaging geometry factors ($g$-factor) for arbitrary non-linear and regularized reconstruction pipelines.

The $g$-factor is the spatially varying noise amplification of an accelerated reconstruction
beyond the $\sqrt{R}$ loss from acquiring fewer samples, $\mathrm{SNR}_{\text{acc}} = \mathrm{SNR}_{\text{full}} / (g\sqrt{R})$.
SENSE has a closed form for it (Pruessmann et al. 1999); an iterative or non-linear reconstruction
does not, so the pseudo-replica method repeats the reconstruction with independent synthetic noise
added to the data and reads the noise level off the spread of the results.

```@docs
pseudo_replica
```

### Usage

```julia
# Run 64 pseudo-replica iterations with fixed scaling
res = pseudo_replica(acq_data, method; replicas = 64, scaling = NoScaling())

mean_img = res.mean
std_img = res.std
g_factor_map = res.g_factor
```

> [!IMPORTANT]
> `pseudo_replica` requires `scaling = FixedScaling(...)` or `scaling = NoScaling()`. Data-dependent scaling (the default `QuantileScaling()`, or `BartScaling()`) rescales each noisy replica independently by its own noise quantile, distorting inter-replica variance.

## References

- Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P. (1999). *SENSE: Sensitivity encoding for fast MRI.* Magnetic Resonance in Medicine, 42(5), 952-962. <https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S> — the $g$-factor and its closed form for SENSE.
- Robson, P. M., et al. (2008). *Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions.* Magnetic Resonance in Medicine, 60(4), 895-907. <https://doi.org/10.1002/mrm.21728> — the pseudo-replica method, [`pseudo_replica`](@ref).
