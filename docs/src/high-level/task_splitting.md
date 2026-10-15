# Task Splitting

When the data have batch dimensions that nothing couples (slices, contrasts, echoes, or time when
there is no temporal regularizer), `reconstruct` splits the problem into one independent solve
per batch element and runs them in parallel. Nothing has to be requested.
[Algorithms and configuration](../tutorials/06_algorithms_and_configuration.md) measures what it
buys.

!!! note "Not to be confused with image decomposition"
    Task splitting splits a *single-image* reconstruction over independent batch dimensions. It is unrelated to [Image Decomposition](image_decomposition.md), which models the reconstructed image itself as a sum of additive components (e.g. low-rank + sparse). The two can be combined — see [Interaction with Task Splitting](image_decomposition.md#image-decomposition-task-splitting).

## Which Dimensions Are Split

Batch dimensions are the image dimensions that are neither Fourier-encoded (`:x`, `:y`, and `:z`
in 3D) nor the coil dimension: the slices of a 2D multi-slice acquisition, `:time`, `:contrast`,
`:echo`, and so on. Each term in the model reports the dimensions it couples through
`get_affected_dims`, and whatever batch dimension no term couples is split:

| Term | Couples |
|---|---|
| `L1Image`, `NonNegative`, `L0Image(; threshold)` | nothing |
| `L1Wavelet2D`, `TotalVariation2D` and the other 2D terms | `:x`, `:y` only, so every batch dimension is split |
| `L1Wavelet3D`, `TotalVariation3D` on multi-slice 2D data | the slice dimension too |
| `L1TemporalFourier`, `TemporalTotalVariation` | `:time` |
| `LowRank`, `RankLimit`, `LocallyLowRank`, `MultiScaleLowRank` | every dimension up to and including `time_dim`; dimensions after it are still split |
| `L0Image(; count)` | everything: a budget of non-zeros is a property of the whole array |

A model that couples `:time` can still be split over the slices of a dynamic multi-slice
acquisition. A custom regularizer joins this mechanism by implementing `get_affected_dims`
(see [Adding a regularizer](../tutorials/12_low_level_interface.md)).

## Controls

| Keyword | Effect |
|---|---|
| `disable_task_splitting = true` | solve the whole stack as one problem |
| `task_executor = MultiThreadingExecutor()` | run the tasks on Julia's threads |
| `task_executor = SequentialExecutor()` | run the tasks one after another, each free to thread internally |

Left unset, `task_executor` runs the tasks on threads when a task is too small to thread by itself
or there are at least as many tasks as threads, and one after another otherwise. Forcing
`SequentialExecutor()` on small tasks leaves the threads idle: a 128² slice is below every
operator's threading threshold, so on 16 threads 32 such slices took 7.3 s one after another,
3.0 s unsplit and 0.73 s in parallel.

Both are run settings: they go to `reconstruct` or to a `ReconstructionConfig`. Data in GPU
memory are not split unless `disable_task_splitting = false` is given explicitly (see
[GPU Reconstruction](gpu.md#gpu-task-splitting)).

## Split and Unsplit Results Differ Slightly

A split run tests the stopping criterion per task, so each slice stops at its own iteration count.
An unsplit run tests one criterion on the whole stack, so every slice runs as many iterations as
the slowest one needs. Both are converged to the requested tolerance, but the images are not
bit-identical unless `reltol = 0` fixes the iteration count.
