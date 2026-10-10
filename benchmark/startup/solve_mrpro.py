"""MRpro's row of the startup benchmark: the operators and solvers `_mrpro_solve` composes in the
comparison harness (benchmark/comparison/scripts/_toolkits.jl): masking ∘ FFT ∘ sensitivities,
`cg` for CG-SENSE, `pgd` on the synthesis form with a `db2` `WaveletOp` and the step from a power
method inside the timed region for L1-wavelet; PyTorch at the thread count under test, one
inter-op thread."""
import time
from child_common import LAMBDA, MAXIT, METHOD, NC, NTHREADS, NX, NY, WAVELET_LEVELS, WAVELET_NAME, load_case, measure

t0 = time.perf_counter()
import torch
import mrpro
from mrpro.operators import CartesianMaskingOp, FastFourierOp, SensitivityOp, WaveletOp
from mrpro.operators.functionals import L1Norm, L2NormSquared
from mrpro.algorithms.optimizers import pgd, cg
t_import = time.perf_counter() - t0
import numpy as np

t0 = time.perf_counter()
torch.set_num_threads(NTHREADS)
torch.set_num_interop_threads(1)
ksp, mps, ref, _ = load_case(np)
y = torch.from_numpy(np.ascontiguousarray(ksp.reshape(1, NC, 1, NY, NX)))
csm = torch.from_numpy(np.ascontiguousarray(mps.reshape(1, NC, 1, NY, NX)))
mask = (y.abs().sum(1, keepdim=True) > 0).to(torch.float32)
if METHOD not in ("wavelet", "cgsense"):
    raise ValueError(METHOD)
t_setup = time.perf_counter() - t0


def solve():
    A = CartesianMaskingOp(mask) @ FastFourierOp(dim=(-2, -1)) @ SensitivityOp(csm)
    (b,) = A.H(y)
    if METHOD == "cgsense":
        x = cg(A.gram, b, max_iterations=MAXIT, tolerance=0.0)[0]
    else:
        torch.manual_seed(0)
        L2 = 1.05 * float(A.operator_norm(torch.randn_like(b), dim=None, max_iterations=30)) ** 2
        W = WaveletOp(domain_shape=tuple(b.shape[-2:]), dim=(-2, -1), wavelet_name=WAVELET_NAME, level=WAVELET_LEVELS)
        (z0,) = W(torch.zeros_like(b))
        (z,) = pgd(f=L2NormSquared(target=y) @ A @ W.H, g=L1Norm(weight=LAMBDA), initial_value=z0,
                   stepsize=0.5 / L2, max_iterations=MAXIT)
        (x,) = W.H(z)
    return x.cpu().numpy()[0, 0, 0]


measure(np, solve, t_import, t_setup, ref)
