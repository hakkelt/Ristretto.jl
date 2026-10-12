"""SigPy's row of the startup benchmark: `SenseRecon` / `L1WaveletRecon` with the parameters
`sigpy_recon` passes in the comparison harness (benchmark/comparison/scripts/_toolkits.jl): `db2`,
zero tolerance, on the CPU."""
import time
from child_common import LAMBDA, MAXIT, METHOD, WAVELET_NAME, load_case, measure

t0 = time.perf_counter()
import sigpy
import sigpy.mri.app as sp_app
t_import = time.perf_counter() - t0
import numpy as np

t0 = time.perf_counter()
ksp, mps, ref, _ = load_case(np)
if METHOD == "wavelet":
    def solve():
        return sp_app.L1WaveletRecon(ksp, mps, LAMBDA, device=sigpy.cpu_device, wave_name=WAVELET_NAME,
                                     max_iter=MAXIT, tol=0.0, show_pbar=False).run()
elif METHOD == "cgsense":
    def solve():
        return sp_app.SenseRecon(ksp, mps, device=sigpy.cpu_device, max_iter=MAXIT, tol=0.0, show_pbar=False).run()
else:
    raise ValueError(METHOD)
t_setup = time.perf_counter() - t0
measure(np, solve, t_import, t_setup, ref)
