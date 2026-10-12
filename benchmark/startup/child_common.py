"""Shared by the Python toolkits' solve scripts (`solve_<toolkit>.py`); the counterpart of
`child_common.jl`, standard library only, so importing it loads nothing the toolkit would not.

    python solve_<toolkit>.py <datadir> <method> <nwarm> <nx> <ny> <nc> <maxit> <lambda>

Prints `RESULT <import s> <setup s> <first s> <nrmse> <warm s,...>` for `run.jl`.
"""
import os
import sys
import time

DATADIR = sys.argv[1]
METHOD = sys.argv[2]
NWARM = int(sys.argv[3])
NX, NY, NC = (int(a) for a in sys.argv[4:7])
MAXIT = int(sys.argv[7])
LAMBDA = float(sys.argv[8])
NTHREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
WAVELET_NAME = "db2"
WAVELET_LEVELS = 3


def load_case(np):
    """The case's arrays in the reversed-axes (C-order) layout SigPy and MRpro take: k-space and
    maps `(coil, y, x)`, reference and mask `(y, x)`."""
    def raw(name, dtype, shape):
        return np.fromfile(os.path.join(DATADIR, name), dtype=dtype).reshape(shape[::-1])
    return (raw("kspace.c64", np.complex64, (NX, NY, NC)), raw("smaps.c64", np.complex64, (NX, NY, NC)),
            raw("reference.c64", np.complex64, (NX, NY)), raw("mask.u8", np.uint8, (NX, NY)) != 0)


def mag_nrmse(np, est, ref):
    a, r = np.abs(est), np.abs(ref)
    a = a * np.linalg.norm(r) / np.linalg.norm(a)
    return float(np.linalg.norm(a - r) / np.linalg.norm(r))


def measure(np, solve, t_import, t_setup, reference):
    t0 = time.perf_counter()
    x = solve()
    t_first = time.perf_counter() - t0
    e = mag_nrmse(np, x, reference)
    warm = []
    for _ in range(NWARM):
        t0 = time.perf_counter()
        solve()
        warm.append(time.perf_counter() - t0)
    print("RESULT", t_import, t_setup, t_first, e, ",".join(map(str, warm)) if warm else "-", flush=True)
