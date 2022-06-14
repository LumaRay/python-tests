import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance


def linalg_norm_T(data):
    a, b = data[0]
    return numpy.linalg.norm(a - b, axis=0)

def linalg_norm_T_cp(data):
    a, b = data[1]
    return cupy.linalg.norm(a - b, axis=0)


def linalg_norm_T_fast(data, _norm=numpy.linalg.norm):
    a, b = data[0]
    return _norm(a - b, axis=0)

def linalg_norm_T_cp_fast(data, _norm=cupy.linalg.norm):
    a, b = data[1]
    return _norm(a - b, axis=0)


def setup(n, _nrr=numpy.random.rand, _crr=cupy.random.rand, _na=numpy.array, _ca=cupy.array):
    a = _nrr(n, 640)
    b = _nrr(n, 640)
    a_cp = _crr(n, 640)
    b_cp = _crr(n, 640)
    out1 = _na([a.T, b.T])
    out1_cp = _ca([a_cp.T, b_cp.T])
    return out1, out1_cp


b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(15)],
    kernels=[
        # linalg_norm_T,
        linalg_norm_T_cp,
        # linalg_norm_T_fast,
        linalg_norm_T_cp_fast,
    ],
    xlabel="len(x), len(y)",
)
b.save("norm.png")
