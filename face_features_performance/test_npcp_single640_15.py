import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

def linalg_norm(data):
    a, b = data[0]
    return numpy.linalg.norm(a - b, axis=1)

def linalg_norm_cp(data):
    a, b = data[2]
    return cupy.linalg.norm(a - b, axis=1)


def linalg_norm_T(data):
    a, b = data[1]
    return numpy.linalg.norm(a - b, axis=0)

def linalg_norm_T_cp(data):
    a, b = data[3]
    return cupy.linalg.norm(a - b, axis=0)


def sqrt_sum(data):
    a, b = data[0]
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=1))

def sqrt_sum_cp(data):
    a, b = data[2]
    return cupy.sqrt(cupy.sum((a - b) ** 2, axis=1))


def sqrt_sum_T(data):
    a, b = data[1]
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=0))

def sqrt_sum_T_cp(data):
    a, b = data[3]
    return cupy.sqrt(cupy.sum((a - b) ** 2, axis=0))


def sqrt_einsum(data):
    a, b = data[0]
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))

def sqrt_einsum_cp(data):
    a, b = data[2]
    a_min_b = a - b
    return cupy.sqrt(cupy.einsum("ij,ij->i", a_min_b, a_min_b))


def sqrt_einsum_T(data):
    a, b = data[1]
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum("ij,ij->j", a_min_b, a_min_b))

def sqrt_einsum_T_cp(data):
    a, b = data[3]
    a_min_b = a - b
    return cupy.sqrt(cupy.einsum("ij,ij->j", a_min_b, a_min_b))


def setup(n):
    a = numpy.random.rand(n, 640)
    b = numpy.random.rand(n, 640)
    a_cp = cupy.random.rand(n, 640)
    b_cp = cupy.random.rand(n, 640)
    out0 = numpy.array([a, b])
    out1 = numpy.array([a.T, b.T])
    out0_cp = cupy.array([a_cp, b_cp])
    out1_cp = cupy.array([a_cp.T, b_cp.T])
    return out0, out1, out0_cp, out1_cp


b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(15)],
    kernels=[
        linalg_norm,
        linalg_norm_cp,
        linalg_norm_T,
        linalg_norm_T_cp,
        sqrt_sum,
        sqrt_sum_cp,
        sqrt_sum_T,
        sqrt_sum_T_cp,
        sqrt_einsum,
        sqrt_einsum_cp,
        sqrt_einsum_T,
        sqrt_einsum_T_cp,
    ],
    xlabel="len(x), len(y)",
    equality_check=None,  # cupy.allclose,
)
b.save("norm.png")
