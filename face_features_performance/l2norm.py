import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

@jit(nopython=True, fastmath=True)
def l2_python_distance_multi_T_jit_fastmath(data):
    for x_idx, x in enumerate(data[1]):
        summ = 0
        for col in range(len(x)):
            summ += x[col] ** 2
        data[1][x_idx] = x / (summ ** 0.5)
    return data[1]

@jit(nopython=True)
def l2_sqrt_sum_multi_jit(data):
    for x_idx, x in enumerate(data[0]):
        data[0][x_idx] = x / numpy.sqrt(numpy.sum(numpy.multiply(x, x)))
    return data[0]

@jit(nopython=True, fastmath=True)
def l2_sqrt_sum_multi_jit_fastmath(data):
    for x_idx, x in enumerate(data[0]):
        data[0][x_idx] = x / numpy.sqrt(numpy.sum(numpy.multiply(x, x)))
    return data[0]

@jit(nopython=True)
def l2_linalg_norm_multi_jit(data):
    for x_idx, x in enumerate(data[0]):
        data[0][x_idx] = x / numpy.linalg.norm(x)
    return data[0]

@jit(nopython=True, fastmath=True)
def l2_linalg_norm_multi_jit_fastmath(data):
    for x_idx, x in enumerate(data[0]):
        data[0][x_idx] = x / numpy.linalg.norm(x)
    return data[0]

@jit(nopython=True, fastmath=True)
def l2_linalg_norm_multi_jit_fastmath_inner(d):
    for x_idx, x in enumerate(d):
        d[x_idx] = x / numpy.linalg.norm(x)
    return d
def l2_linalg_norm_multi_jit_fastmath_slow(data):
    return l2_linalg_norm_multi_jit_fastmath_inner(data[0])

@jit(nopython=True, fastmath=True)
def l2_linalg_norm_multi_jit_fastmath_cp(data):
    for x_idx, x in enumerate(data[2]):
        data[2][x_idx] = cupy.divide(x, cupy.linalg.norm(x))
    return data[2]

@jit(nopython=True, fastmath=True)
def l2_sqrt_einsum_multi_jit_fastmath(data):
    for x_idx, x in enumerate(data[0]):
        data[0][x_idx] = x / numpy.sqrt(numpy.einsum("ij,ij->i", x, x))
    return data[0]

@jit(nopython=True, fastmath=True)
def l2_linalg_norm_multi_T_jit_fastmath(data):
    for x_idx, x in enumerate(data[1]):
        data[1][x_idx] = x / numpy.linalg.norm(x, axis=0)
    return data[1]

def l2_sqrt_sum_multi(data):
    for x_idx, x in enumerate(data[0]):
        data[0][x_idx] = x / numpy.sqrt(numpy.sum(numpy.multiply(x, x)))
    return data[0]

def l2_sqrt_sum_multi_cp(data):
    for x_idx, x in enumerate(data[2]):
        data[2][x_idx] = cupy.divide(x, cupy.sqrt(cupy.sum(cupy.multiply(x, x))))
    return data[2]

def l2_sqrt_sum_multi_opt(data):
    return numpy.divide(data[0], numpy.sqrt(numpy.sum(numpy.multiply(data[0], data[0]), axis=1))[:, None])

def l2_sqrt_sum_multi_opt_cp(data):
    return cupy.divide(data[2], cupy.sqrt(cupy.sum(cupy.multiply(data[2], data[2]), axis=1))[:, None])

def l2_sqrt_sum_multi_opt_T_cp(data):
    return cupy.divide(data[3], cupy.sqrt(cupy.sum(cupy.multiply(data[3], data[3]), axis=0))[None, :])  # .T

def l2_sqrt_einsum_multi_opt(data):
    return numpy.divide(data[0], numpy.sqrt(numpy.einsum("ij,ij->i", data[0], data[0]))[:, None])

def l2_sqrt_einsum_multi_opt_cp(data):
    return cupy.divide(data[2], cupy.sqrt(cupy.einsum("ij,ij->i", data[2], data[2]))[:, None])

def l2_linalg_norm_multi_opt(data):
    return numpy.divide(data[0], numpy.linalg.norm(data[0], axis=1)[:, None])

def l2_linalg_norm_multi_opt_cp(data):
    return cupy.divide(data[2], cupy.linalg.norm(data[2], axis=1)[:, None])

def l2_linalg_norm_multi_opt_T(data):
    return numpy.divide(data[1], numpy.linalg.norm(data[1], axis=0)[None, :])  # .T

def l2_linalg_norm_multi_opt_T_cp(data):
    return cupy.divide(data[3], cupy.linalg.norm(data[3], axis=0)[None, :])  # .T


def setup(n):
    cupy.get_default_memory_pool().free_all_blocks()
    out0 = numpy.random.rand(n, 640)
    out0_cp = cupy.random.rand(n, 640)
    out1 = out0.T
    out1_cp = cupy.array([out0_cp.T])
    return out0, out1, out0_cp, out1_cp

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(10)],
    kernels=[
        # # l2_sqrt_sum_multi_jit,  # slower than l2_linalg_norm_multi_jit, l2_sqrt_sum_multi_jit_fastmath, discarded
        # # l2_sqrt_sum_multi_jit_fastmath,  # little slower than l2_linalg_norm_multi_jit_fastmath, discarded
        # # l2_python_distance_multi_T_jit_fastmath,  # slower than l2_linalg_norm_multi_jit_fastmath, discarded
        # # l2_linalg_norm_multi_jit,  # slower than l2_linalg_norm_multi_jit_fastmath, discarded
        # l2_linalg_norm_multi_jit_fastmath,  # best if < ~80
        # # l2_linalg_norm_multi_jit_fastmath_slow,  # almost the same speed as l2_linalg_norm_multi_jit_fastmath
        # # l2_linalg_norm_multi_jit_fastmath_cp,  # cupy not supported by jit ???
        # # l2_sqrt_einsum_multi_jit_fastmath,  # einsum not supported by jit
        # # l2_linalg_norm_multi_T_jit_fastmath,  # axis not supported by jit
        # # l2_sqrt_sum_multi,  # slower than l2_sqrt_sum_multi_jit, discarded
        # # l2_sqrt_sum_multi_cp,  #  very slow
        # # l2_sqrt_sum_multi_opt,  # slower than l2_sqrt_einsum_multi_opt, discarded
        l2_sqrt_sum_multi_opt_cp,  # best if > ~80
        l2_sqrt_sum_multi_opt_T_cp,  # almost equal to l2_sqrt_sum_multi_opt_cp
        # # l2_sqrt_einsum_multi_opt,  # slower than l2_linalg_norm_multi_jit_fastmath, discarded
        # # l2_sqrt_einsum_multi_opt_cp,  # alot slower than l2_linalg_norm_multi_opt_cp, discarded
        # # l2_linalg_norm_multi_opt_T,  # little slower than l2_sqrt_sum_multi_opt, discarded
        # # l2_linalg_norm_multi_opt_T_cp,  # slower than l2_sqrt_sum_multi_opt_cp, discarded
        # # l2_linalg_norm_multi_opt,  # little slower than l2_sqrt_sum_multi_opt, discarded
        # # l2_linalg_norm_multi_opt_cp,  # slower than l2_linalg_norm_multi_opt_T_cp, discarded
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")
