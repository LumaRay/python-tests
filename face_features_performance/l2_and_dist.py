import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

@jit(nopython=True, fastmath=True)
def inner_python_distance_T_jit_nopython_fastmath(d):
    a, b = d
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5
    return c
def python_distance_T_jit_nopython_fastmath(data):
    return inner_python_distance_T_jit_nopython_fastmath(data[1])

def inner_sqrt_einsum0(d):
    a_min_b = numpy.subtract(d[0], d[1])
    return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))

def sqrt_einsum0(data):
    a_min_b = numpy.subtract(data[0][0], data[0][1])
    return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))

def inner_linalg_norm0_T_cp(d):
    return cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)

def linalg_norm0_T_cp(data):
    return cupy.linalg.norm(cupy.subtract(data[3][0], data[3][1]), axis=0)






def l2_sqrt_sum_multi_opt_cp(data):
    return cupy.divide(data[2], cupy.sqrt(cupy.sum(cupy.multiply(data[2], data[2]), axis=1))[:, None])

def l2_sqrt_sum_multi_opt_T_cp(data):
    return cupy.divide(data[3], cupy.sqrt(cupy.sum(cupy.multiply(data[3], data[3]), axis=0))[None, :])  # .T

@jit(nopython=True, fastmath=True)
def inner_l2_linalg_norm_multi_jit_fastmath(d):
    for x_idx, x in enumerate(d):
        d[x_idx] = x / numpy.linalg.norm(x)
    return d
def l2_linalg_norm_multi_jit_fastmath(data):
    return inner_l2_linalg_norm_multi_jit_fastmath(data[0])

@jit(nopython=True, fastmath=True)
def inner_l2_linalg_norm_multi_T_jit_fastmath(d):
    for x_idx, x in enumerate(d):
        d[x_idx] = x / numpy.linalg.norm(x)
    return d
def l2_linalg_norm_multi_T_jit_fastmath(data):
    return inner_l2_linalg_norm_multi_T_jit_fastmath(data[1])





@jit(nopython=True, fastmath=True)
def l2_and_python_distance_T_jit_fastmath(data):
    a, b = data[1]
    for x_idx, x in enumerate(a):
        summ = 0
        for col in range(len(x)):
            summ += x[col] ** 2
        a[x_idx] = x / (summ ** 0.5)
    for x_idx, x in enumerate(b):
        summ = 0
        for col in range(len(x)):
            summ += x[col] ** 2
        b[x_idx] = x / (summ ** 0.5)
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5
    return c

def l2_and_python_distance_T_jit_nopython_fastmath(data):
    return inner_python_distance_T_jit_nopython_fastmath(l2_linalg_norm_multi_T_jit_fastmath(data))

def l2_and_sqrt_einsum0(data):
    return inner_sqrt_einsum0(l2_linalg_norm_multi_jit_fastmath(data))

def l2_and_linalg_norm0_T_cp(data):
    return inner_linalg_norm0_T_cp(l2_sqrt_sum_multi_opt_T_cp(data))

def l2_and_linalg_norm0_T_cp_inline(data):
    d = cupy.divide(data[3], cupy.sqrt(cupy.sum(cupy.multiply(data[3], data[3]), axis=0))[None, :])  # .T
    return cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)

'''def compound_algo(data):
    if len(data[0][0]) < 12:
        return inner_python_distance_T_jit_nopython_fastmath(l2_linalg_norm_multi_T_jit_fastmath(data))
    elif 12 <= len(data[0][0]) <= 90:
        return inner_sqrt_einsum0(l2_linalg_norm_multi_jit_fastmath(data))
    else:
        return inner_linalg_norm0_T_cp(l2_sqrt_sum_multi_opt_T_cp(data))'''

def compound_algo2(data):
    if len(data[0][0]) < 50:
        return inner_sqrt_einsum0(l2_linalg_norm_multi_jit_fastmath(data))
    else:
        return inner_linalg_norm0_T_cp(l2_sqrt_sum_multi_opt_T_cp(data))

def compound_algo2_inline(data):
    if len(data[0][0]) < 50:
        d = l2_linalg_norm_multi_jit_fastmath(data)
        a_min_b = numpy.subtract(d[0], d[1])
        return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))
    else:
        d = cupy.divide(data[3], cupy.sqrt(cupy.sum(cupy.multiply(data[3], data[3]), axis=0))[None, :])  # .T
        return cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)

def setup(n):
    cupy.get_default_memory_pool().free_all_blocks()
    a = numpy.random.rand(n, 640)
    b = numpy.random.rand(n, 640)
    a_cp = cupy.random.rand(n, 640)
    b_cp = cupy.random.rand(n, 640)
    out0 = numpy.array([a, b])
    out1 = numpy.array([a.T, b.T])
    out0_cp = None  # cupy.array([a_cp, b_cp])
    out1_cp = cupy.array([a_cp.T, b_cp.T])
    return out0, out1, out0_cp, out1_cp

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(15)],
    kernels=[
        # # l2_and_python_distance_T_jit_fastmath,  # slower than l2_and_sqrt_einsum0, discarded
        # # compound_algo, has l2_and_python_distance_T_jit_nopython_fastmath, discarded
        # # l2_and_python_distance_T_jit_nopython_fastmath,  # slower than l2_and_sqrt_einsum0, discarded
        # l2_and_sqrt_einsum0,  # best if < 50
        # l2_and_linalg_norm0_T_cp,  # best if >= 50
        # compound_algo2,  # very little slower than compound_algo2_inline
        compound_algo2_inline,
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")
