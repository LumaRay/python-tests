import numpy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit, njit
# sudo pip3 install --upgrade tbb

def sqrt_sum_T(data):
    a, b = data
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=0))


def sqrt_sum_T_py(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit()
def sqrt_sum_T_py_jit(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(fastmath=True)
def sqrt_sum_T_py_jit_fastmath(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(nopython=True)
def sqrt_sum_T_py_jit_nopython(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(nopython=True, fastmath=True)
def sqrt_sum_T_py_jit_nopython_fastmath(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(parallel=True)
def sqrt_sum_T_py_jit_parappel(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(parallel=True, fastmath=True)
def sqrt_sum_T_py_jit_parappel_fastmath(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(nopython=True, parallel=True)
def sqrt_sum_T_py_jit_nopython_parappel(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


@jit(nopython=True, parallel=True, fastmath=True)
def sqrt_sum_T_py_jit_nopython_parappel_fastmath(data):
    a, b = data
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5

    return c


def setup(n):
    a = numpy.random.rand(n, 640)
    b = numpy.random.rand(n, 640)
    out1 = numpy.array([a.T, b.T])
    return out1


b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(19)],
    kernels=[
        sqrt_sum_T,
        # sqrt_sum_T_py,
        # sqrt_sum_T_py_jit,
        sqrt_sum_T_py_jit_fastmath,
        # sqrt_sum_T_py_jit_nopython,
        # sqrt_sum_T_py_jit_nopython_fastmath,
        # sqrt_sum_T_py_jit_parappel,
        # sqrt_sum_T_py_jit_parappel_fastmath,
        # sqrt_sum_T_py_jit_nopython_parappel,
        # sqrt_sum_T_py_jit_nopython_parappel_fastmath,
    ],
    xlabel="len(x), len(y)",
    equality_check=None,
)
b.save("norm.png")
