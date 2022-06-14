import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

def sqrt_sum_T_cp(data):
    cupy.subtract(data[0], data[1], out=data[0])
    return cupy.sqrt(cupy.sum(data[0] ** 2, axis=0))

def setup(n):
    return cupy.random.rand(2, 640, n, dtype=cupy.float32)

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(18)],
    kernels=[
        sqrt_sum_T_cp,  # 1.3Gb/18
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")
