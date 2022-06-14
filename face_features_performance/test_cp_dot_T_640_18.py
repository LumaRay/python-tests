import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

def dot_product1_cp(data):
    cupy.subtract(data[0], data[1], out=data[0])
    return cupy.sqrt(cupy.diag(cupy.dot(data[0].T, data[0])))

def setup(n):
    return cupy.random.rand(2, 640, n, dtype=cupy.float32)

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(16)],
    kernels=[
        dot_product1_cp,  # 4.4Gb/16
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")
