import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance


def linalg_norm(data):
    a, b = data[0]
    return cupy.linalg.norm(a - b, axis=1)


def linalg_norm_T(data):
    a, b = data[1]
    return cupy.linalg.norm(a - b, axis=0)


def sqrt_sum(data):
    a, b = data[0]
    return cupy.sqrt(cupy.sum((a - b) ** 2, axis=1))


def sqrt_sum_T(data):
    a, b = data[1]
    return cupy.sqrt(cupy.sum((a - b) ** 2, axis=0))


def sqrt_einsum(data):
    a, b = data[0]
    a_min_b = a - b
    return cupy.sqrt(cupy.einsum("ij,ij->i", a_min_b, a_min_b))


def sqrt_einsum_T(data):
    a, b = data[1]
    a_min_b = a - b
    return cupy.sqrt(cupy.einsum("ij,ij->j", a_min_b, a_min_b))


def setup(n):
    a = cupy.random.rand(n, 640)
    b = cupy.random.rand(n, 640)
    out0 = cupy.array([a, b])
    out1 = cupy.array([a.T, b.T])
    return out0, out1


b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(12)],
    kernels=[
        linalg_norm,
        linalg_norm_T,
        sqrt_sum,
        sqrt_sum_T,
        sqrt_einsum,
        sqrt_einsum_T,
    ],
    xlabel="len(x), len(y)",
    equality_check=cupy.allclose,
)
b.save("norm.png")
