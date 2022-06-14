import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit, cuda, vectorize


# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

def scipy_distance(data):
    a, b = data[0]
    return list(map(distance.euclidean, a, b))

def python_distance_T(data):
    a, b = data[1]
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5
    return c

'''@jit()
def sqrt_sum_T_py_jit_nopython1(data):
    d = data[1]
    a, b = d
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5
    return c'''

'''@cuda.jit(device=True)
def inner_python_distance_T_cuda_jit_nopython_fastmath(a, b):
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        summ = 0
        for row in range(a.shape[0]):
            summ += (a[row, col] - b[row, col]) ** 2
        c[col] = summ ** 0.5
    return c
@vectorize(['float64(float64, float64)'], target='cuda')
def inner_python_distance_T_vertorize_nopython_fastmath(a, b):
    return inner_python_distance_T_cuda_jit_nopython_fastmath(a, b)
def python_distance_T_cuda_numba_nopython_fastmath(data):
    a, b = data[1]
    return inner_python_distance_T_vertorize_nopython_fastmath(a, b)'''

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

@jit(nopython=True, fastmath=True)
def inner_linalg_norm_T_jit_nopython_fastmath(d):
    a, b = d
    c = numpy.zeros((a.shape[1],), dtype=numpy.float64)
    for col in range(a.shape[1]):
        c[col] = numpy.linalg.norm(a[:, col] - b[:, col])
    return c
def linalg_norm_T_jit_nopython_fastmath(data):
    return inner_linalg_norm_T_jit_nopython_fastmath(data[1])

@jit(nopython=True, fastmath=True)
def inner_linalg_norm0_T_jit_nopython_fastmath(d):
    c = numpy.zeros((d[0].shape[1],), dtype=numpy.float64)
    for col in range(d[0].shape[1]):
        c[col] = numpy.linalg.norm(d[0][:, col] - d[1][:, col])
    return c
def linalg_norm0_T_jit_nopython_fastmath(data):
    return inner_linalg_norm0_T_jit_nopython_fastmath(data[1])

@jit(nopython=True, fastmath=True)
def inner_sqrt_sum0_jit_nopython_fastmath(d):
    c = numpy.zeros((d[0].shape[0],), dtype=numpy.float64)
    for row in range(d[0].shape[0]):
        c[row] = numpy.sqrt(numpy.sum(numpy.subtract(d[0][row, :], d[1][row, :]) ** 2, axis=0))
    return c
def sqrt_sum0_jit_nopython_fastmath(data):
    return inner_sqrt_sum0_jit_nopython_fastmath(data[0])

@jit(nopython=True, fastmath=True)
def inner_sqrt_einsum0_jit_nopython_fastmath(d):
    c = numpy.zeros((d[0].shape[0],), dtype=numpy.float64)
    for row in range(d[0].shape[0]):
        a_min_b = numpy.subtract(d[0][row, :], d[1][row, :])
        c[row] = numpy.sqrt(numpy.einsum("ij,ij->j", a_min_b, a_min_b))
    return c
def sqrt_einsum0_jit_nopython_fastmath(data):
    return inner_sqrt_einsum0_jit_nopython_fastmath(data[0])

def linalg_norm(data):
    a, b = data[0]
    return numpy.linalg.norm(a - b, axis=1)

def linalg_norm_cp(data):
    a, b = data[2]
    res = cupy.linalg.norm(a - b, axis=1)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res


def linalg_norm_T(data):
    a, b = data[1]
    return numpy.linalg.norm(a - b, axis=0)

def linalg_norm_T_cp(data):
    a, b = data[3]
    res = cupy.linalg.norm(a - b, axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def linalg_norm0_T_cp(data):
    res = cupy.linalg.norm(cupy.subtract(data[3][0], data[3][1]), axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def linalg_norm0_T_cp_1(data):
    res = cupy.linalg.norm(cupy.subtract(data[3][0], data[3][1][0]), axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def linalg_norm0_all_cp(data):  # 1000x1000 = 100ms
    cp_face_features_base = data[2][0]
    cp_face_features_new = data[2][1]
    cp_diff_holder = cupy.zeros_like(cp_face_features_base)
    cp_dist_holder = cupy.zeros((cp_face_features_new.shape[0], cp_face_features_base.shape[0]), dtype=cupy.float32)
    for face_features_new_idx, cp_face_features_new in enumerate(cp_face_features_new):
        cupy.subtract(cp_face_features_base, cp_face_features_new, out=cp_diff_holder)
        cp_dist_holder[face_features_new_idx, :] = cupy.linalg.norm(cp_diff_holder, axis=1)
        cupy.cuda.stream.get_current_stream().synchronize()
    return cp_dist_holder

def linalg_norm0_all_1_cp(data):  # 1000x1000 = 100ms
    cp_face_features_base = data[2][0]
    cp_face_features_new = data[2][1][:1]
    cp_diff_holder = cupy.zeros_like(cp_face_features_base)
    cp_dist_holder = cupy.zeros((cp_face_features_new.shape[0], cp_face_features_base.shape[0]), dtype=cupy.float32)
    for face_features_new_idx, cp_face_features_new in enumerate(cp_face_features_new):
        cupy.subtract(cp_face_features_base, cp_face_features_new, out=cp_diff_holder)
        cp_dist_holder[face_features_new_idx, :] = cupy.linalg.norm(cp_diff_holder, axis=1)
        cupy.cuda.stream.get_current_stream().synchronize()
    return cp_dist_holder

def linalg_norm0_all_10_cp(data):  # 1000x1000 = 100ms
    cp_face_features_base = data[2][0]
    cp_face_features_new = data[2][1][:10]
    cp_diff_holder = cupy.zeros_like(cp_face_features_base)
    cp_dist_holder = cupy.zeros((cp_face_features_new.shape[0], cp_face_features_base.shape[0]), dtype=cupy.float32)
    for face_features_new_idx, cp_face_features_new in enumerate(cp_face_features_new):
        cupy.subtract(cp_face_features_base, cp_face_features_new, out=cp_diff_holder)
        cp_dist_holder[face_features_new_idx, :] = cupy.linalg.norm(cp_diff_holder, axis=1)
        cupy.cuda.stream.get_current_stream().synchronize()
    return cp_dist_holder

def linalg_norm0_T_all_cp(data):
    cp_face_features_base = data[3][0]
    cp_face_features_new = data[3][1]
    cp_diff_holder = cupy.zeros_like(cp_face_features_base)
    cp_dist_holder = cupy.zeros((cp_face_features_new.shape[1], cp_face_features_base.shape[1]), dtype=cupy.float32)
    for face_features_new_idx, cp_face_features_new in enumerate(cp_face_features_new):
        cupy.subtract(cp_face_features_base, cp_face_features_new, out=cp_diff_holder)
        cp_dist_holder[:, face_features_new_idx] = cupy.linalg.norm(cp_diff_holder, axis=0)
        cupy.cuda.stream.get_current_stream().synchronize()
    return cp_dist_holder


def linalg_norm0_T_multi_cp(data):
    c = None
    if data[3][0].shape[0] > 8:
        for _ in range(int(data[3][0].shape[0] / 8)):
            c = cupy.linalg.norm(cupy.subtract(data[3][0][:8], data[3][1][:8]), axis=0)
            cupy.cuda.stream.get_current_stream().synchronize()
    else:
        c = cupy.linalg.norm(cupy.subtract(data[3][0], data[3][1]), axis=0)
        cupy.cuda.stream.get_current_stream().synchronize()
    return c


def sqrt_sum0(data):
    return numpy.sqrt(numpy.sum(numpy.subtract(data[0][0], data[0][1]) ** 2, axis=1))

def sqrt_sum(data):
    a, b = data[0]
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=1))

def sqrt_sum_cp(data):
    a, b = data[2]
    res = cupy.sqrt(cupy.sum((a - b) ** 2, axis=1))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_mem_cp(data):
    a, b = data[2]
    cupy.subtract(a, b, out=a)
    cupy.multiply(a, a, out=a)
    res = cupy.sqrt(cupy.sum(a, axis=1))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res


def sqrt_sum_T(data):
    a, b = data[1]
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=0))

def sqrt_sum_T_cp(data):
    a, b = data[3]
    res = cupy.sqrt(cupy.sum((a - b) ** 2, axis=0))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_T_mem_nosync_cp(data):
    # cp_stream.use()
    a, b = data[3]
    cupy.subtract(a, b, out=a)
    cupy.multiply(a, a, out=a)
    res = cupy.sqrt(cupy.sum(a, axis=0))
    # cp_stream.synchronize()
    return res

def sqrt_sum_T_mem_cp(data):
    a, b = data[3]
    cupy.subtract(a, b, out=a)
    cupy.multiply(a, a, out=a)
    res = cupy.sqrt(cupy.sum(a, axis=0))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_T_mem_multi_cp(data):
    res = None
    if data[3][0].shape[0] > 8:
        for _ in range(int(data[3][0].shape[0] / 8)):
            a, b = data[3][0][:8], data[3][1][:8]
            cupy.subtract(a, b, out=a)
            cupy.multiply(a, a, out=a)
            res = cupy.sqrt(cupy.sum(a, axis=0))
            cupy.cuda.stream.get_current_stream().synchronize()
    else:
        a, b = data[3]
        cupy.subtract(a, b, out=a)
        cupy.multiply(a, a, out=a)
        res = cupy.sqrt(cupy.sum(a, axis=0))
        cupy.cuda.stream.get_current_stream().synchronize()
    return res


def sqrt_einsum0(data):
    a_min_b = numpy.subtract(data[0][0], data[0][1])
    return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))

def sqrt_einsum(data):
    a, b = data[0]
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))

def sqrt_einsum2(data):
    return numpy.sqrt(numpy.einsum("ij,ij->i", numpy.subtract(data[0][0], data[0][1]), numpy.subtract(data[0][0], data[0][1])))

def sqrt_einsum_multi(data):
    a, b = data[0]
    if a.shape[0] > 8:
        part_len = 8
        c = None
        for _ in range(int(a.shape[0] / part_len)):
            a_min_b = a[:part_len] - b[:part_len]
            c = numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))
        return c
    else:
        a_min_b = a - b
        return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))

def sqrt_einsum_cp(data):
    a, b = data[2]
    a_min_b = a - b
    res = cupy.sqrt(cupy.einsum("ij,ij->i", a_min_b, a_min_b))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res


def sqrt_einsum_T(data):
    a, b = data[1]
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum("ij,ij->j", a_min_b, a_min_b))

def sqrt_einsum_T_cp(data):
    a, b = data[3]
    a_min_b = a - b
    res = cupy.sqrt(cupy.einsum("ij,ij->j", a_min_b, a_min_b))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def dot_product1(data):
    a, b = data[0]
    a_min_b = a - b
    return numpy.sqrt(numpy.diag(numpy.dot(a_min_b, a_min_b.T)))

def dot_product1_cp(data):
    a, b = data[2]
    a_min_b = a - b
    res = cupy.sqrt(cupy.diag(cupy.dot(a_min_b, a_min_b.T)))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def dot_product2(data):
    a, b = data[0]
    a_T, b_T = data[1]
    a_min_b = a - b
    a_min_b_T = a_T - b_T
    return numpy.sqrt(numpy.diag(numpy.dot(a_min_b, a_min_b_T)))

def dot_product2_cp(data):
    a, b = data[2]
    a_T, b_T = data[3]
    res = cupy.sqrt(numpy.diag(cupy.dot(a - b, a_T - b_T)))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def compound_algo(data):
    if len(data[0][0]) < 12:
        return python_distance_T_jit_nopython_fastmath(data)
    elif 12 <= len(data[0][0]) <= 90:
        return sqrt_einsum0(data)
    else:
        return linalg_norm0_T_cp(data)

def compound_algo_sum(data):
    if len(data[0][0]) < 10:
        return python_distance_T_jit_nopython_fastmath(data)
    elif 10 <= len(data[0][0]) <= 100:
        return sqrt_einsum0(data)
    else:
        return sqrt_sum_mem_cp(data)

def compound_algo_inline(data):
    if len(data[0][0]) < 12:
        return python_distance_T_jit_nopython_fastmath(data)
    elif 12 <= len(data[0][0]) <= 90:
        a_min_b = numpy.subtract(data[0][0], data[0][1])
        return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))
    else:
        return cupy.linalg.norm(cupy.subtract(data[3][0], data[3][1]), axis=0)

def compound_algo_multi(data):
    if len(data[0][0]) < 12:
        return python_distance_T_jit_nopython_fastmath(data)
    elif 12 <= len(data[0][0]) <= 90:
        return sqrt_einsum_multi(data)
    else:
        return linalg_norm0_T_multi_cp(data)

def compound_algo_to_cpu(data):
    if len(data[0][0]) < 12:
        return python_distance_T_jit_nopython_fastmath(data)
    elif 12 <= len(data[0][0]) <= 90:
        return sqrt_einsum0(data)
    else:
        return cupy.asnumpy(linalg_norm0_T_cp(data))

def compound_algo_to_cpu_and_back(data):
    if len(data[0][0]) < 12:
        return python_distance_T_jit_nopython_fastmath(data)
    elif 12 <= len(data[0][0]) <= 90:
        return sqrt_einsum0(data)
    else:
        a_cp = cupy.asarray(data[0])
        return cupy.asnumpy(linalg_norm0_T_cp(data))

def setup(n):
    cupy.get_default_memory_pool().free_all_blocks()
    a = numpy.random.rand(n, 640)
    b = numpy.random.rand(n, 640)
    a_cp = cupy.random.rand(n, 640)
    b_cp = cupy.random.rand(n, 640)
    # b_cp = cupy.random.rand(1, 640)
    out0 = numpy.array([a, b])
    out1 = numpy.array([a.T, b.T])
    out0_cp = None  # cupy.array([a_cp, b_cp])
    out1_cp = None  # cupy.array([a_cp.T, b_cp.T])
    cupy.cuda.stream.get_current_stream().synchronize()
    return out0, out1, out0_cp, out1_cp

'''def setup_copymem(n):
    a = numpy.random.rand(n, 640)
    b = numpy.random.rand(n, 640)
    a_cp = cupy.asarray(a)
    b_cp = cupy.asarray(b)
    out0 = numpy.array([a, b])
    out1 = numpy.array([a.T, b.T])
    out0_cp = None  # cupy.array([a_cp, b_cp])
    out1_cp = cupy.array([a_cp.T, b_cp.T])
    cupy.cuda.stream.get_current_stream().synchronize()
    return out0, out1, out0_cp, out1_cp'''

def setup_small(n):
    out0 = None
    out1 = None
    if n < 100:
        a = numpy.random.rand(n, 640)
        b = numpy.random.rand(n, 640)
        out0 = numpy.array([a, b])
        out1 = numpy.array([a.T, b.T])
    a_cp = cupy.random.rand(n, 640)
    b_cp = cupy.random.rand(n, 640)
    out0_cp = None  # cupy.array([a_cp, b_cp])
    out1_cp = cupy.array([a_cp.T, b_cp.T])
    cupy.cuda.stream.get_current_stream().synchronize()
    return out0, out1, out0_cp, out1_cp

# stream = cp.cuda.Stream(non_blocking=True)
# cp_stream = cupy.cuda.Stream(non_blocking=False)

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(18)],
    kernels=[
        # scipy_distance,
        # python_distance_T,  # discarded
        # # sqrt_sum_T_py_jit_nopython1,
        # python_distance_T_jit_nopython_fastmath,  # best if < ~10
        # python_distance_T_cuda_numba_nopython_fastmath,
        # # linalg_norm_T_jit_nopython_fastmath,  # little slower than python_distance_T_jit_nopython_fastmath, discarded
        # # linalg_norm0_T_jit_nopython_fastmath,  # equal to linalg_norm_T_jit_nopython_fastmath, discarded
        # # sqrt_sum0_jit_nopython_fastmath,  # slower than python_distance_T_jit_nopython_fastmath if < ~20, discarded
        # # sqrt_einsum0_jit_nopython_fastmath,  # einsum unsupported by jit
        # # linalg_norm,  # slower than sqrt_sum if 10 **2+, discarding
        # linalg_norm_cp,  # fastest, equal to sqrt_sum_mem_cp if > 100  ////////////fastest on cupy, a little slower than linalg_norm_T_cp, but too slow with 10 ** 5+ (GPU OOM???)
        # # linalg_norm_T,  # slower than linalg_norm, discarded
        # linalg_norm_T_cp,  # slower than linalg_norm0_T_cp////////////slower than linalg_norm0_T_cp, discarded
        # linalg_norm0_T_cp,  # fastest ////////////best if > ~90, fastest on cupy, but too slow with 10 ** 5+ (2**18=5GB GPU OOM???)
        # linalg_norm0_T_cp_1,  # fastest, equal to sqrt_sum_T_mem_cp if > 100 (obvious, cuz with a single vector)
        # linalg_norm0_all_cp,  # slowest 1000x1000 = 200-300ms ////////////1000x1000 = 100ms
        # linalg_norm0_all_1_cp,  # slower than sqrt_sum_mem_cp
        # linalg_norm0_all_10_cp,  # slow 1000x1000 = 3ms 10000x10000 = 20ms
        # linalg_norm0_T_all_cp,  #
        # linalg_norm0_T_multi_cp,  # very slow, but should check ////////////much slower than linalg_norm0_T_cp
        # # sqrt_sum,  # very little slower than sqrt_sum0, discarding
        # # sqrt_sum0,  # slower than sqrt_einsum, discarding
        # sqrt_sum_cp,  # slower than sqrt_sum_mem_cp 1000x1000 = 1ms ////////////a little slower than sqrt_sum_T_cp, almost equal, but discarding
        # sqrt_sum_mem_cp,  # best if > ~100 little slower than sqrt_sum_T_mem_cp, little slower than linalg_norm_cp if < 100
        # # sqrt_sum_T,  # slower than sqrt_sum if < 10 ** 2, discarded
        # sqrt_sum_T_cp,  # slow ////////////second to fastest linalg_norm_T_cp, discarded
        # sqrt_sum_T_mem_cp,  # equal to sqrt_sum_T_mem_cp if > 100, 50000 = 10ms
        # sqrt_sum_T_mem_multi_cp,  # little slower than linalg_norm0_T_multi_cp if < 10000, slower than sqrt_sum_T_mem_cp
        # sqrt_sum_T_mem_nosync_cp,
        # # sqrt_einsum,  # very little slower than sqrt_einsum0, discarded
        sqrt_einsum0,  # best if ~10 < ... < ~100, 10000 = 10ms, 100000 = 100ms
        # # sqrt_einsum2,  # slower than sqrt_einsum, discarded
        # sqrt_einsum_multi,
        # sqrt_einsum_cp,  # slower than sqrt_sum_mem_cp ////////////slower on cupy then other cuda algorithms, discarded
        # # sqrt_einsum_T,  # slower than sqrt_einsum_cp, discarded
        # sqrt_einsum_T_cp,  # slower than sqrt_sum_T_mem_cp ////////////slower than sqrt_einsum_cp, discarded
        # # dot_product1,  # slower than sqrt_einsum, discarding
        # dot_product1_cp,  # very slow ////////////slower than linalg_norm_T_cp, discarded
        # # dot_product2,  # slower than dot_product1 if < 10 ** 3, discarded
        # dot_product2_cp,  # very slow ////////////slightly slower than dot_product1_cp, discarded
        # compound_algo,  # very little slower on cuda than compound_algo_inline
        # compound_algo_inline,  # +++
        # compound_algo_multi,
        # compound_algo_to_cpu,
        # compound_algo_to_cpu_and_back,
        # compound_algo_sum,
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")
