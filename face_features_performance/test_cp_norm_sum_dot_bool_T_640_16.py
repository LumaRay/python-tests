import numpy
import cupy
# sudo pip3 install perfplot
import perfplot
from scipy.spatial import distance
from numba import jit
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

def linalg_norm0_T_cp(data):
    return cupy.linalg.norm(cupy.subtract(data[0], data[1]), axis=0)

def linalg_norm0_T_cp_memfree(data):
    cupy.get_default_memory_pool().free_all_blocks()
    # cupy.get_default_pinned_memory_pool().free_all_blocks()
    return cupy.linalg.norm(cupy.subtract(data[0], data[1]), axis=0)

def linalg_norm_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    tmp = cupy.subtract(data[0][0], data[0][1])  # , out=data[0])
    res = cupy.linalg.norm(tmp, axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def linalg_norm_int8_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    # data = (data * 100).astype(cupy.int8)
    cupy.subtract(data[2][0], data[2][1], out=data[4])
    res = cupy.linalg.norm(data[4], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def linalg_norm_bool_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    data = data > 0
    cupy.subtract(data[0], data[1], out=data[0])
    res = cupy.linalg.norm(data[0], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum0_T_cp(data):
    return cupy.sqrt(cupy.sum(cupy.subtract(data[0], data[1]) ** 2, axis=0))

def sqrt_sum_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    cupy.subtract(data[0][0], data[0][1], out=data[3])
    cupy.multiply(data[3], data[3], out=data[3])
    res = cupy.sqrt(cupy.sum(data[3], axis=0))
    # res = cupy.sum(data[0][0], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_T_float16_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    cupy.subtract(data[5][0], data[5][1], out=data[6])
    cupy.multiply(data[6], data[6], out=data[6])
    res = cupy.sqrt(cupy.sum(data[6], axis=0))
    # res = cupy.sum(data[0][0], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_T_street_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    cupy.subtract(data[0][0], data[0][1], out=data[3])
    cupy.abs(data[3], out=data[3])
    # cupy.multiply(data[0][0], data[0][0], out=data[0][0])
    # res = cupy.sqrt(cupy.sum(data[0] ** 2, axis=0))
    res = cupy.sum(data[3], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

'''def sqrt_sum_int8_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    data = (data * 100).astype(cupy.int8)
    cupy.subtract(data[0], data[1], out=data[0])
    # res = cupy.sqrt(cupy.sum(data[0] ** 2, axis=0))
    res = cupy.sum(data[0] ** 2, axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res'''

def sqrt_sum_int8_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    # data = (data * 100).astype(cupy.int8)
    cupy.subtract(data[2][0], data[2][1], out=data[4])
    cupy.multiply(data[4], data[4], out=data[4])
    res = cupy.sqrt(cupy.sum(data[4], axis=0))
    # res = cupy.sum(data[2][0], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

'''def differenceImageV5(img1, img2):
  a = img1-img2
  b = img2-img1
  c = img1>img2
  return a*c + b*(~c)

def differenceImageV6(img1, img2):
  a = img1-img2
  b = np.uint8(img1<img2) * 254 + 1
  return a * b'''

def sqrt_sum_int8_T_street_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    # data = (data * 100).astype(cupy.int8)
    cupy.subtract(data[2][0], data[2][1], out=data[4])
    # cupy.multiply(data[2][0], data[2][0], out=data[2][0])
    cupy.abs(data[4], out=data[4])
    # cupy.multiply(data[4], cupy.uint8(data[2][0] < data[2][1]) * 254 + 1, out=data[4])
    # res = cupy.sqrt(cupy.sum(data[2][0], axis=0))
    res = cupy.sum(data[4], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_int8_T_street_2_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    # data = (data * 100).astype(cupy.int8)
    cupy.subtract(data[2][0], data[2][1], out=data[4])
    # cupy.multiply(data[2][0], data[2][0], out=data[2][0])
    # cupy.abs(data[4], out=data[4])
    # d = cupy.uint8(data[2][0] < data[2][1]) * 254 + 1
    d = (data[2][0] < data[2][1]).astype(cupy.int8) * 254 + 1
    cupy.multiply(data[4], d, out=data[4])
    # res = cupy.sqrt(cupy.sum(data[2][0], axis=0))
    res = cupy.sum(data[4], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_bool_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    data = data > 0
    cupy.subtract(data[0], data[1], out=data[0])
    # cupy.bitwise_xor(data[0], data[1], out=data[0])
    # res = cupy.sqrt(cupy.sum(data[0] ** 2, axis=0))
    res = cupy.sum(data[0], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_bool_xor_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    # data1 = data > 0
    # cupy.subtract(data[0], data[1], out=data[0])
    cupy.bitwise_xor(data[1][0], data[1][1], out=data[1][0])
    # res = cupy.sqrt(cupy.sum(data[0] ** 2, axis=0))
    res = cupy.sqrt(cupy.sum(data[1][0], axis=0))
    # res = cupy.sum(data[0], axis=0)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum_bool_xor_count_T_cp(data):
    cupy.cuda.stream.get_current_stream().synchronize()
    # data1 = data > 0
    # cupy.subtract(data[0], data[1], out=data[0])
    cupy.bitwise_xor(data[1][0], data[1][1], out=data[1][0])
    # res = cupy.sqrt(cupy.sum(data[0] ** 2, axis=0))
    # res = cupy.sum(data[0], axis=0)
    res = cupy.sqrt(cupy.count_nonzero(data[1][0], axis=0))
    cupy.cuda.stream.get_current_stream().synchronize()
    return res

def sqrt_sum2_T_cp(data):
    cupy.subtract(data[0], data[1], out=data[0])
    cupy.multiply(data[0], data[0], out=data[0])
    return cupy.sqrt(cupy.sum(data[0], axis=0))

def dot_product1_cp(data):
    cupy.subtract(data[0], data[1], out=data[0])
    return cupy.sqrt(cupy.diag(cupy.dot(data[0].T, data[0])))

def l2_sqrt_sum_multi_opt_T_cp(data):
    return cupy.divide(data, cupy.sqrt(cupy.sum(cupy.multiply(data, data), axis=0))[None, :])  # .T

def l2_and_linalg_norm0_T_cp_inline(data):
    d = cupy.divide(data, cupy.sqrt(cupy.sum(cupy.multiply(data, data), axis=0))[None, :])  # .T
    return cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)

def l2_and_linalg_norm0_T_cp_inline_multi2(data):
    r = None
    for _ in range(2):
        d = cupy.divide(data[..., :int(data.shape[-1] / 2)], cupy.sqrt(cupy.sum(cupy.multiply(data[..., :int(data.shape[-1] / 2)], data[..., :int(data.shape[-1] / 2)]), axis=0))[None, :])  # .T
        r = cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)
    return r

def l2_and_linalg_norm0_T_cp_inline_multi4(data):
    r = None
    for _ in range(4):
        d = cupy.divide(data[..., :int(data.shape[-1] / 4)], cupy.sqrt(cupy.sum(cupy.multiply(data[..., :int(data.shape[-1] / 4)], data[..., :int(data.shape[-1] / 4)]), axis=0))[None, :])  # .T
        r = cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)
    return r

def l2_and_linalg_norm0_T_cp_inline_multi8(data):
    r = None
    for _ in range(8):
        d = cupy.divide(data[..., :int(data.shape[-1] / 8)], cupy.sqrt(cupy.sum(cupy.multiply(data[..., :int(data.shape[-1] / 8)], data[..., :int(data.shape[-1] / 8)]), axis=0))[None, :])  # .T
        r = cupy.linalg.norm(cupy.subtract(d[0], d[1]), axis=0)
    return r

def setup(n):
    '''mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()

    # Create an array on CPU.
    # NumPy allocates 400 bytes in CPU (not managed by CuPy memory pool).
    a_cpu = numpy.ndarray(100, dtype=numpy.float32)
    print(a_cpu.nbytes)  # 400

    # You can access statistics of these memory pools.
    print(mempool.used_bytes())  # 0
    print(mempool.total_bytes())  # 0
    print(pinned_mempool.n_free_blocks())  # 0

    # Transfer the array from CPU to GPU.
    # This allocates 400 bytes from the device memory pool, and another 400
    # bytes from the pinned memory pool.  The allocated pinned memory will be
    # released just after the transfer is complete.  Note that the actual
    # allocation size may be rounded to larger value than the requested size
    # for performance.
    a = cupy.array(a_cpu)
    print(a.nbytes)  # 400
    print(mempool.used_bytes())  # 512
    print(mempool.total_bytes())  # 512
    print(pinned_mempool.n_free_blocks())  # 1

    # When the array goes out of scope, the allocated device memory is released
    # and kept in the pool for future reuse.
    a = None  # (or `del a`)
    print(mempool.used_bytes())  # 0
    print(mempool.total_bytes())  # 512
    print(pinned_mempool.n_free_blocks())  # 1

    # You can clear the memory pool by calling `free_all_blocks`.
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    print(mempool.used_bytes())  # 0
    print(mempool.total_bytes())  # 0
    print(pinned_mempool.n_free_blocks())  # 0'''

    cupy.get_default_memory_pool().free_all_blocks()
    # cupy.get_default_pinned_memory_pool().free_all_blocks()
    res = cupy.random.rand(2, 640, n, dtype=cupy.float32) - 0.5
    res_bool = res > 0
    # res_int = cupy.random.randint(-5, 5, (2, 640, n), dtype=cupy.int8)
    res_int = cupy.around((res * 10)).astype(cupy.int8)
    cupy.cuda.stream.get_current_stream().synchronize()
    return res, res_bool, res_int, res[0].copy(), res_int[0].copy(), res.astype(cupy.float16), res[0].astype(cupy.float16)

# Use managed memory
# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
# Note that if you pass malloc_managed() directly to set_allocator() without constructing a MemoryPool instance,
# when the memory is freed it will be released back to the system immediately, which may or may not be desired.
# cupy.cuda.set_allocator(cupy.cuda.malloc_managed)  # CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(2, 15)],
    kernels=[
        # linalg_norm0_T_cp,  # 2Gb/18 4Gb/19 (slowdown) 1.3Gb/18 (with memfree, no slowdown) 5.1/20 (with memfree, no slowdown)
        # linalg_norm0_T_cp_memfree,  #
        # linalg_norm_T_cp,  # 1.3Gb/18, slower than sqrt_sum0_T_cp 1Gb/18 (with memfree, no slowdown) 3.9/20 (with memfree, no slowdown)
        # linalg_norm_int8_T_cp,
        # linalg_norm_bool_T_cp,
        # sqrt_sum0_T_cp,  # 2GB/18, slower than linalg_norm0_T_cp 1.3Gb/18 (with memfree, no slowdown) 5.1/20 (with memfree, no slowdown)
        sqrt_sum_T_cp,  # 1.3Gb/18, slower than linalg_norm_T_cp
        sqrt_sum_T_float16_cp,
        # sqrt_sum_T_street_cp,  # just like sqrt_sum_T_cp
        # sqrt_sum_int8_T_cp,  # fastest
        # sqrt_sum_int8_T_street_cp,  # just like sqrt_sum_int8_T_cp
        # sqrt_sum_int8_T_street_2_cp,  # very slow
        # sqrt_sum_bool_T_cp,
        # sqrt_sum_bool_xor_T_cp,  # similar to sqrt_sum_int8_T_cp, but inaccurate
        # sqrt_sum_bool_xor_count_T_cp,  # like sqrt_sum_bool_xor_T_cp
        # sqrt_sum2_T_cp,  # 2Gb/18, slower than sqrt_sum_T_cp
        # dot_product1_cp,  # 4.4Gb/16
        # l2_sqrt_sum_multi_opt_T_cp,  # 1.2Gb/17 2.3Gb/18 (with memfree, no slowdown) almost same speed as linalg_norm0_T_cp
        # l2_and_linalg_norm0_T_cp_inline,  # 1.2Gb/17 2.3Gb/18 (with memfree, no slowdown)
        # l2_and_linalg_norm0_T_cp_inline_multi2,  # 1Gb/17 0.5Gb/16 (with memfree, no slowdown)
        # l2_and_linalg_norm0_T_cp_inline_multi4,  # 1.3Gb/18 0.7Gb/17 0.4Gb/16 0.2Gb/15 (with memfree, no slowdown)
        # l2_and_linalg_norm0_T_cp_inline_multi8,  # ~1ms if 8 parts, ~0.1ms if 1 part (128K part = 17x1 = 0.7Gb)
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")
