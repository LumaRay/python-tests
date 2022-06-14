import gc

import numpy
import cupy
import cupyx
import cupyx.scipy.ndimage
import perfplot
import cv2

import tensorflow as tf

from numba import jit

# SRC_SIZE = (720, 1280)
# DST_SIZE = (640, 640)

# SRC_SIZE = (200, 300)
# DST_SIZE = (160, 160)

SRC_SIZE = (200, 300)
DST_SIZE = (128, 128)

TENSORFLOW_GPU_MEMORY_LIMIT_MB = 128  # 256 # 512 # 1024  # 2048
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=TENSORFLOW_GPU_MEMORY_LIMIT_MB)])
    except RuntimeError as e:
        print(e)

import torch
from torch.utils.dlpack import to_dlpack
import torchvision
from torchvision import transforms as trans

def resize_cv2_upload_cv2(data):
    d_src = None
    for di, d in enumerate(data):
        r = cv2.resize(d, DST_SIZE)
        d_src = cv2.cuda_GpuMat()
        d_src.upload(r)
    return d_src

@jit(nopython=True, fastmath=True)
def resize_cv2_jit(data):
    res = numpy.zeros((data.shape[0], DST_SIZE[0], DST_SIZE[1], data.shape[-1]), dtype=numpy.uint8)
    for di, d in enumerate(data):
        res[di] = cv2.resize(d, DST_SIZE)
def resize_cv2_upload_cupy_jit_fastmath(data):
    res = resize_cv2_jit(data)
    d_src = cupy.asarray(res)
    return d_src

def resize_cv2_upload_cupy(data):
    res = numpy.zeros((data.shape[0], DST_SIZE[0], DST_SIZE[1], data.shape[-1]), dtype=numpy.uint8)
    for di, d in enumerate(data):
        res[di] = cv2.resize(d, DST_SIZE)
    d_src = cupy.asarray(res)
    return d_src

# def resize_cv2_upload_tf(data):
#     res = numpy.zeros((data.shape[0], DST_SIZE[0], DST_SIZE[1], data.shape[-1]), dtype=numpy.uint8)
#     for di, d in enumerate(data):
#         res[di] = cv2.resize(d, DST_SIZE)
#     with tf.device('/GPU:0'):
#         d_res = tf.convert_to_tensor(res, dtype=tf.float32)
#     tf.to
#     return d_res

def upload_cv2_resize_cv2(data):
    d_res = None  # cv2.cuda_GpuMat()
    for di, d in enumerate(data):
        d_src = cv2.cuda_GpuMat()
        d_src.upload(d)
        d_res = cv2.cuda.resize(d_src, DST_SIZE)
    return d_res

def upload_cupy_resize_scipy(data):
    d_src = cupy.asarray(data)
    M = cupy.eye(4)
    M[0][0] = M[1][1] = 0.5
    smaller_shape = DST_SIZE + (d_src.shape[-1],)
    smaller = cupy.zeros(((d_src.shape[0],) + smaller_shape), dtype=cupy.uint8)
    for di, d in enumerate(d_src):
        cupyx.scipy.ndimage.affine_transform(d, M, output_shape=DST_SIZE + (3,), output=smaller[di], mode='opencv')
    return smaller

def upload_cupy_resize_tf(data):
    d_src = cupy.asarray(data)
    dl_src = d_src.toDlpack()
    tf_src = tf.experimental.dlpack.from_dlpack(dl_src)
    res = tf.image.resize(tf_src, [DST_SIZE[0], DST_SIZE[1]])
    # res = tf.cast(res, 'uint8').eval()
    return res

def upload_cupy_resize_torch(data):
    # ddd = cv2.imread('/home/thermalview/Desktop/ThermalView/tests/train_models/test_set/S_2_3_0-2020_11_19__14_26_25_FULL_NORMAL_SOURCE.jpg')
    # data = numpy.expand_dims(ddd, 0)
    d_src = cupy.asarray(data)
    dl_src = d_src.toDlpack()
    torch_src = torch.utils.dlpack.from_dlpack(dl_src)
    torch_src = torch_src.permute(0, 3, 1, 2)
    # resize_transform = trans.Compose([
    #     trans.Resize([int(640), int(640)]),
    #     trans.ToTensor(),
    # ])
    # res = resize_transform(torch_src)
    res = torch.nn.functional.interpolate(torch_src, DST_SIZE)
    # test = res[0].permute(1, 2, 0).cpu().numpy()
    return res

def upload_cupy_resize_torch_opt(data):
    return torch.nn.functional.interpolate(torch.utils.dlpack.from_dlpack(cupy.asarray(data).toDlpack()).permute(0, 3, 1, 2), DST_SIZE)

def upload_torch_resize_torch(data):
    torch_data = torch.from_numpy(data)
    cuda0 = torch.device('cuda:0')
    torch_src = torch_data.to(cuda0)
    torch_src = torch_src.permute(0, 3, 1, 2)
    res = torch.nn.functional.interpolate(torch_src, DST_SIZE)
    return res

def resize_torch_upload_torch(data):
    torch_data = torch.from_numpy(data)
    torch_data = torch_data.permute(0, 3, 1, 2)
    torch_res = torch.nn.functional.interpolate(torch_data, DST_SIZE)
    cuda0 = torch.device('cuda:0')
    torch_src = torch_res.to(cuda0)
    return torch_src

def resize_torch_upload_cupy(data):
    torch_data = torch.from_numpy(data)
    torch_data = torch_data.permute(0, 3, 1, 2)
    torch_res = torch.nn.functional.interpolate(torch_data, DST_SIZE)
    np_res = torch_res.numpy()
    d_res = cupy.asarray(np_res)
    return d_res

def setup(n):
    tensor = 1
    v = tf.constant([1])
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    gc.collect()
    return (numpy.random.rand(n, SRC_SIZE[0], SRC_SIZE[1], 3) * 255).astype(numpy.uint8)  # numpy.random.rand(n, 1280, 720, 3).astype(numpy.uint8)  # numpy.zeros((n, 1280, 720, 3), dtype=numpy.uint8)

b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(6)],
    kernels=[
        # resize_cv2_upload_cv2,  # slower than upload_cv2_resize_cv2
        # upload_cv2_resize_cv2,  # slower than upload_torch_resize_torch
        resize_cv2_upload_cupy,  # slower than upload_torch_resize_torch
        # resize_cv2_upload_tf,  # cannot transfer TF CPU->GPU
        # resize_cv2_upload_cupy_jit_fastmath,  # cannot find opencv attributes
        # upload_cupy_resize_scipy,  # slower than upload_cupy_resize_tf
        # upload_cupy_resize_tf,  # slower than resize_cv2_upload_cupy
        upload_cupy_resize_torch,  # best if 640x640 # best if 160x160 and >= 3 # best if 128x128 and >= 5
        # upload_cupy_resize_torch_opt,  # almost the same time as upload_cupy_resize_torch
        # resize_torch_upload_cupy,  # slower than resize_torch_upload_torch
        # resize_torch_upload_torch,  # slower than upload_torch_resize_torch
        upload_torch_resize_torch,  # slower than upload_cupy_resize_torch if 640x640  # best if 160x160 and < 3 # best if 128x128 and < 5
    ],
    xlabel="len(x), len(y)",
    # equality_check=cupy.allclose,
    equality_check=None,
)
# b.show(relative_to=0)
b.save("norm.png")