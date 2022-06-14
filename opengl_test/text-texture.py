import pytest
import unittest

import numpy

import cupy
from cupy import testing
from cupy.cuda import runtime
from cupy.cuda.texture import (ChannelFormatDescriptor, CUDAarray,
                               ResourceDescriptor, TextureDescriptor,
                               TextureObject, TextureReference)


dev = cupy.cuda.Device(runtime.getDevice())


@testing.gpu
@testing.parameterize(*testing.product({
    'xp': ('numpy', 'cupy'),
    'stream': (True, False),
    'dimensions': ((67, 0, 0), (67, 19, 0), (67, 19, 31)),
    'n_channels': (1, 2, 4),
    'dtype': (numpy.float16, numpy.float32, numpy.int8, numpy.int16,
              numpy.int32, numpy.uint8, numpy.uint16, numpy.uint32),
}))
class TestCUDAarray(unittest.TestCase):
    def test_array_gen_cpy(self):
        xp = numpy if self.xp == 'numpy' else cupy
        stream = None if not self.stream else cupy.cuda.Stream()
        width, height, depth = self.dimensions
        n_channel = self.n_channels

        dim = 3 if depth != 0 else 2 if height != 0 else 1
        shape = (depth, height, n_channel*width) if dim == 3 else \
                (height, n_channel*width) if dim == 2 else \
                (n_channel*width,)

        # generate input data and allocate output buffer
        if self.dtype in (numpy.float16, numpy.float32):
            arr = xp.random.random(shape).astype(self.dtype)
            kind = runtime.cudaChannelFormatKindFloat
        else:  # int
            # randint() in NumPy <= 1.10 does not have the dtype argument...
            arr = xp.random.randint(100, size=shape).astype(self.dtype)
            if self.dtype in (numpy.int8, numpy.int16, numpy.int32):
                kind = runtime.cudaChannelFormatKindSigned
            else:
                kind = runtime.cudaChannelFormatKindUnsigned
        arr2 = xp.zeros_like(arr)

        assert arr.flags['C_CONTIGUOUS']
        assert arr2.flags['C_CONTIGUOUS']

        # create a CUDA array
        ch_bits = [0, 0, 0, 0]
        for i in range(n_channel):
            ch_bits[i] = arr.dtype.itemsize * 8
        # unpacking arguments using *ch_bits is not supported before PY35...
        ch = ChannelFormatDescriptor(ch_bits[0], ch_bits[1], ch_bits[2],
                                     ch_bits[3], kind)
        cu_arr = CUDAarray(ch, width, height, depth)

        # copy from input to CUDA array, and back to output
        cu_arr.copy_from(arr, stream)
        cu_arr.copy_to(arr2, stream)

        # check input and output are identical
        if stream is not None:
            dev.synchronize()
        assert (arr == arr2).all()

