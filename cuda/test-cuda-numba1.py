import numpy
import math
from numba import cuda

# @cuda.jit
# def my_kernel(io_array):
#     # Thread id in a 1D block
#     tx = cuda.threadIdx.x
#     # Block id in a 1D grid
#     ty = cuda.blockIdx.x
#     # Block width, i.e. number of threads per block
#     bw = cuda.blockDim.x
#     # Compute flattened index inside the array
#     pos = tx + ty * bw
#     if pos < io_array.size:  # Check array boundaries
#         io_array[pos] *= 2 # do the computation

@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# # Create the data array - usually initialized some other way
# data = numpy.ones(256*1024*1024*2)
#
# # Set the number of threads in a block
# threadsperblock = 32
#
# # Calculate the number of thread blocks in the grid
# blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
#
# # Now start the kernel
# my_kernel[blockspergrid, threadsperblock](data)
# my_kernel[blockspergrid, threadsperblock](data)
# my_kernel[blockspergrid, threadsperblock](data)
# my_kernel[blockspergrid, threadsperblock](data)
# my_kernel[blockspergrid, threadsperblock](data)
# my_kernel[blockspergrid, threadsperblock](data)
# my_kernel[blockspergrid, threadsperblock](data)
#
# # Print the result
# print(data)

# Host code
data = numpy.ones(256*1024*1024*2)
# threadsperblock = 32
threadsperblock = 128
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
my_kernel[blockspergrid, threadsperblock](data)
my_kernel[blockspergrid, threadsperblock](data)
my_kernel[blockspergrid, threadsperblock](data)
my_kernel[blockspergrid, threadsperblock](data)
my_kernel[blockspergrid, threadsperblock](data)
my_kernel[blockspergrid, threadsperblock](data)
print(data)