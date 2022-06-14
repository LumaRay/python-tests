import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.tools

cuda.init()
assert cuda.Device.count() >= 1

# context = pycuda.tools.make_default_context()
# context = cudagl.make_context(device)
# context = context

import pycuda.autoinit

curr_gpu = pycuda.autoinit.device


from ctypes import *
hdc = windll.user32.GetDC(1)
print(hdc)

# from OpenGL.GL import *
# import OpenGL
import OpenGL.WGL

hglrc = OpenGL.WGL.wglCreateContext(hdc)
OpenGL.WGL.wglMakeCurrent(hdc, hglrc)


ctx_gl = pycuda.gl.make_context(curr_gpu, flags=pycuda.gl.graphics_map_flags.NONE)

ctx_gl = ctx_gl