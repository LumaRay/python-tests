import sys
import threading
import time

import cv2
import numpy as np
from pafy import pafy
from scipy.misc import ascent
# https://pixnio.com/photos/people
import OpenGL.GL as gl
import wx
from wx.glcanvas import GLCanvas

# import cv2
# print(cv2.getBuildInformation())

import cupy as cp
from cupy.cuda import runtime
dev = cp.cuda.Device(runtime.getDevice())

import pycuda
import pycuda.driver
import pycuda.gl

import pycuda.autoinit

# curr_gpu = pycuda.autoinit.device
# ctx_gl = pycuda.gl.make_context(curr_gpu, flags=pycuda.gl.graphics_map_flags.NONE)

# sudo systemctl restart nvargus-daemon
#vcap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink").format(1920, 1080), cv2.CAP_GSTREAMER)

#vcap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True").format(1920, 1080), cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

# vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)

#vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! queue max-size-buffers=1 leaky=downstream ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! queue max-size-buffers=1 leaky=downstream ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! queue ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080,format=(string)BGRx ! queue ! videoconvert ! queue ! appsink sync=false", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! decodebin ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=20 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=200 ! rtph264depay ! h264parse ! queue ! omxh264dec ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=20 ! decodebin ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nv3dsink -e", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=500 ! rtph264depay ! h264parse ! omxh264dec ! nvoverlaysink overlay-x=800 overlay-y=50 overlay-w=640 overlay-h=480 overlay=2", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.7.147", cv2.CAP_FFMPEG)
#vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080,format=(string)BGRx ! queue ! videoconvert ! queue ! appsink sync=false", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! videoconvert ! queue ! appsink sync=false", cv2.CAP_GSTREAMER)
# gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! decodebin ! autovideosink
# gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=20 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink
# gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=500 ! rtph264depay ! h264parse ! omxh264dec ! nvoverlaysink overlay-x=800 overlay-y=50 overlay-w=640 overlay-h=480 overlay=2
# rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! decodebin ! nvvidconv ! video/x-raw, format=I420, width=1920, height=816 ! appsink
# gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! queue ! h264parse ! omxh264dec ! nveglglessink -e
# gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nv3dsink -e
# gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! qtdemux ! h264parse ! queue ! omxh264dec ! nvvidconv ! tee ! xvimagesink

# gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! nvoverlaysink

# vPafy = pafy.new('https://www.youtube.com/watch?v=HC9FDfuUpKQ')
# play = vPafy.getbest()  # (preftype="webm")
# vcap = cv2.VideoCapture(play.url)

cap_width, cap_height = 1280, 720

test_acc = 0
test_cnt = 0

class Canvas(GLCanvas):

    def __init__(self, parent):
        """create the canvas """
        super(Canvas, self).__init__(parent)
        self.texture = None
        self.parent = parent

        self.context = wx.glcanvas.GLContext(self)

        # execute self.onPaint whenever the parent frame is repainted
        self.Bind(wx.EVT_PAINT, self.onPaint)

        self.Bind(wx.EVT_WINDOW_DESTROY, self.onDestroy)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        '''self.parent.timerUpdateUI = wx.Timer(self.parent)
        self.parent.Bind(wx.EVT_TIMER, self.UpdateUI)
        self.parent.timerUpdateUI.Start(1000. / 1)'''

        self.closeThreads = False
        self.hGetNormalFrameThread = threading.Thread(target=self.getNormalFrameThread)  # , args=(self,))
        self.hGetNormalFrameThread.start()

        self.frame = None

        # pycuda.driver.init()
        # dev = pycuda.driver.Device(0)
        # pycuda.gl.init()
        # self.cuda_gl_context = pycuda.gl.make_context(dev)
        # pycuda.gl.BufferObjectMapping()

    def initTexture(self):
        """init the texture - this has to happen after an OpenGL context
        has been created
        """

        # make the OpenGL context associated with this canvas the current one
        self.SetCurrent(self.context)

        curr_gpu = pycuda.autoinit.device
        self.ctx_gl = pycuda.gl.make_context(curr_gpu, flags=pycuda.gl.graphics_map_flags.NONE)  # WRITE_DISCARD

        #self.data = np.uint8(np.flipud(ascent()))
        #self.data = np.zeros((1080, 1920), dtype=np.uint8)
        # self.data = np.zeros((cap_height, cap_width, 4), dtype=np.uint8)
        '''img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
        self.data = cv2.imread(img_path)
        self.data = np.flipud(self.data)
        self.data = np.dstack((self.data, np.zeros(self.data.shape[:-1])))
        # self.data = np.dstack((self.data, np.zeros((self.data.shape[:-1]) + (1,))))'''

        # ret, frame = vcap.read()
        # if ret:
        #     frame = np.flipud(frame)
        #     self.data = np.dstack((frame, np.zeros(frame.shape[:-1], dtype=frame.dtype)))

        # init a buffer
        # https://stackoverflow.com/questions/21765604/draw-image-from-vertex-buffer-object-generated-with-cuda-using-opengl
        # gl.glGenBuffers(1, buffer)
        self.buffer = gl.glGenBuffers(1)
        # gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        # gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, cap_width * cap_height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, cap_width * cap_height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # this.bufferResource = new CUgraphicsResource();
        # cuGraphicsGLRegisterBuffer(bufferResource, self.buffer,
        #                            CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)

        # generate a texture id, make it current
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        # texture mode and parameters controlling wrapping and scaling
        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        # map the image data to the texture. note that if the input
        # type is GL_FLOAT, the values must be in the range [0..1]

        # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.data.shape[1], self.data.shape[0], 0, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.data)
        # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, cap_width, cap_height, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.data)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, cap_width, cap_height, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, None)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.cuda_buf = pycuda.gl.RegisteredBuffer(int(self.buffer))  # , pycuda.gl.graphics_map_flags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
        # cuda_buf_mapping = cuda_buf.map()
        self.cuda_buf_mapping = self.cuda_buf.map()
        d_buf_ptr, d_buf_size = self.cuda_buf_mapping.device_ptr_and_size()
        # memory = cp.cuda.Memory(2048000000)
        # ptr = cp.cuda.MemoryPointer(memory, 0)
        self.cp_mem = cp.cuda.memory.UnownedMemory(d_buf_ptr, d_buf_size, self.cuda_buf_mapping)
        self.cp_memptr = cp.cuda.memory.MemoryPointer(self.cp_mem, 0)
        # memptr.memset(100, 100000)
        # cp_rand = cp.random.randint(100, size=(cap_height, cap_width, 4)).astype(cp.uint8)
        # cp_rand.data
        # memptr.copy_from_device_async(cp_rand.)
        # memptr.memset_async(100, 100000)
        self.cp_arr = cp.ndarray(shape=(cap_height, cap_width, 4), memptr=self.cp_memptr, dtype=cp.uint8)
        # cp_arr[0:100, 0:100, 0] = 200
        # ri = pycuda.gl.RegisteredImage(int(self.texture), gl.GL_TEXTURE_2D)


        #self.data[30:40, 30:40] = 255


    def onPaint(self, event):
        """called when window is repainted """
        # make sure we have a texture to draw
        if not self.texture:
            self.initTexture()
            # make the OpenGL context associated with this canvas the current one
            self.SetCurrent(self.context)
        self.onDraw()
        # if self.frame is not None:
        #     gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.frame)
        #     self.SwapBuffers()

    def onDraw(self):
        """draw function """

        time_start_full = round(time.monotonic() * 1000)

        time_start_prep = time_start_full

        # set the viewport and projection
        w, h = self.GetSize()
        gl.glViewport(0, 0, w, h)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, 0, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        time_taken_prep = round(time.monotonic() * 1000) - time_start_prep

        # ACTION!
        action_start = round(time.monotonic() * 1000)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, cap_height * cap_width * 4, self.frame, gl.GL_DYNAMIC_DRAW)
        # buf_mem_pointer = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        '''gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, cap_height * cap_width * 4, self.frame, gl.GL_DYNAMIC_DRAW)
        buf_mem_pointer = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY)  # gl.GL_READ_WRITE)  #   #   #
        map_array = (gl.GLbyte * cap_height * cap_width * 4).from_address(buf_mem_pointer)
        # new_array = np.ctypeslib.as_array(map_array, shape=(cap_height * cap_width * 4,))  # .astype(np.uint8)
        new_array = np.ctypeslib.as_array(map_array, shape=(cap_height, cap_width, 4))  # .astype(np.uint8)
        # new_array = new_array.reshape((cap_height, cap_width, 4))
        # new_array = np.flip(new_array, 0)
        # new_array = np.ndarray(shape=(cap_height, cap_width, 4), dtype=np.uint8, buffer=buf_mem_pointer, offset=0)
        # import ctypes
        # new_array = np.ctypeslib.as_array(buf_mem_pointer, shape=None)
        # new_array = np.ndarray(shape=(cap_height, cap_width, 4), dtype=np.uint8, buffer=ctypes.c_void_p.from_address(buf_mem_pointer))
        # new_array[:50, :100, :] = 100
        # new_array = new_array.astype(np.uint8)
        new_array = new_array.reshape((cap_height, cap_width, 4))
        # new_array3 = new_array[:, :, :3]  # .astype(np.uint8)
        # new_array3 = new_array3.view(np.uint8)
        # new_array3 += 128
        # new_array3 = np.ascontiguousarray(new_array, dtype=np.uint8)
        # new_array3[:50, :100, :] = 100
        # new_array[...] = cv2.flip(new_array, 0)
        cv2.putText(new_array,
                    "M!",
                    (100, 100),
                    cv2.FONT_HERSHEY_TRIPLEX,  # font,
                    2,  # -2,
                    (255, 255, 255, 255),
                    2,  # thickness
                    cv2.LINE_AA)
        # new_array[...] = np.flipud(new_array)
        gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)'''

        # ca_arr.copy_from(cp_arr)
        # pycuda.gl.RegisteredImage(self.buffer, pycuda.gl.RegisteredImage.GL_TEXTURE_2D, flags=pycuda.gl.graphics_map_flags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
        '''cuda_buf = pycuda.gl.RegisteredBuffer(int(self.buffer))  # , pycuda.gl.graphics_map_flags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
        # cuda_buf_mapping = cuda_buf.map()
        cuda_buf_mapping = cuda_buf.map()
        d_buf_ptr, d_buf_size = cuda_buf_mapping.device_ptr_and_size()
        # memory = cp.cuda.Memory(2048000000)
        # ptr = cp.cuda.MemoryPointer(memory, 0)
        mem = cp.cuda.memory.UnownedMemory(d_buf_ptr, d_buf_size, cuda_buf_mapping)
        memptr = cp.cuda.memory.MemoryPointer(mem, 0)
        # memptr.memset(100, 100000)
        # cp_rand = cp.random.randint(100, size=(cap_height, cap_width, 4)).astype(cp.uint8)
        # cp_rand.data
        # memptr.copy_from_device_async(cp_rand.)
        # memptr.memset_async(100, 100000)
        cp_arr = cp.ndarray(shape=(cap_height, cap_width, 4), memptr=memptr, dtype=cp.uint8)
        cp_arr[0:100, 0:100, 0] = 200'''
        self.cp_arr[0:100, 0:100, 0] = 200
        # cp_arr.reshape((cap_height * cap_width * 4,))
        # cp_arr = cp.ndarray(shape=(cap_height * cap_width * 4,), dtype=cp.uint8, memptr=memptr)
        # cp_arr[0:234] = 55
        '''cp_arr_2 = cp_arr.reshape(cap_height, cap_width, 4)
        cp_arr_2_fl = cp.flipud(cp_arr_2)
        cp_arr_2_fl[0:30, 0:30] = 100
        del cp_arr_2_fl
        cp_arr_2_fl = None
        del cp_arr_2
        cp_arr_2 = None
        del cp_arr
        cp_arr = None
        memptr = None
        mem = None'''
        # np_arr = cp.asnumpy(cp_arr)
        # arr = cuda_buf_mapping.array(0, 0)
        # cuda_buf_mapping.unmap()
        # rgb_arr = pycuda.driver.from_device(
        #     d_buf_ptr,
        #     # shape=(1, cap_height, cap_width, 4),
        #     shape=(cap_height, cap_width, 4),
        #     dtype=np.uint8)
        # rgb_arr = pycuda.driver.to_device(
        #     d_buf_ptr,
        #     # shape=(1, cap_height, cap_width, 4),
        #     shape=(cap_height, cap_width, 4),
        #     dtype=np.uint8)
        '''cuda_buf_mapping.unmap()
        cuda_buf_mapping = None
        cuda_buf.unregister()
        cuda_buf = None'''

        # self.ctx_gl.synchronize()

        action_time_taken = round(time.monotonic() * 1000) - action_start

        time_start_copy_tex = round(time.monotonic() * 1000)

        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        # gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.buffer)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, cap_width, cap_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        # enable textures, bind to our texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glEnable(gl.GL_TEXTURE_2D)

        time_taken_tex = round(time.monotonic() * 1000) - time_start_copy_tex

        '''if self.frame is not None:
            #gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.frame)
            rc1 = sys.getrefcount(self.frame)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.frame)
            rc2 = sys.getrefcount(self.frame)
            self.frame = None
            #print('teximage2d ', self.frame.flags['C_CONTIGUOUS'], rc1, rc2)'''

        if self.frame is not None:
            # cp_arr = cp.asarray(self.frame)

            '''# create a CUDA array
            ch_bits = [0, 0, 0, 0]
            for i in range(n_channel):
                ch_bits[i] = arr.dtype.itemsize * 8
            # unpacking arguments using *ch_bits is not supported before PY35...
            ch = ChannelFormatDescriptor(ch_bits[0], ch_bits[1], ch_bits[2],
                                         ch_bits[3], kind)
            cu_arr = CUDAarray(ch, width, height, depth)

            # copy from input to CUDA array, and back to output
            cu_arr.copy_from(arr, stream)
            cu_arr.copy_to(arr2, stream)'''

            '''depth = 0
            width, height = 1280, 720
            n_channel = 4
            shape = (height, n_channel * width)
            # kind = cp.cuda.runtime.cudaChannelFormatKindFloat
            kind = cp.cuda.runtime.cudaChannelFormatKindUnsigned
            arr = cp.random.randint(100, size=shape).astype(self.frame.dtype)
            arr2 = cp.zeros_like(arr)
            assert arr.flags['C_CONTIGUOUS']
            assert arr2.flags['C_CONTIGUOUS']
            # create a CUDA array
            ch_bits = [0, 0, 0, 0]
            for i in range(n_channel):
                ch_bits[i] = arr.dtype.itemsize * 8
            # unpacking arguments using *ch_bits is not supported before PY35...
            ch = cp.cuda.texture.ChannelFormatDescriptor(ch_bits[0], ch_bits[1], ch_bits[2], ch_bits[3], kind)
            cu_arr = cp.cuda.texture.CUDAarray(ch, width, height, depth)

            # cp_arr = cp_arr.reshape((cp_arr.shape[0], cp_arr.shape[1] * cp_arr.shape[2]))
            # ca_arr = cp.cuda.texture.CUDAarray(
            #     cp.cuda.texture.ChannelFormatDescriptor(8, 8, 8, 8, cp.cuda.runtime.cudaChannelFormatKindUnsigned),
            #     cp_arr.shape[1],
            #     cp_arr.shape[0],
            #     0,
            #     # cp.cuda.runtime.cudaArrayDefault
            # )

            stream = cp.cuda.Stream()
            # copy from input to CUDA array, and back to output
            cu_arr.copy_from(arr, stream)
            # ca_arr.copy_from(arr, stream)
            # cu_arr.copy_from(cp_arr, stream)
            cu_arr.copy_to(arr2, stream)

            # check input and output are identical
            if stream is not None:
                dev.synchronize()
            assert (arr == arr2).all()'''

            '''gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            # cudaGraphicsSubResourceGetMappedArray( & array, resource, 0, 0);
            # cuda_buf = pycuda.gl.RegisteredImage(int(self.texture), gl.GL_TEXTURE_2D)
            cuda_buf = pycuda.gl.RegisteredImage(int(self.buffer), gl.GL_TEXTURE_2D)
            # pycuda.gl.RegisteredImage(int(self.texture), pycuda.gl.RegisteredImage.GL_TEXTURE_2D, flags=pycuda.gl.graphics_map_flags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
            # cuda_buf = pycuda.gl.RegisteredBuffer(int(self.buffer))  # , pycuda.gl.graphics_map_flags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
            # cuda_buf_mapping = cuda_buf.map()
            cuda_buf_mapping = cuda_buf.map()
            d_buf_ptr, d_buf_size = cuda_buf_mapping.device_ptr_and_size()
            # memory = cp.cuda.Memory(2048000000)
            # ptr = cp.cuda.MemoryPointer(memory, 0)
            # mem = cp.cuda.memory.UnownedMemory(d_buf_ptr, d_buf_size, self)
            # memptr = cp.cuda.memory.MemoryPointer(mem, 0)
            # cp_arr = cp.ndarray(shape=(cap_height, cap_width, 4), memptr=memptr)
            # cp_arr[0, 0, 0] = 100
            # cp_arr = None
            # memptr = None
            # mem = None
            # np_arr = cp.asnumpy(cp_arr)
            # arr = cuda_buf_mapping.array(0, 0)
            cuda_buf_mapping.unmap()
            cuda_buf.unregister()
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)'''

            # cuda_pbo_resource = pycuda.gl.RegisteredBuffer(int(pbo_buffer), pycuda.gl.graphics_map_flags.WRITE_DISCARD)
            # cp.cuda..driver.texRefSetArray(self.frame, cu_arr)
            # pycuda.driver.TextureReference.set_array()
            # cuTexRefSetArray

            # printing
            # gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.buffer)
            # gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, cap_width, cap_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

            #gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.frame)
            rc1 = sys.getrefcount(self.frame)
            # gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.frame)
            # gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, cap_width, cap_height, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.frame)
            rc2 = sys.getrefcount(self.frame)
            self.frame = None
            #print('teximage2d ', self.frame.flags['C_CONTIGUOUS'], rc1, rc2)

        time_start_post = round(time.monotonic() * 1000)
        gl.glColor3f(1, 1, 1)

        # draw a quad
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(0, 1)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(0, 0)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(1, 0)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(1, 1)
        gl.glEnd()

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0) #my
        gl.glDisable(gl.GL_TEXTURE_2D)

        time_taken_post = round(time.monotonic() * 1000) - time_start_post

        # gl.gXSwapIntervalEXT()
        # swap the front and back buffers so that the texture is visible
        time_start_swap = round(time.monotonic() * 1000)
        self.SwapBuffers()
        time_taken_swap = round(time.monotonic() * 1000) - time_start_swap

        time_taken_full = round(time.monotonic() * 1000) - time_start_full
        global test_cnt, test_acc
        test_cnt += 1
        test_acc += time_taken_full
        print("total", time_taken_full,
              "rtotal", round(test_acc / test_cnt),
              "prep", time_taken_prep,
              "action", action_time_taken,
              "tex", time_taken_tex,
              "post", time_taken_post,
              "swap", time_taken_swap
              )

    def getNormalFrameThread(self):
        while not self.closeThreads:
            time.sleep(0.010)
            self.UpdateUI(None)

    def UpdateUI(self, evt):
        if self.frame is not None:
            return
        # ret, frame = vcap.read()
        # ret, frame = True, np.zeros((1080, 1920, 4), dtype=np.uint8)
        ret, frame = True, (np.random.rand(cap_height, cap_width, 3) * 255).astype('B')
        if ret:
            #self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.flipud(frame)
            self.frame = np.dstack((frame, np.zeros(frame.shape[:-1], dtype=frame.dtype)))
            #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2BGRA)
            '''gl.glDeleteTextures(1, self.texture)
            self.texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 1920, 1080, 0, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.frame)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) #my'''
            wx.CallAfter(self.parent.Refresh)

    def OnClose(self, event):
        self.Destroy()

    def onDestroy(self, event=None):
        ## clean up resources as needed here
        del self.cp_arr
        self.cp_arr = None
        del self.cp_memptr
        self.cp_memptr = None
        del self.cp_mem
        self.cp_mem = None
        self.cuda_buf_mapping.unmap()
        del self.cuda_buf_mapping
        self.cuda_buf_mapping = None
        self.cuda_buf.unregister()
        del self.cuda_buf
        self.cuda_buf = None
        event.Skip()

def run():
    app = wx.App()
    fr = wx.Frame(None, size=(512, 512), title='wxPython texture demo')
    canv = Canvas(fr)
    fr.Show()
    app.MainLoop()

if __name__ == "__main__":
    run()