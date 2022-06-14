import sys
import threading
import time

import numpy as np
from scipy.misc import ascent
# https://pixnio.com/photos/people
import OpenGL.GL as gl
import wx
from wx.glcanvas import GLCanvas

import cv2

# sudo systemctl restart nvargus-daemon
#vcap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink").format(1920, 1080), cv2.CAP_GSTREAMER)

#vcap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True").format(1920, 1080), cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
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

class Canvas(GLCanvas):

    def __init__(self, parent):
        """create the canvas """
        super(Canvas, self).__init__(parent)
        self.texture = None
        self.parent = parent

        self.context = wx.glcanvas.GLContext(self)

        # execute self.onPaint whenever the parent frame is repainted
        self.Bind(wx.EVT_PAINT, self.onPaint)

        '''self.parent.timerUpdateUI = wx.Timer(self.parent)
        self.parent.Bind(wx.EVT_TIMER, self.UpdateUI)
        self.parent.timerUpdateUI.Start(1000. / 1)'''

        self.closeThreads = False
        self.hGetNormalFrameThread = threading.Thread(target=self.getNormalFrameThread)  # , args=(self,))
        self.hGetNormalFrameThread.start()

        self.frame = None

    def initTexture(self):
        """init the texture - this has to happen after an OpenGL context
        has been created
        """

        # make the OpenGL context associated with this canvas the current one
        self.SetCurrent(self.context)

        #self.data = np.uint8(np.flipud(ascent()))
        #self.data = np.zeros((1080, 1920), dtype=np.uint8)
        self.data = np.zeros((1080, 1920, 4), dtype=np.uint8)
        '''img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
        self.data = cv2.imread(img_path)
        self.data = np.flipud(self.data)
        self.data = np.dstack((self.data, np.zeros(self.data.shape[:-1])))
        # self.data = np.dstack((self.data, np.zeros((self.data.shape[:-1]) + (1,))))'''
        ret, frame = vcap.read()
        if ret:
            frame = np.flipud(frame)
            self.data = np.dstack((frame, np.zeros(frame.shape[:-1])))
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
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.data.shape[1], self.data.shape[0], 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.data)

        #self.data[30:40, 30:40] = 255

    def onPaint(self, event):
        """called when window is repainted """
        # make sure we have a texture to draw
        if not self.texture:
            self.initTexture()
        self.onDraw()

    def onDraw(self):
        """draw function """

        # make the OpenGL context associated with this canvas the current one
        self.SetCurrent(self.context)

        # set the viewport and projection
        w, h = self.GetSize()
        gl.glViewport(0, 0, w, h)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, 0, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # enable textures, bind to our texture
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        if self.frame is not None:
            #gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.frame)
            rc1 = sys.getrefcount(self.frame)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.frame)
            rc2 = sys.getrefcount(self.frame)
            self.frame = None
            #print('teximage2d ', self.frame.flags['C_CONTIGUOUS'], rc1, rc2)

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

        # swap the front and back buffers so that the texture is visible
        self.SwapBuffers()

    def getNormalFrameThread(self):
        while not self.closeThreads:
            time.sleep(0.040)
            self.UpdateUI(None)

    def UpdateUI(self, evt):
        if self.frame is not None:
            return
        ret, frame = vcap.read()
        if ret:
            #self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.flipud(frame)
            self.frame = np.dstack((frame, np.zeros(frame.shape[:-1])))
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

def run():
    app = wx.App()
    fr = wx.Frame(None, size=(512, 512), title='wxPython texture demo')
    canv = Canvas(fr)
    fr.Show()
    app.MainLoop()

if __name__ == "__main__":
    run()