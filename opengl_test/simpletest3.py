import sys
import math
import time

# https://www.transmissionzero.co.uk/software/freeglut-devel/
# rename freeglut.dll to freeglut64.vc14 (if you have a 32-bit interpreter, rename to freeglut32.vc14)
# put freeglut64.vc14 in system32

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

import cv2

import numpy as np

# img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
# img_data = cv2.imread(img_path)
# img_data = cv2.flip(img_data, 0)
img_data = np.random.random((1080, 1920, 3)).astype(np.uint8)

glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE)

texture = glGenTextures(1)
#
# buffer = glGenBuffers(1)
# # glBindBuffer(GL_PIXEL_PACK_BUFFER, self.buffer)
# glBindBuffer(GL_ARRAY_BUFFER, buffer)
# # glBufferData(GL_PIXEL_PACK_BUFFER, cap_width * cap_height * 3, None, gl.GL_DYNAMIC_DRAW)
# glBufferData(GL_ARRAY_BUFFER, img_data.shape[1] * img_data.shape[0] * 4, None, GL_DYNAMIC_DRAW)
# glBindBuffer(GL_ARRAY_BUFFER, 0)

def init():
    # glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
    glShadeModel(GL_FLAT)
    glEnable(GL_DEPTH_TEST)

    global texture
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, None)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img_data.shape[1], img_data.shape[0], 0, GL_BGRA, GL_UNSIGNED_BYTE, None)

    glBindTexture(GL_TEXTURE_2D, 0)

def display():
    action_start = round(time.monotonic() * 1000)
    #global texName
    global texture, buffer

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1, 1, 0, 0, 1)
    # gluPerspective(60.0, w/h, 1.0, 30.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # glTranslatef(0.0, 0.0, -3.6)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # glBindBuffer(GL_ARRAY_BUFFER, buffer)
    buf_upload_start = round(time.monotonic() * 1000)
    # glBufferSubData(GL_ARRAY_BUFFER, 0, img_data.shape[1] * img_data.shape[0] * 3, img_data)
    buf_upload_time_taken = round(time.monotonic() * 1000) - buf_upload_start
    # glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer)
    glBindTexture(GL_TEXTURE_2D, texture)
    tex_copy_start = round(time.monotonic() * 1000)
    # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_data.shape[1], img_data.shape[0], GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_data.shape[1], img_data.shape[0], GL_BGRA, GL_UNSIGNED_BYTE, None)
    # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_data.shape[1], img_data.shape[0], GL_BGR, GL_UNSIGNED_BYTE, img_data)
    tex_copy_time_taken = round(time.monotonic() * 1000) - tex_copy_start
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)


    glEnable(GL_TEXTURE_2D)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 1)
    glTexCoord2f(0, 0)
    glVertex2f(0, 0)
    glTexCoord2f(1, 0)
    glVertex2f(1, 0)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glEnd()
    glFlush()
    glDisable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

    action_time_taken = round(time.monotonic() * 1000) - action_start
    print("action", action_time_taken, "buf_upload", buf_upload_time_taken, "tex_copy", tex_copy_time_taken)

def reshape(w, h):
    glViewport(0, 0, w, h)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # # gluPerspective(60.0, w/h, 1.0, 30.0)
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
    # # glTranslatef(0.0, 0.0, -3.6)

def idle():
    display()
    pass

def keyboard(key, x, y):
    pass

glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
glutInitWindowSize(500, 500)
glutInitWindowPosition(100, 100)
# glutCreateWindow('texture')
glutCreateWindow(b'texture')
init()
glutDisplayFunc(display)
glutIdleFunc(idle)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboard)
glutMainLoop()

