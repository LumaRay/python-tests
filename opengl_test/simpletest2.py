import sys
import math
import time

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

import cv2

img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
img_data = cv2.imread(img_path)
# img_data = cv2.flip(img_data, 0)

texture = glGenTextures(1)

global_timer = None

def init():
    global image, texName
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    # glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glShadeModel(GL_FLAT)
    glEnable(GL_DEPTH_TEST)

    global texture
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

def display():
    action_start = round(time.monotonic() * 1000)
    #global texName
    global texture

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1, 1, 0, 0, 1)
    # gluPerspective(60.0, w/h, 1.0, 30.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # glTranslatef(0.0, 0.0, -3.6)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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

    # glFinish()

    glBindTexture(GL_TEXTURE_2D, texture)
    tex_upload_start = round(time.monotonic() * 1000)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_data.shape[1], img_data.shape[0], GL_BGR, GL_UNSIGNED_BYTE, img_data)
    # glFinish()
    tex_upload_time_taken = round(time.monotonic() * 1000) - tex_upload_start
    glBindTexture(GL_TEXTURE_2D, 0)

    # glutSwapBuffers()

    global global_timer
    global_fps = 0
    if global_timer is not None:
        global_fps = round(1 / (time.monotonic() - global_timer))
    global_timer = time.monotonic()
    action_time_taken = round(time.monotonic() * 1000) - action_start
    print("global_fps", global_fps, "action_time_taken", action_time_taken, "ms", "tex_upload_time_taken", tex_upload_time_taken, "ms")

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
# glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
glutInitWindowSize(500, 500)
glutInitWindowPosition(100, 100)
glutCreateWindow('texture')
init()
glutDisplayFunc(display)
glutIdleFunc(idle)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboard)
glutMainLoop()

