import sys
import math

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

def init():
    global image, texName
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glShadeModel(GL_FLAT)
    glEnable(GL_DEPTH_TEST)

    import cv2
    # import Image, numpy
    # img = Image.open('flagEn.bmp') # .jpg, .bmp, etc. also work
    # img_data = numpy.array(list(img.getdata()), numpy.int8)
    img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
    img_data = cv2.imread(img_path)
    img_data = cv2.flip(img_data, 0)

    global texture
    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, img_data)
    glBindTexture(GL_TEXTURE_2D, 0)

def display():
    #global texName
    global texture
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(-2, -1, 0)
    glTexCoord2f(0, 1)
    glVertex3f(-2, 1, 0)
    glTexCoord2f(1, 1)
    glVertex3f(0, 1, 0)
    glTexCoord2f(1, 0)
    glVertex3f(0, -1, 0)
    glTexCoord2f(0, 0)
    glVertex3f(1, -1, 0)
    glTexCoord2f(0, 1)
    glVertex3f(1, 1, 0)
    glTexCoord2f(1, 1)
    glVertex3f(1+math.sqrt(2), 1, -math.sqrt(2))
    glTexCoord2f(1, 0)
    glVertex3f(1+math.sqrt(2), -1, -math.sqrt(2))
    glEnd()
    glFlush()
    glDisable(GL_TEXTURE_2D)

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, w/h, 1.0, 30.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -3.6)

def idle():
    dfdf = 5
    dfdf = dfdf
    pass

def keyboard(key, x, y):
    pass

glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
glutInitWindowSize(500, 500)
glutInitWindowPosition(100, 100)
glutCreateWindow('texture')
init()
glutDisplayFunc(display)
glutIdleFunc(idle)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboard)
glutMainLoop()

