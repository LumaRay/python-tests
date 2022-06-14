import math
import time

import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import cv2

img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
img_data = cv2.imread(img_path)
img_data = cv2.flip(img_data, 0)

def main():
    pg.init()
    # display = (1680, 1050)
    display = (500, 500)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)

    # glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    # glShadeModel(GL_FLAT)
    # glEnable(GL_DEPTH_TEST)

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    # glTranslatef(0.0, 0.0, -5)
    # glTranslatef(1.0, 0.0, -3)

    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_BGR, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        action_start = round(time.monotonic() * 1000)

        window_size = pg.display.get_window_size()
        glViewport(0, 0, window_size[0], window_size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 1, 0, 0, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT)

        # glRotatef(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_TEXTURE_2D)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glBindTexture(GL_TEXTURE_2D, texture)
        glBegin(GL_QUADS)
        # glTexCoord2f(0, 0)
        # glVertex3f(-2, -1, 0)
        # glTexCoord2f(0, 1)
        # glVertex3f(-2, 1, 0)
        # glTexCoord2f(1, 1)
        # glVertex3f(0, 1, 0)
        # glTexCoord2f(1, 0)
        # glVertex3f(0, -1, 0)
        glTexCoord2f(0, 1)
        glVertex2f(0, 1)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(1, 0)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glEnd()
        tex_upload_start = round(time.monotonic() * 1000)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1920, 1080, GL_BGR, GL_UNSIGNED_BYTE, img_data)
        tex_upload_time_taken = round(time.monotonic() * 1000) - tex_upload_start
        glBindTexture(GL_TEXTURE_2D, 0)
        glFlush()
        glDisable(GL_TEXTURE_2D)

        pg.display.flip()

        action_time_taken = round(time.monotonic() * 1000) - action_start
        print("action_time_taken", action_time_taken, "ms", "tex_upload_time_taken", tex_upload_time_taken, "ms")

        pg.time.wait(10)

if __name__ == "__main__":
    main()
