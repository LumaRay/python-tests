import math
import time

import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import cv2

img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
img_data = cv2.imread(img_path)
img_data = cv2.flip(img_data, 0)


cubeVertices = ((1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1))
cubeEdges = ((0,1),(0,3),(0,4),(1,2),(1,7),(2,5),(2,3),(3,6),(4,6),(4,7),(5,6),(5,7))
cubeQuads = ((0,3,6,4),(2,5,6,3),(1,2,5,7),(1,0,4,7),(7,4,6,5),(2,3,0,1))

def wireCube():
    glBegin(GL_LINES)
    for cubeEdge in cubeEdges:
        for cubeVertex in cubeEdge:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()

def solidCube():
    glBegin(GL_QUADS)
    for cubeQuad in cubeQuads:
        for cubeVertex in cubeQuad:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()

def videoTexture(texture):
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
    glVertex3f(1 + math.sqrt(2), 1, -math.sqrt(2))
    glTexCoord2f(1, 0)
    glVertex3f(1 + math.sqrt(2), -1, -math.sqrt(2))
    glEnd()
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1920, 1080, GL_BGR, GL_UNSIGNED_BYTE, img_data)
    glBindTexture(GL_TEXTURE_2D, 0)
    glFlush()
    glDisable(GL_TEXTURE_2D)

def main():
    pg.init()
    display = (1680, 1050)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

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

        glRotatef(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # solidCube()
        #wireCube()
        videoTexture(texture)
        pg.display.flip()

        action_time_taken = round(time.monotonic() * 1000) - action_start
        print("action_time_taken", action_time_taken, "ms")

        pg.time.wait(10)

if __name__ == "__main__":
    main()